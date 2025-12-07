import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_script_log.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

import multiprocessing
import pandas as pd
import numpy as np
import os
import random
import optuna
from pathlib import Path

# Importy
from indicators import add_rsi, add_wave_trend, add_trendline_features, add_atr, add_temporal_features
from data_loader import discover_data_files, fetch_and_cache_market_caps, preload_all_data
from backtester import optimize_timeframe_group, init_backtester_worker, run_simulation
from analysis import analyze_by_market_cap
from ai_manager import TradeManagerEnv as EntryEnv, train_ai_manager as train_entry_manager
from exit_manager import ExitManagerEnv as ExitEnv, train_exit_manager

# ZMIANA: Importujemy RecurrentPPO zamiast PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Funkcje pomocnicze (precompute & get_data) ---

def precompute_heavy_indicators(df):
    if 'day_sin' not in df.columns: df = add_temporal_features(df)
    if 'atr' not in df.columns:
        df = add_atr(df, period=14)
        df['atr_norm'] = df['atr'] / df['close']
    if 'price_change' not in df.columns:
        df['price_change'] = np.log(df['close'] / df['close'].shift(1))
    if 'adx' not in df.columns: df = add_trendline_features(df, lookback=72)
    if 'tp_calc' not in df.columns:
        df['tp_calc'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['tp_calc'] * df['volume']
        df['month'] = df.index.to_period('M')
    df.ffill(inplace=True); df.bfill(inplace=True)
    return df

def get_combined_data_for_training(params, files_to_use, data_store, target_timeframe=None):
    MIN_BLOCK_SIZE_FOR_INDICATORS = 150 
    processed_df_list = []
    candidates = []
    for pair, timeframes in files_to_use.items():
        if target_timeframe:
            if target_timeframe in timeframes and target_timeframe in data_store[pair]:
                candidates.append(data_store[pair][target_timeframe])
        else:
            for tf in timeframes:
                if tf in data_store[pair]: candidates.append(data_store[pair][tf])
    if not candidates: return pd.DataFrame()

    min_len = min(len(df) for df in candidates if len(df) >= MIN_BLOCK_SIZE_FOR_INDICATORS)
    block_size = min_len
    
    for df_original in candidates:
        if len(df_original) < MIN_BLOCK_SIZE_FOR_INDICATORS: continue
        if not target_timeframe and len(df_original) > block_size:
            max_start = len(df_original) - block_size
            start = random.randint(0, max_start)
            df_block = df_original.iloc[start : start + block_size].copy()
        else:
            df_block = df_original.copy()

        # Zabezpieczenie przed brakiem wskaźników statycznych
        if 'adx' not in df_block.columns: df_block = precompute_heavy_indicators(df_block)
        
        # Wskaźniki dynamiczne (zależne od params Optuny)
        df_block = add_rsi(df_block, params.get("RSI_PERIOD", 14))
        df_block = add_wave_trend(df_block, params.get("WT_N1", 10), params.get("WT_N2", 21))
        df_block['ema_fast'] = df_block['close'].ewm(span=params["EMA_FAST"]).mean()
        df_block['ema_slow'] = df_block['close'].ewm(span=params["EMA_SLOW"]).mean()
        df_block['price_change'] = df_block['close'].pct_change()
        df_block['signal_ema'] = df_block['close'].ewm(span=50).mean()
        
        df_block.ffill(inplace=True); df_block.bfill(inplace=True)
        processed_df_list.append(df_block)
    
    if not processed_df_list: return pd.DataFrame()
    combined_df = pd.concat(processed_df_list)
    combined_df.sort_index(inplace=True)
    required_columns = ['close', 'ema_fast', 'ema_slow', 'RSI', 'WT1', 'WT2', 'res_slope_norm', 'day_sin']
    return combined_df.dropna(subset=required_columns)

# --- GŁÓWNA PĘTLA ---

def main():
    logging.info("\n" + "="*80)
    logging.info(f"START SYSTEMU (wersja LSTM + Split): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80 + "\n")

    files_structure_raw = discover_data_files() 
    STABLECOINS = ['USDT', 'USDC', 'DAI', 'TUSD', 'BUSD', 'FDUSD', 'USDP', 'PYUSD', 'PAX', 'GUSD', 'USDD', 'FRAX']
    BLACKLISTED = STABLECOINS + ['WBTC', 'WETH', 'WBNB', 'WAVAX', 'WMATIC', 'WFTM', 'STETH']

    files_structure = {} 
    for ticker, timeframes in files_structure_raw.items():
        base = ticker.split('_')[0].upper()
        if base not in BLACKLISTED:
            files_structure[ticker] = timeframes
    
    all_tickers = list(files_structure.keys())
    market_caps = fetch_and_cache_market_caps(all_tickers)
    sorted_tickers = sorted(market_caps, key=market_caps.get, reverse=True)
    #sorted_tickers = sorted_tickers[:20] # Ograniczenie do 20 par
    
    # <<< Podział na 3 zbiory (Train, Validation, Test) >>>
    n_total = len(sorted_tickers)
    n_train = int(n_total * 0.60) # 60%
    n_val = int(n_total * 0.20)   # 20%
    # Reszta (20%) na test
    
    training_tickers = sorted_tickers[:n_train]
    validation_tickers = sorted_tickers[n_train : n_train + n_val]
    testing_tickers = sorted_tickers[n_train + n_val:]
    
    logging.info(f"📊 Podział danych: Train={len(training_tickers)}, Val={len(validation_tickers)}, Test={len(testing_tickers)}")

    tickers_to_load = list(set(training_tickers + validation_tickers + testing_tickers))
    files_to_load = {k: v for k, v in files_structure.items() if k in tickers_to_load}
    
    # Ładowanie i pre-kalkulacja
    data_store = preload_all_data(files_to_load)

    logging.info("⚡ Pre-kalkulacja wskaźników...")
    count = 0
    for pair in data_store:
        for tf in data_store[pair]:
            data_store[pair][tf] = precompute_heavy_indicators(data_store[pair][tf])
            count += 1
    logging.info(f"✅ Pre-kalkulacja gotowa.")

    base_params = {"SL_PCT": 0.05, "TRADE_DIRECTION": "both", "EMA_FAST": 12, "EMA_SLOW": 26, "RSI_PERIOD": 14, "WT_N1": 10, "WT_N2": 21, "USE_EMA": False}
    
    # Pliki dla treningu generalisty
    training_files = {k: v for k, v in files_structure.items() if k in training_tickers}
    
    models_dir = SCRIPT_DIR / "Models"
    models_dir.mkdir(exist_ok=True)
    
    entry_model_path = models_dir / "ai_entry_lstm_model.zip"
    exit_model_path = models_dir / "ai_exit_lstm_model.zip"
    
    # --- FAZA 1: Ciągły Trening (Incremental Learning) ---
    logging.info("🤖 FAZA 1: Trening / Douczanie Modeli LSTM...")
    
    # KROK 1: Pobranie danych treningowych
    training_data = get_combined_data_for_training(base_params, training_files, data_store)
    if training_data.empty: 
        logging.error("Brak danych treningowych."); return

    # KROK 2: Inicjalizacja środowisk (MUSI BYĆ PRZED LOAD)
    temp_entry_env = DummyVecEnv([lambda: EntryEnv(training_data, base_params)])
    temp_exit_env = DummyVecEnv([lambda: ExitEnv(training_data, base_params)])

    # KROK 3: Ładowanie lub tworzenie modeli
    # --- 1. ENTRY MODEL (WEJŚCIE) ---
    if entry_model_path.exists():
        logging.info("🔄 Wczytywanie istniejącego modelu Entry... DOUCZANIE.")
        # Wczytujemy stary model do istniejącego env
        entry_model = RecurrentPPO.load(entry_model_path, env=temp_entry_env, device="cpu", ent_coef=0.05) 
    else:
        logging.info("✨ Tworzenie nowego modelu Entry od zera...")
        # Tworzymy nowy model
        entry_model = RecurrentPPO("MlpLstmPolicy", temp_entry_env, verbose=1, ent_coef=0.05, device="cpu")

    # TRENUJEMY (Niezależnie czy stary czy nowy)
    entry_model.learn(total_timesteps=5000) 
    entry_model.save(entry_model_path)
    logging.info("✅ Model Entry zapisany po treningu.")

    # --- 2. EXIT MODEL (WYJŚCIE) ---
    if exit_model_path.exists():
        logging.info("🔄 Wczytywanie istniejącego modelu Exit... DOUCZANIE.")
        exit_model = RecurrentPPO.load(exit_model_path, env=temp_exit_env, device="cpu", ent_coef=0.05)
    else:
        logging.info("✨ Tworzenie nowego modelu Exit od zera...")
        exit_model = RecurrentPPO("MlpLstmPolicy", temp_exit_env, verbose=1, ent_coef=0.05, device="cpu")

    exit_model.learn(total_timesteps=5000)
    exit_model.save(exit_model_path)
    logging.info("✅ Model Exit zapisany po treningu.")
    
    init_backtester_worker(data_store)

    # --- FAZA 2: Optymalizacja (Optuna) na zbiorze VALIDATION ---
    logging.info("🚀 FAZA 2: Optymalizacja parametrów (Zbiór Walidacyjny)...")
    all_timeframes = sorted(list(set(tf for tfs in files_structure.values() for tf in tfs)))
    
    try:
        for timeframe in all_timeframes:
            # Używamy validation_tickers!
            pairs = [p for p, tfs in files_structure.items() if timeframe in tfs and p in validation_tickers]
            if not pairs: continue
            
            logging.info(f"  -> Optymalizacja {timeframe} na {len(pairs)} parach walidacyjnych.")
            optimize_timeframe_group(entry_model, exit_model, pairs, timeframe, n_trials=20, storage_dir=SCRIPT_DIR)
            
    except KeyboardInterrupt:
        logging.warning("Przerwano optymalizację.")

    # --- FAZA 3: Analiza końcowa (Zbiór Testowy - Out of Sample) ---
    logging.info("📊 FAZA 3: Analiza końcowa (Zbiór Testowy)...")
    
    db_url = f"sqlite:///{SCRIPT_DIR}/optuna_studies.db"
    all_trades_list = []
    
    for timeframe in all_timeframes:
        # Używamy testing_tickers! To dane, których system nigdy nie widział.
        pairs = [p for p, tfs in files_structure.items() if timeframe in tfs and p in testing_tickers]
        if not pairs: continue
        
        logging.info(f"  -> Testowanie {timeframe} na {len(pairs)} parach testowych.")
        
        study_name = f"study_{timeframe}_group_exit_manager"
        try:
            study = optuna.load_study(study_name=study_name, storage=db_url)
            best_params = {**base_params, **study.best_params}
        except:
            logging.warning(f"Brak badania dla {timeframe}, używam domyślnych.")
            best_params = base_params

        # (Pominięto tutaj Fine-Tuning dla uproszczenia kodu, używamy Generalist + Best Params)
        
        for pair in pairs:
            res = run_simulation(best_params, pair, timeframe, entry_model, exit_model)
            all_trades_list.extend(res['trades'])

    if all_trades_list:
        results_df = pd.DataFrame(all_trades_list)
        if 'timeframe' in results_df.columns:
            results_df.rename(columns={'timeframe': 'Timeframe', 'pair': 'Pair'}, inplace=True)
        
        results_df.dropna(subset=['exit_price', 'avg_price'], inplace=True)
        
        results_df['PnL'] = 0.0
        longs = results_df['side'] == 'long'
        shorts = results_df['side'] == 'short'
        # Uwzględnienie prowizji w PnL procentowym
        fee_pct = 0.06
        results_df.loc[longs, 'PnL'] = (((results_df['exit_price'] - results_df['avg_price']) / results_df['avg_price']) * 100) - fee_pct
        results_df.loc[shorts, 'PnL'] = (((results_df['avg_price'] - results_df['exit_price']) / results_df['avg_price']) * 100) - fee_pct
        
        # Nowa funkcja analityczna z wykresem
        analyze_by_market_cap(results_df, market_caps)
    else:
        logging.warning("Brak transakcji do analizy.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
