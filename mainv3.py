# mainv3.py
import multiprocessing
import pandas as pd
import numpy as np
import os
import time
import random
import keyboard
import optuna
import random
from pathlib import Path
from indicators import add_rsi, add_wave_trend, add_trendline_features, add_atr, add_temporal_features

from data_loader import discover_data_files, fetch_and_cache_market_caps, preload_all_data
from backtester import optimize_timeframe_group, init_backtester_worker, run_simulation
from analysis import analyze_by_market_cap
from ai_manager import TradeManagerEnv as EntryEnv, train_ai_manager as train_entry_manager
from exit_manager import ExitManagerEnv as ExitEnv, train_exit_manager

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Definicje funkcji (pozostają bez zmian) ---

def get_combined_data_for_training(params, files_to_use, data_store, target_timeframe=None):
    """
    Przygotowuje i łączy dane do treningu, dodając wszystkie potrzebne wskaźniki.
    """
    
    MIN_BLOCK_SIZE_FOR_INDICATORS = 150 
    processed_df_list = []
    all_dfs_raw = []
    
    if target_timeframe:
        print(f"  -> Przygotowywanie danych treningowych TYLKO dla interwału: {target_timeframe}...")
        for pair, timeframes in files_to_use.items():
            if target_timeframe in timeframes and \
               pair in data_store and \
               target_timeframe in data_store[pair]:
                
                df_to_add = data_store[pair][target_timeframe].copy()
                if len(df_to_add) >= MIN_BLOCK_SIZE_FOR_INDICATORS:
                    all_dfs_raw.append(df_to_add)
                else:
                    print(f"  -> Pomijanie {pair} {target_timeframe} (zbyt mało danych: {len(df_to_add)})")
    else:
        print("  -> Tryb 'Wszystkie interwały'. Aktywowanie balansowania danych...")
        for pair, timeframes in files_to_use.items():
            for tf in timeframes:
                if pair in data_store and tf in data_store[pair]:
                    if len(data_store[pair][tf]) >= MIN_BLOCK_SIZE_FOR_INDICATORS:
                        all_dfs_raw.append(data_store[pair][tf].copy())

    if not all_dfs_raw:
        print("  -> BŁĄD: Nie znaleziono wystarczających danych do połączenia.")
        return pd.DataFrame()

    min_len = min(len(df) for df in all_dfs_raw)
    block_size = min_len
    
    if not target_timeframe: # Stosuj balansowanie tylko w trybie "Wszystkie interwały"
        print(f"  -> Balansowanie do najkrótszego zbioru: {block_size} świec (na interwał/parę).")

    print("  -> Przetwarzanie bloków danych (Obliczanie wskaźników PRZED łączeniem)...")
    
    for df_original in all_dfs_raw:
        df_block = None
        if not target_timeframe: # Logika balansowania
            if len(df_original) > block_size:
                max_start_index = len(df_original) - block_size
                start_index = random.randint(0, max_start_index)
                df_block = df_original.iloc[start_index : start_index + block_size].copy()
            else:
                df_block = df_original.copy()
        else: # Tryb jednego interwału - bierzemy wszystko
            df_block = df_original.copy()

        df_block = add_temporal_features(df_block)
        df_block = add_rsi(df_block, params.get("RSI_PERIOD", 14))
        df_block = add_wave_trend(df_block, params.get("WT_N1", 10), params.get("WT_N2", 21))
        df_block['ema_fast'] = df_block['close'].ewm(span=params["EMA_FAST"]).mean()
        df_block['ema_slow'] = df_block['close'].ewm(span=params["EMA_SLOW"]).mean()
        df_block = add_atr(df_block, period=14) 
        df_block['price_change'] = df_block['close'].pct_change()
        df_block['atr_norm'] = df_block['atr'] / df_block['close']
        df_block['signal_ema'] = df_block['close'].ewm(span=50).mean()
        df_block = add_trendline_features(df_block, lookback=72)
        
        df_block.ffill(inplace=True); df_block.bfill(inplace=True)
        processed_df_list.append(df_block)
    
    if not processed_df_list:
        print("  -> BŁĄD: Nie znaleziono danych do połączenia (po balansowaniu).")
        return pd.DataFrame()

    combined_df = pd.concat(processed_df_list)
    combined_df.sort_index(inplace=True)
    
    print(f"  -> Połączone dane mają teraz {len(combined_df)} wierszy.")

    required_columns = ['close', 'ema_fast', 'ema_slow', 'RSI', 'WT1', 'WT2', 
                        'res_slope_norm', 'day_sin', 'hour_sin']
    
    if not all(col in combined_df.columns for col in required_columns):
        print("  -> OSTRZEŻENIE: Brak wymaganych kolumn po obliczeniu wskaźników.")
        return pd.DataFrame()

    cleaned_df = combined_df.dropna(subset=required_columns)

    if cleaned_df.empty:
        print("  -> KRYTYCZNY BŁĄD: Zbiór danych treningowych jest pusty PO czyszczeniu.")

    print("  -> ✅ Dane treningowe gotowe.")
    return cleaned_df

def main():
    # --- Konfiguracja i ładowanie danych (bez zmian) ---
    files_structure = discover_data_files(); all_tickers = list(files_structure.keys()); market_caps = fetch_and_cache_market_caps(all_tickers)
    
    sorted_tickers = sorted(market_caps, key=market_caps.get, reverse=True)

    # --- NOWA LOGIKA WYBORU GRUP ---
    print(f"Łącznie posortowanych tickerów: {len(sorted_tickers)}")

    # Grupa Treningowa: TOP 1-100 (lub mniej, jeśli nie ma wystarczająco dużo)
    num_training = 100
    training_tickers = sorted_tickers[:num_training]

    # Grupa Testowa: Miejsca 101-200 (zamiast losowych)
    num_testing = 100
    testing_tickers = sorted_tickers[num_training : num_training + num_testing]

    if not testing_tickers:
        print("OSTRZEŻENIE: Brak tickerów w grupie testowej (101-200). Używam reszty tickerów.")
        testing_tickers = sorted_tickers[num_training:]
    # --- KONIEC NOWEJ LOGIKI ---
        
    print("\n" + "="*50); print(f"📊 Podział na grupy:"); print(f"  -> Grupa Treningowa (TOP {num_training} MCap): {len(training_tickers)} par"); print(f"  -> Grupa Testowa (Losowa próbka): {len(testing_tickers)} par"); print("="*50)
    
    tickers_to_load = list(set(training_tickers + testing_tickers)); files_structure_to_load = {k: v for k, v in files_structure.items() if k in tickers_to_load}
    print(f"\nŁącznie zostanie wczytanych {len(tickers_to_load)} par (treningowe + testowe)."); data_store = preload_all_data(files_structure_to_load)

    base_params = {"SL_PCT": 0.05, "TRADE_DIRECTION": "both", "EMA_FAST": 12, "EMA_SLOW": 26, "RSI_PERIOD": 14, "WT_N1": 10, "WT_N2": 21, "USE_EMA": False}
    training_files_structure = {k: v for k, v in files_structure.items() if k in training_tickers}
    
    models_dir = SCRIPT_DIR / "Models"; models_dir.mkdir(exist_ok=True)
    
    # Nazwy modeli "Generalist"
    model_suffix = "_combined"
    entry_model_path = models_dir / f"ai_entry_strategist_model{model_suffix}.zip"
    exit_model_path = models_dir / f"ai_exit_manager_model{model_suffix}.zip"
    
    entry_model, exit_model = None, None
    
    # --- FAZA 1: Trening Modeli Ogólnych (Generalist) ---
    print("\n" + "="*50); print("🤖 FAZA 1: Trening Modeli Ogólnych (Generalist)..."); print("="*50 + "\n")
    
    # Używamy TRAIN_ON_SINGLE_TIMEFRAME = None, aby trenować na wszystkich interwałach
    training_data = get_combined_data_for_training(base_params, 
                                                 training_files_structure, 
                                                 data_store, 
                                                 target_timeframe=None)
    
    if training_data.empty: print("Krytyczny błąd: Brak danych do treningu."); return
    
    # Tworzymy env, które będą używane do ładowania modeli
    temp_entry_env = DummyVecEnv([lambda: EntryEnv(training_data, base_params)])
    temp_exit_env = DummyVecEnv([lambda: ExitEnv(training_data, base_params)])
    
    if os.path.exists(entry_model_path): 
        print(f"✅ Znaleziono model wejścia (Generalist). Ładowanie..."); 
        entry_model = PPO.load(entry_model_path, env=temp_entry_env)
    if entry_model is None: 
        entry_model = train_entry_manager(training_data, base_params, timesteps=50000, save_path=entry_model_path)
    
    if os.path.exists(exit_model_path): 
        print(f"✅ Znaleziono model wyjścia (Generalist). Ładowanie..."); 
        exit_model = PPO.load(exit_model_path, env=temp_exit_env)
    if exit_model is None: 
        exit_model = train_exit_manager(training_data, base_params, timesteps=75000, save_path=exit_model_path)
    
    if not all([entry_model, exit_model]): print("Krytyczny błąd: Nie udało się przygotować modeli AI."); return

    all_timeframes = sorted(list(set(tf for tfs in files_structure.values() for tf in tfs)))
    
    # Przekazujemy załadowane modele "Generalist" do Optuny
    init_backtester_worker(data_store)
    
    # --- FAZA 2: Optymalizacja Parametrów (Optuna) ---
    print("\n" + "="*50); print("🚀 FAZA 2: Rozpoczynanie pętli optymalizacji (używa modeli 'Generalist')."); print("💡 Wciśnij 'q', aby zakończyć i przejść do dostrajania (Fazy 3)."); print("="*50 + "\n")
    
    cycle_num = 1
    try:
        while True:
            print(f"\n--- Cykl {cycle_num} ---")
            for timeframe in all_timeframes:
                if keyboard.is_pressed('q'): raise KeyboardInterrupt
                print(f"\n--- Interwał: {timeframe} ---")
                pairs_for_timeframe = [p for p, tfs in files_structure.items() if timeframe in tfs and p in testing_tickers]
                if not pairs_for_timeframe: 
                    print(f"Brak par z losowej grupy testowej dla interwału {timeframe}.")
                    continue
                
                # Optuna używa modeli "Generalist"
                optimize_timeframe_group(entry_model, exit_model, pairs_for_timeframe, timeframe, n_trials=10, storage_dir=SCRIPT_DIR)
            cycle_num += 1; time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nPrzerwano pętlę optymalizacji. Przechodzenie do Fazy 3: Dostrajanie...")
    
    # --- FAZA 3: Dostrajanie (Fine-Tuning) Modeli ---
    print("\n" + "="*50); print("🤖 FAZA 3: Rozpoczynanie dostrajania modeli specjalistycznych..."); print("="*50 + "\n")
    
    # Definiujemy, jak bardzo "dostroić" modele
    FINE_TUNE_TIMESTEPS_ENTRY = 25000
    FINE_TUNE_TIMESTEPS_EXIT = 35000 
    
    for timeframe in all_timeframes:
        print(f"\n--- Dostrajanie dla interwału: {timeframe} ---")
        
        try:
            # 1. Załaduj najlepsze parametry z Optuny
            study_name = f"study_{timeframe}_group_exit_manager"
            storage_name = f"sqlite:///{SCRIPT_DIR / study_name}.db"
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            best_params_from_study = study.best_params
            
            # Łączymy bazowe z najlepszymi (aby mieć pewność, że wszystkie klucze są)
            current_params = {**base_params, **best_params_from_study}

            print(f"  -> Znaleziono najlepsze parametry: {best_params_from_study}")

            # 2. Przygotuj specjalistyczne dane treningowe
            print(f"  -> Przygotowywanie specjalistycznych danych dla {timeframe}...")
            specialist_data = get_combined_data_for_training(
                current_params,
                training_files_structure, # Używamy danych treningowych
                data_store, 
                target_timeframe=timeframe # Kluczowa zmiana
            )
            
            if specialist_data.empty:
                print(f"  -> BŁĄD: Brak danych do dostrojenia dla {timeframe}. Pomijanie.")
                continue
            
            # 3. Dostrój i zapisz model WEJŚCIA
            print(f"  -> Dostrajam model WEJŚCIA (Generalist)...")
            # Używamy 'specialist_data' i 'current_params' do stworzenia env dla .learn()
            specialist_entry_env = DummyVecEnv([lambda: EntryEnv(specialist_data, current_params)])
            entry_model_generalist = PPO.load(entry_model_path, env=specialist_entry_env) # Ładujemy "Generalist"
            entry_model_generalist.learn(total_timesteps=FINE_TUNE_TIMESTEPS_ENTRY) # Trenujemy dalej
            
            specialist_entry_path = models_dir / f"ai_entry_strategist_model_specialist_{timeframe}.zip"
            entry_model_generalist.save(specialist_entry_path)
            print(f"  -> ✅ Zapisano model wejścia: {specialist_entry_path.name}")

            # 4. Dostrój i zapisz model WYJŚCIA
            print(f"  -> Dostrajam model WYJŚCIA (Generalist)...")
            specialist_exit_env = DummyVecEnv([lambda: ExitEnv(specialist_data, current_params)])
            exit_model_generalist = PPO.load(exit_model_path, env=specialist_exit_env) # Ładujemy "Generalist"
            exit_model_generalist.learn(total_timesteps=FINE_TUNE_TIMESTEPS_EXIT) # Trenujemy dalej

            specialist_exit_path = models_dir / f"ai_exit_manager_model_specialist_{timeframe}.zip"
            exit_model_generalist.save(specialist_exit_path)
            print(f"  -> ✅ Zapisano model wyjścia: {specialist_exit_path.name}")

        except Exception as e:
            print(f"❌ Błąd podczas dostrajania dla {timeframe}: {e}")

    # --- FAZA 4: Końcowa Analiza (z użyciem Modeli Specjalistycznych) ---
    print("\n" + "="*50); print("📊 FAZA 4: Końcowa analiza z użyciem modeli SPECJALISTYCZNYCH..."); print("="*50 + "\n")
    
    all_trades_list = []
    try:
        for timeframe in all_timeframes:
            pairs_for_timeframe = [p for p, tfs in files_structure.items() if timeframe in tfs and p in testing_tickers]
            if not pairs_for_timeframe: continue
            
            print(f"\n--- Analiza końcowa dla: {timeframe} ---")
            
            try:
                # 1. Załaduj najlepsze parametry (te same co w Fazie 3)
                study_name = f"study_{timeframe}_group_exit_manager"
                storage_name = f"sqlite:///{SCRIPT_DIR / study_name}.db"
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                best_params = study.best_params
                
                # 2. Załaduj modele SPECJALISTYCZNE
                specialist_entry_path = models_dir / f"ai_entry_strategist_model_specialist_{timeframe}.zip"
                specialist_exit_path = models_dir / f"ai_exit_manager_model_specialist_{timeframe}.zip"

                if not specialist_entry_path.exists() or not specialist_exit_path.exists():
                    print(f"  -> OSTRZEŻENIE: Brak modeli specjalistycznych dla {timeframe}. Pomijanie analizy.")
                    continue
                    
                print(f"  -> Ładowanie modeli specjalistycznych dla {timeframe}...")
                
                # Używamy 'temp_entry_env' i 'temp_exit_env' z Fazy 1 do ładowania.
                # Zapewniają one poprawny *kształt* przestrzeni.
                entry_model_specialist = PPO.load(specialist_entry_path, env=temp_entry_env)
                exit_model_specialist = PPO.load(specialist_exit_path, env=temp_exit_env)
                
                print(f"  -> Uruchamianie symulacji na grupie testowej (z modelami specjalistycznymi)...")
                
                # 3. Uruchom symulację na grupie TESTOWEJ z parą (Best Params + Specialist Model)
                for pair in pairs_for_timeframe:
                    sim_result = run_simulation(
                        best_params, 
                        pair, 
                        timeframe, 
                        entry_model_specialist,  # Użyj modelu specjalisty
                        exit_model_specialist    # Użyj modelu specjalisty
                    )
                    all_trades_list.extend(sim_result['trades'])
                    
            except Exception as e:
                print(f"Błąd wczytywania lub ponownego uruchamiania symulacji dla {timeframe}: {e}")
        
        if not all_trades_list:
            print("Nie zebrano żadnych transakcji do końcowej analizy.")
            return

        print(f"✅ Zebrano łącznie {len(all_trades_list)} transakcji do analizy.")
        
        results_df = pd.DataFrame(all_trades_list)
        results_df.dropna(subset=['exit_price', 'avg_price'], inplace=True)

        if results_df.empty:
            print("Brak zakończonych transakcji do analizy.")
            return

        results_df['PnL'] = 0.0
        longs = results_df['side'] == 'long'
        shorts = results_df['side'] == 'short'
        
        results_df.loc[longs, 'PnL'] = ((results_df['exit_price'] - results_df['avg_price']) / results_df['avg_price']) * 100
        results_df.loc[shorts, 'PnL'] = ((results_df['avg_price'] - results_df['exit_price']) / results_df['avg_price']) * 100
        
        # Przekazujemy pełny DataFrame (z pojedynczymi transakcjami) ORAZ market_caps
        analyze_by_market_cap(results_df, market_caps)

    except Exception as e:
        print(f"Krytyczny błąd w Fazie 4 (Analiza): {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
