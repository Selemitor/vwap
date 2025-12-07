# backtester.py
import pandas as pd
import numpy as np
import optuna
from functools import partial
from pathlib import Path
import logging
# Importujemy tylko lekkie wskaźniki, reszta jest już w danych!
from indicators import add_rsi, add_wave_trend 

GLOBAL_DATA_STORE = {}

def init_backtester_worker(data_store):
    """Inicjalizuje worker z danymi (które mają już policzone ciężkie wskaźniki)."""
    global GLOBAL_DATA_STORE
    GLOBAL_DATA_STORE = data_store

def run_simulation(params, pair, timeframe, entry_model, exit_model):
    """
    Symulacja handlu w trybie autonomicznym (AI sprawdza każdą świecę).
    """
    # 1. Pobranie danych
    df_original = GLOBAL_DATA_STORE.get(pair, {}).get(timeframe)
    if df_original is None or df_original.empty: 
        return {"total_pnl": 0, "trades": []}
    
    df = df_original.copy()
    
    # 2. Obliczenie wskaźników dynamicznych (zależnych od parametrów Optuny)
    df = add_rsi(df, params.get("RSI_PERIOD", 14))
    df = add_wave_trend(df, params.get("WT_N1", 10), params.get("WT_N2", 21))
    df['ema_fast'] = df['close'].ewm(span=params["EMA_FAST"]).mean()
    df['ema_slow'] = df['close'].ewm(span=params["EMA_SLOW"]).mean()
    
    # 3. Sprawdzenie wymaganych kolumn
    required_cols = ['adx', 'day_sin', 'price_change', 'ema_fast', 'ema_slow']
    if not all(col in df.columns for col in required_cols):
        return {"total_pnl": 0, "trades": []}

    # 4. Diagnostyka Danych (Czy dropna nie usuwa wszystkiego?)
    initial_len = len(df)
    # Usuwamy NaN powstałe przy obliczaniu wskaźników
    df.dropna(subset=['ema_fast', 'ema_slow', 'RSI', 'price_change'], inplace=True)
    final_len = len(df)
    
    # Debug print - pokazuje czy mamy dane (rzadko, żeby nie spamować)
    if np.random.rand() < 0.005: 
        print(f"DEBUG DATA [{pair}]: Start={initial_len}, Po czyszczeniu={final_len}")

    if df.empty: return {"total_pnl": 0, "trades": []}

    # --- SYMULACJA ---
    all_trades = []; position = None
    lookback_window = 10
    last_trade_pnl = 0.0 
    
    # Ustalenie startu (musi być miejsce na lookback)
    start_idx = max(lookback_window, 72)
    if len(df) <= start_idx: return {"total_pnl": 0, "trades": []}
    
    # Przechodzimy przez świece
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        idx = df.index[i]
        
        # --- 1. ZARZĄDZANIE OTWARTĄ POZYCJĄ ---
        if position:
            exit_signal = False; exit_price = None; reason = ""
            
            # A. Sztywne reguły (SL, TP, TSL)
            if position['tsl_active']:
                sl_pct = params['SL_PCT'] * position['sl_mult']
                if position['side'] == 'long':
                    position['tsl_peak'] = max(position['tsl_peak'], row['high'])
                    tsl_price = position['tsl_peak'] * (1 - sl_pct)
                    if row['low'] <= tsl_price: exit_price = tsl_price; reason = "TSL"
                else:
                    position['tsl_peak'] = min(position['tsl_peak'], row['low'])
                    tsl_price = position['tsl_peak'] * (1 + sl_pct)
                    if row['high'] >= tsl_price: exit_price = tsl_price; reason = "TSL"
            
            elif not position['tsl_active']:
                if position['side']=='long':
                    if row['low'] <= position['sl']: exit_price = position['sl']; reason = "SL"
                    elif row['high'] >= position['tp']: exit_price = position['tp']; reason = "TP"
                else:
                    if row['high'] >= position['sl']: exit_price = position['sl']; reason = "SL"
                    elif row['low'] <= position['tp']: exit_price = position['tp']; reason = "TP"
            
            # Wykonanie wyjścia sztywnego (SL/TP/TSL)
            if exit_price:
                position.update({'exit_time': idx, 'exit_reason': reason, 'exit_price': exit_price})
                all_trades.append(position); position = None
                
                # Obliczenie PnL dla pamięci AI
                pnl_raw = (exit_price - row['close']) / row['close'] if reason == "TP" else -params['SL_PCT']
                last_trade_pnl = np.clip(pnl_raw * 10, -1, 1)
                continue

            # B. AI Exit Manager (Decyzja co robić dalej)
            time_in_trade = (i - df.index.get_loc(position['entry_time']))
            unrealized_pnl = (row['close'] - position['avg_price']) * position['side_int'] / position['avg_price']
            
            # Budowa obserwacji dla Exit Managera
            obs_state = np.array([unrealized_pnl, time_in_trade / 24.0])
            obs_simple_ta = np.array([
                row.get('RSI', 50) / 100.0, row.get('WT1', 0) / 100.0, row.get('WT2', 0) / 100.0, 
                (row['ema_fast'] - row['close']) / row['close'], (row['ema_slow'] - row['close']) / row['close']
            ])
            obs_trend_ta = row[['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']].values.astype(float)
            obs_temporal = row[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']].values.astype(float)
            
            obs_exit = np.hstack([obs_state, obs_simple_ta, obs_trend_ta, obs_temporal])
            
            # Zabezpieczenie przed NaN/Inf w Exit Modelu
            obs_exit = np.nan_to_num(obs_exit)
            obs_exit = np.clip(obs_exit, -10.0, 10.0)
            obs_exit = obs_exit.astype(np.float32)
            
            action_exit, _ = exit_model.predict(obs_exit, deterministic=True)
            
            if action_exit == 1 and not position['breakeven_set']:
                # Action 1: BE
                position['sl'] = position['avg_price']; position['breakeven_set'] = True; position['tsl_active'] = False
            
            elif action_exit == 2:
                # Action 2: Close Now
                exit_price = row['close']
                position.update({'exit_time': idx, 'exit_reason': "AI_Close", 'exit_price': exit_price})
                all_trades.append(position)
                
                pnl_raw = (exit_price - position['avg_price']) / position['avg_price'] if position['side'] == 'long' else (position['avg_price'] - exit_price) / position['avg_price']
                last_trade_pnl = np.clip(pnl_raw * 10, -1, 1) 
                position = None 
                
            elif action_exit == 3 and not position['tsl_active'] and not position['breakeven_set']:
                # Action 3: TSL
                if unrealized_pnl > 0:
                    position['tsl_active'] = True
                    position['tsl_peak'] = row['high'] if position['side']=='long' else row['low']
            continue

        # --- 2. LOGIKA WEJŚCIA (AUTONOMICZNA) ---
        if not position:
            frame = df.iloc[i-lookback_window : i]
            
            # Przygotowanie danych wejściowych
            obs_original = frame[['price_change', 'ema_fast', 'ema_slow', 'RSI', 'WT1', 'WT2', 'atr_norm']].values
            obs_original[:, 1] = (frame['ema_fast'] - frame['close']) / frame['close']
            obs_original[:, 2] = (frame['ema_slow'] - frame['close']) / frame['close']
            
            obs_trendline = frame[['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']].values
            obs_temporal = frame[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']].values
            obs_last_pnl = np.full((lookback_window, 1), last_trade_pnl)
            
            # --- TWORZENIE OBSERWACJI (TUTAJ BYŁ BŁĄD) ---
            final_obs = np.hstack([obs_original, obs_trendline, obs_temporal, obs_last_pnl])
            
            # Zabezpieczenie przed błędami numerycznymi (NaN, Inf)
            final_obs = np.nan_to_num(final_obs)
            final_obs = np.clip(final_obs, -10.0, 10.0) # Przycinamy wartości ekstremalne
            final_obs = final_obs.astype(np.float32)
            
            # Pytamy AI o zdanie
            action, _ = entry_model.predict(final_obs, deterministic=True)
            signal_val, sl_mult, rr_mult = action
            
            side = None
            
            # <<< TRYB PRODUKCYJNY: Wyższy próg >>>
            # Wymagamy, aby AI było przekonane (np. > 0.3 lub < -0.3)
            # Jeśli ustawisz za wysoko (np. 0.8), transakcji będzie bardzo mało.
            threshold = 0.3
            
            if signal_val > threshold:
                side = 'long'
            elif signal_val < -threshold:
                side = 'short'
            
            if side:
                entry_p = row['close'] 
                dist = params['SL_PCT'] * sl_mult
                position = {
                    'pair': pair, 'timeframe': timeframe, 'side': side, 'side_int': 1 if side=='long' else -1,
                    'entry_time': idx, 'sl_mult': sl_mult, 'rr_mult': rr_mult,
                    'avg_price': entry_p, 'breakeven_set': False, 'tsl_active': False,
                    'tsl_peak': -np.inf if side=='long' else np.inf,
                    'sl': entry_p * (1 - dist) if side == 'long' else entry_p * (1 + dist),
                    'tp': entry_p * (1 + dist * rr_mult) if side == 'long' else entry_p * (1 - dist * rr_mult)
                }

    # --- OBLICZANIE WYNIKÓW KOŃCOWYCH ---
    FEE_AND_SLIPPAGE = 0.0006 
    total_pnl = 0
    for pos in all_trades:
        if 'exit_price' in pos:
            entry = pos['avg_price']; exit_p = pos['exit_price']
            raw_pnl = (exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry
            net_pnl = raw_pnl - FEE_AND_SLIPPAGE
            total_pnl += (net_pnl * 100)
            
    return {"total_pnl": total_pnl, "trades": all_trades}

def objective_function(trial, entry_model, exit_model, pairs_in_group, timeframe):
    params = {
        "SL_PCT": trial.suggest_float("SL_PCT", 0.01, 0.15, log=True),
        "TRADE_DIRECTION": trial.suggest_categorical("TRADE_DIRECTION", ["long", "short", "both"]),
        "EMA_FAST": trial.suggest_int("EMA_FAST", 5, 50),
        "EMA_SLOW": trial.suggest_int("EMA_SLOW", 60, 200),
        "RSI_PERIOD": trial.suggest_int("RSI_PERIOD", 7, 30),
        "WT_N1": trial.suggest_int("WT_N1", 8, 20),
        "WT_N2": trial.suggest_int("WT_N2", 18, 40),
        "USE_EMA": trial.suggest_categorical("USE_EMA", [True, False]),
    }
    if params["EMA_FAST"] >= params["EMA_SLOW"]: return -9999
    
    total_pnl = 0
    count = 0
    for pair in pairs_in_group:
        result = run_simulation(params, pair, timeframe, entry_model, exit_model)
        total_pnl += result["total_pnl"]
        if result['trades']: count += 1
    
    if count == 0: return 0
    return total_pnl / len(pairs_in_group)

def optimize_timeframe_group(entry_model, exit_model, pairs_in_group, timeframe, n_trials=50, storage_dir="."):
    db_path = Path(storage_dir) / "optuna_studies.db"
    storage_name = f"sqlite:///{db_path}"
    
    study_name = f"study_{timeframe}_group_exit_manager"
    
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    
    obj = partial(objective_function, entry_model=entry_model, exit_model=exit_model, pairs_in_group=pairs_in_group, timeframe=timeframe)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    
    return study.best_params, study.best_value
