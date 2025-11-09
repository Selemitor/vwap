# backtester.py
import pandas as pd
import numpy as np
import optuna
from functools import partial
from pathlib import Path

# <<< POPRAWKA: Importujemy nową funkcję >>>
from indicators import add_rsi, add_wave_trend, add_atr, add_trendline_features, add_temporal_features
from data_loader import preload_all_data

GLOBAL_DATA_STORE = {}

def init_backtester_worker(data_store):
    global GLOBAL_DATA_STORE
    GLOBAL_DATA_STORE = data_store

def run_simulation(params, pair, timeframe, entry_model, exit_model):
    """Uruchamia symulację z nowymi cechami (Czas) i logiką (TSL)."""
    df_original = GLOBAL_DATA_STORE.get(pair, {}).get(timeframe)
    if df_original is None or df_original.empty: return {"total_pnl": 0, "trades": []}
    
    df = df_original.copy()
    
    # --- Obliczanie wszystkich wskaźników ---
    df = add_rsi(df, params.get("RSI_PERIOD", 14))
    df = add_wave_trend(df, params.get("WT_N1", 10), params.get("WT_N2", 21))
    df['ema_fast'] = df['close'].ewm(span=params["EMA_FAST"]).mean()
    df['ema_slow'] = df['close'].ewm(span=params["EMA_SLOW"]).mean()
    df = add_atr(df, period=14)
    df['price_change'] = df['close'].pct_change()
    df['atr_norm'] = df['atr'] / df['close']
    df = add_trendline_features(df, lookback=72)
    # <<< POPRAWKA: Dodajemy cechy czasowe >>>
    df = add_temporal_features(df)
    
    if 'adx' not in df.columns or 'day_sin' not in df.columns:
        return {"total_pnl": 0, "trades": []} # Pomijamy, jeśli dane są niekompletne
        
    df.ffill(inplace=True); df.bfill(inplace=True)
    
    # --- Obliczanie VWAP ---
    df['tp_calc'] = (df['high'] + df['low'] + df['close']) / 3; df['tp_vol'] = df['tp_calc'] * df['volume']; df['month'] = df.index.to_period('M')
    monthly_levels = []
    for month, group in df.groupby('month'):
        if group['volume'].sum() == 0: continue
        vwap = group['tp_vol'].sum() / group['volume'].sum(); sigma = group['tp_calc'].std()
        levels = {'month': month, 'vwap': vwap, 'band_up_2.0': vwap + 2 * sigma, 'band_down_2.0': vwap - 2 * sigma}
        monthly_levels.append(levels)
    if not monthly_levels: return {"total_pnl": 0, "trades": []}
    levels_df = pd.DataFrame(monthly_levels).set_index('month'); df['prev_month'] = df['month'] - 1; df = df.join(levels_df, on='prev_month')

    all_trades = []; position = None; month_traded = set()
    lookback_window = 10
    last_trade_pnl = -1.0 

    for i in range(max(lookback_window, 72), len(df)):
        idx, row = df.index[i], df.iloc[i]
        if pd.isna(row['vwap']): continue
        
        # --- Logika zarządzania pozycją ---
        if position:
            # <<< POPRAWKA: Logika Trailing Stop Lossa (TSL) >>>
            if position['tsl_active']:
                sl_pct = params['SL_PCT'] * position['sl_mult'] # Używamy SL zdefiniowanego przez AI
                exit_price = None
                
                if position['side'] == 'long':
                    position['tsl_peak'] = max(position['tsl_peak'], row['high'])
                    tsl_price = position['tsl_peak'] * (1 - sl_pct)
                    if row['low'] <= tsl_price:
                        exit_price = tsl_price
                else: # Short
                    position['tsl_peak'] = min(position['tsl_peak'], row['low'])
                    tsl_price = position['tsl_peak'] * (1 + sl_pct)
                    if row['high'] >= tsl_price:
                        exit_price = tsl_price
                
                if exit_price:
                    position.update({'exit_time': idx, 'exit_reason': "TSL", 'exit_price': exit_price})
                    pnl_pct = (exit_price - position['avg_price']) / position['avg_price'] * position['side_int']
                    last_trade_pnl = np.clip(pnl_pct * 10, -1, 1)
                    all_trades.append(position); position = None
                    continue

            # Sprawdzanie stałego SL (tylko jeśli TSL nie jest aktywny)
            if position.get('sl') is not None and not position['tsl_active']:
                if (position['side'] == 'long' and row['low'] <= position['sl']) or \
                   (position['side'] == 'short' and row['high'] >= position['sl']):
                    position.update({'exit_time': idx, 'exit_reason': "SL", 'exit_price': position['sl']})
                    pnl_pct = (position['sl'] - position['avg_price']) / position['avg_price'] * position['side_int']
                    last_trade_pnl = np.clip(pnl_pct * 10, -1, 1)
                    all_trades.append(position); position = None
                    continue
            
            # Sprawdzanie stałego TP (tylko jeśli TSL nie jest aktywny)
            if position.get('tp') is not None and not position['tsl_active']:
                if (position['side'] == 'long' and row['high'] >= position['tp']) or \
                   (position['side'] == 'short' and row['low'] <= position['tp']):
                    position.update({'exit_time': idx, 'exit_reason': "TP_FIXED", 'exit_price': position['tp']})
                    pnl_pct = (position['tp'] - position['avg_price']) / position['avg_price'] * position['side_int']
                    last_trade_pnl = np.clip(pnl_pct * 10, -1, 1)
                    all_trades.append(position); position = None
                    continue
            
            # Logika AI do zarządzania wyjściem
            time_in_trade = (idx - position['entry_time']).total_seconds() / 3600
            unrealized_pnl = (row['close'] - position['avg_price']) * position['side_int'] / position['avg_price']
            
            # <<< POPRAWKA: Rozbudowana obserwacja dla Agenta Wyjścia (16 cech) >>>
            obs_state = np.array([
                unrealized_pnl,
                time_in_trade / 24.0
            ])
            obs_simple_ta = np.array([
                row.get('RSI', 50) / 100.0,
                row.get('WT1', 0) / 100.0,
                row.get('WT2', 0) / 100.0,
                (row['ema_fast'] - row['close']) / row['close'],
                (row['ema_slow'] - row['close']) / row['close']
            ])
            obs_trend_ta = row[['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']].values
            obs_temporal = row[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']].values
            
            obs_exit = np.hstack([obs_state, obs_simple_ta, obs_trend_ta, obs_temporal]).astype(np.float32)
            action_exit, _ = exit_model.predict(np.nan_to_num(obs_exit), deterministic=True)
            
            # <<< POPRAWKA: Logika dla 4 akcji >>>
            if action_exit == 1 and not position['breakeven_set']: # Ustaw BE
                position['sl'] = position['avg_price']
                position['breakeven_set'] = True
                position['tsl_active'] = False # BE anuluje TSL
            
            elif action_exit == 2: # Zamknij teraz
                position.update({'exit_time': idx, 'exit_reason': "AI_CLOSE", 'exit_price': row['close']})
                pnl_pct = (row['close'] - position['avg_price']) / position['avg_price'] * position['side_int']
                last_trade_pnl = np.clip(pnl_pct * 10, -1, 1)
                all_trades.append(position); position = None
            
            elif action_exit == 3 and not position['tsl_active'] and not position['breakeven_set']: # Aktywuj TSL
                if unrealized_pnl > 0: # Aktywuj tylko na plusie
                    position['tsl_active'] = True
                    position['tsl_peak'] = row['high'] if position['side'] == 'long' else row['low']
                    position['tp'] = None # TSL anuluje stały TP
            
            continue

        # --- Logika wejścia ---
        if not position and row['month'] not in month_traded:
            signal_price, side = None, None
            
            ema_filter_long = (not params["USE_EMA"]) or (params["USE_EMA"] and row['ema_fast'] > row['ema_slow'])
            ema_filter_short = (not params["USE_EMA"]) or (params["USE_EMA"] and row['ema_fast'] < row['ema_slow'])
            long_signal_condition = row['low'] <= row['band_down_2.0']
            short_signal_condition = row['high'] >= row['band_up_2.0']
            
            if params["TRADE_DIRECTION"] in ["long", "both"] and long_signal_condition and ema_filter_long:
                signal_price, side = row['band_down_2.0'], 'long'
            elif params["TRADE_DIRECTION"] in ["short", "both"] and short_signal_condition and ema_filter_short:
                signal_price, side = row['band_up_2.0'], 'short'
            
            if side:
                frame = df.iloc[i - lookback_window : i]
                
                obs_original = frame[['price_change', 'ema_fast', 'ema_slow', 'RSI', 'WT1', 'WT2', 'atr_norm']].values
                obs_original[:, 1] = (frame['ema_fast'] - frame['close']) / frame['close']
                obs_original[:, 2] = (frame['ema_slow'] - frame['close']) / frame['close']
                
                obs_trendline = frame[['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']].values
                # <<< POPRAWKA: Dodajemy cechy czasowe do obserwacji Agenta Wejścia >>>
                obs_temporal = frame[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']].values

                obs_last_pnl = np.full((lookback_window, 1), last_trade_pnl)

                # <<< POPRAWKA: Łączymy wszystkie 4 grupy (17 cech) >>>
                final_obs = np.hstack([obs_original, obs_trendline, obs_temporal, obs_last_pnl])
                final_obs = np.nan_to_num(final_obs.astype(np.float32))

                action, _ = entry_model.predict(final_obs, deterministic=True)
                confirm, sl_mult, rr_mult = action

                if confirm > 0.5:
                    position = {
                        'pair': pair, 'timeframe': timeframe, 'side': side, 
                        'side_int': 1 if side=='long' else -1, 'entry_time': idx, 
                        'sl_mult': sl_mult, 'rr_mult': rr_mult, 'executed_entries': [], 
                        'breakeven_set': False, 'avg_price': signal_price,
                        # <<< POPRAWKA: Dodajemy stan TSL do pozycji >>>
                        'tsl_active': False,
                        'tsl_peak': -np.inf if side == 'long' else np.inf
                    }
                    
                    sl_distance_pct = params['SL_PCT'] * sl_mult
                    tp_distance_pct = sl_distance_pct * rr_mult 
                    
                    if side == 'long':
                        position['sl'] = signal_price * (1 - sl_distance_pct)
                        position['tp'] = signal_price * (1 + tp_distance_pct)
                    else:
                        position['sl'] = signal_price * (1 + sl_distance_pct)
                        position['tp'] = signal_price * (1 - tp_distance_pct)
                        
                    position['executed_entries'].append({'price': signal_price, 'size': 1.0, 'time': idx}); month_traded.add(row['month'])

    pnl = 0
    for pos in all_trades:
        if pos.get('exit_price') is None or 'avg_price' not in pos: continue
        avg_entry = pos['avg_price']
        if pos['side'] == 'long': pnl += ((pos['exit_price'] - avg_entry) / avg_entry) * 100
        else: pnl += ((avg_entry - pos['exit_price']) / avg_entry) * 100
    return {"total_pnl": pnl, "trades": all_trades}

def objective_function(trial, entry_model, exit_model, pairs_in_group, timeframe):
    # Parametry Optuny pozostają bez zmian
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
    for pair in pairs_in_group:
        result = run_simulation(params, pair, timeframe, entry_model, exit_model)
        total_pnl += result["total_pnl"]
    
    return total_pnl / len(pairs_in_group) if pairs_in_group else 0

def optimize_timeframe_group(entry_model, exit_model, pairs_in_group, timeframe, n_trials=50, storage_dir="."):
    study_name = f"study_{timeframe}_group_exit_manager"
    storage_name = "postgresql://postgres:twojehaslo@localhost:5432/optuna_db"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    objective_with_args = partial(objective_function, entry_model=entry_model, exit_model=exit_model, pairs_in_group=pairs_in_group, timeframe=timeframe)
    study.optimize(objective_with_args, n_trials=n_trials, n_jobs=-1, gc_after_trial=True)

    return study.best_params, study.best_value
