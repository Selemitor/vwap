# ai_manager.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
# <<< POPRAWKA: Importujemy obie funkcje >>>
from indicators import add_atr, add_trendline_features, add_temporal_features

class TradeManagerEnv(gym.Env):
    def __init__(self, df, params):
        super(TradeManagerEnv, self).__init__()
        
        # Upewniamy się, że DF ma wszystkie potrzebne kolumny
        if 'atr' not in df.columns: df = add_atr(df)
        if 'price_change' not in df.columns: df['price_change'] = df['close'].pct_change()
        if 'atr_norm' not in df.columns: df['atr_norm'] = df['atr'] / df['close']
        if 'signal_ema' not in df.columns: df['signal_ema'] = df['close'].ewm(span=50).mean()
        if 'adx' not in df.columns: df = add_trendline_features(df, lookback=72)
        # <<< POPRAWKA: Dodajemy cechy czasowe >>>
        if 'day_sin' not in df.columns: df = add_temporal_features(df)
        
        df.ffill(inplace=True); df.bfill(inplace=True)
        self.df = df
        
        self.params = params
        self.lookback_window = 10
        
        # <<< POPRAWKA: 7 TA + 5 Trend + 4 Czas + 1 PnL = 17 cech >>>
        self.num_features = 17 
        self.max_steps = len(df) - self.lookback_window - 100

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.lookback_window, self.num_features), dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.7, 0.7]),
            high=np.array([1.0, 2.0, 3.0]),
            dtype=np.float32
        )
        
        self.last_trade_pnl = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(72, self.max_steps - 100)
        self.last_trade_pnl = 0.0
        return self._next_observation(), {}

    def _next_observation(self):
        start = self.current_step
        end = start + self.lookback_window
        
        if end > len(self.df):
            end = len(self.df)
            start = end - self.lookback_window
            if start < 0: start = 0
        
        frame = self.df.iloc[start:end]
        actual_len = len(frame)
        if actual_len == 0:
            return np.zeros((self.lookback_window, self.num_features), dtype=np.float32)

        obs_original = frame[['price_change', 'ema_fast', 'ema_slow', 'RSI', 'WT1', 'WT2', 'atr_norm']].values
        obs_original[:, 1] = (frame['ema_fast'] - frame['close']) / frame['close']
        obs_original[:, 2] = (frame['ema_slow'] - frame['close']) / frame['close']

        obs_trendline = frame[['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']].values
        # <<< POPRAWKA: Dodajemy cechy czasowe >>>
        obs_temporal = frame[['day_sin', 'day_cos', 'hour_sin', 'hour_cos']].values
        
        obs_last_pnl_val = self.last_trade_pnl
        
        if actual_len < self.lookback_window:
            pad_width = ((0, self.lookback_window - actual_len), (0, 0))
            obs_original = np.pad(obs_original, pad_width, 'edge')
            obs_trendline = np.pad(obs_trendline, pad_width, 'edge')
            obs_temporal = np.pad(obs_temporal, pad_width, 'edge') # Padujemy również czas

        obs_last_pnl = np.full((self.lookback_window, 1), obs_last_pnl_val)
        
        # <<< POPRAWKA: Łączymy wszystkie 4 grupy cech >>>
        final_obs = np.hstack([obs_original, obs_trendline, obs_temporal, obs_last_pnl])
        return np.nan_to_num(final_obs.astype(np.float32))

    def step(self, action):
        confirm, sl_mult, rr_mult = action
        
        entry_row_index = self.current_step + self.lookback_window
        if entry_row_index >= len(self.df):
            self.current_step += 1
            done = True
            return self._next_observation(), 0, done, False, {}
            
        entry_row = self.df.iloc[entry_row_index]
        
        if pd.isna(entry_row['close']) or pd.isna(entry_row['signal_ema']):
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self._next_observation(), 0, done, False, {}
            
        side = 'long' if entry_row['close'] < entry_row['signal_ema'] else 'short'
        
        real_outcome = 0
        if confirm > 0.5:
            real_outcome = self._simulate_trade(entry_row_index, side, sl_mult, rr_mult)
        else:
            real_outcome = -self._simulate_trade(entry_row_index, side, 1.0, 1.0)
        
        self.last_trade_pnl = real_outcome
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._next_observation(), real_outcome, done, False, {}

    def _simulate_trade(self, entry_step_index, side, sl_mult, rr_mult):
        trade_horizon = 48
        entry_price = self.df.iloc[entry_step_index]['close']
        
        if entry_price == 0 or pd.isna(entry_price): return 0
        
        sl_pct_modified = self.params['SL_PCT'] * sl_mult
        
        if side == 'long':
            sl_price = entry_price * (1 - sl_pct_modified)
            tp_price = entry_price + (entry_price - sl_price) * rr_mult
        else:
            sl_price = entry_price * (1 + sl_pct_modified)
            tp_price = entry_price - (sl_price - entry_price) * rr_mult
            
        for i in range(1, trade_horizon + 1):
            current_step_index = entry_step_index + i
            if current_step_index >= len(self.df): return 0
            row = self.df.iloc[current_step_index]
            if side == 'long':
                if row['low'] <= sl_price: return -1
                if row['high'] >= tp_price: return 1
            else:
                if row['high'] >= sl_price: return -1
                if row['low'] <= tp_price: return 1
        
        final_price_index = entry_step_index + trade_horizon
        if final_price_index >= len(self.df): return 0
            
        final_price = self.df.iloc[final_price_index]['close']
        if pd.isna(final_price): return 0
        
        pnl = (final_price - entry_price) / entry_price if side == 'long' else (entry_price - final_price) / entry_price
        return np.clip(pnl * 10, -1, 1)

def train_ai_manager(df, params, timesteps=50000, save_path="ai_entry_strategist_model.zip"):
    # <<< POPRAWKA: Zmieniony komunikat >>>
    print("🤖 Rozpoczynanie treningu Agenta AI (Wejście, 17 cech)...")
    env = TradeManagerEnv(df, params)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"✅ Trening zakończony. Model '{save_path}' jest gotowy do użytku.")
    return model