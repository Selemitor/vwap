# exit_manager.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# <<< POPRAWKA: Importujemy wskaźniki, aby sprawdzić, czy istnieją >>>
from indicators import add_atr, add_trendline_features, add_temporal_features

class ExitManagerEnv(gym.Env):
    """Środowisko do nauki agenta, jak dynamicznie zarządzać otwartą pozycją."""
    def __init__(self, df, params):
        super().__init__()
        
        # Upewniamy się, że DF ma wszystkie potrzebne kolumny
        if 'atr' not in df.columns: df = add_atr(df)
        if 'adx' not in df.columns: df = add_trendline_features(df, lookback=72)
        if 'day_sin' not in df.columns: df = add_temporal_features(df)
        
        df.ffill(inplace=True); df.bfill(inplace=True)
        self.df = df
        self.params = params
        self.max_steps = len(df) - 2
        
        # <<< POPRAWKA: Rozszerzona przestrzeń obserwacji >>>
        # 2 (stan) + 5 (TA) + 5 (Trend) + 4 (Czas) = 16 cech
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        # <<< POPRAWKA: Rozszerzona przestrzeń akcji >>>
        # Akcje: 0=Trzymaj, 1=Przesuń SL na BE, 2=Zamknij teraz, 3=Aktywuj Trailing Stop (TSL)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(72, self.max_steps - 100) # Start po wygrzaniu wskaźników
        self.trade_open_step = self.current_step
        self.entry_price = self.df.iloc[self.current_step]['close']
        self.side = np.random.choice([1, -1]) # 1 dla long, -1 dla short
        self.stop_loss = self.entry_price - (self.entry_price * self.params['SL_PCT']) * self.side
        
        # <<< POPRAWKA: Nowe zmienne stanu >>>
        self.breakeven_set = False
        self.tsl_active = False
        self.tsl_peak_price = -np.inf if self.side == 1 else np.inf # Szczyt ceny od aktywacji TSL

        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        time_in_trade = (self.current_step - self.trade_open_step) / 50.0 # Normalizujemy
        unrealized_pnl = (row['close'] - self.entry_price) * self.side / self.entry_price
        
        # <<< POPRAWKA: Zbieramy wszystkie 16 cech >>>
        obs_state = np.array([
            unrealized_pnl,
            time_in_trade
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
        
        obs = np.hstack([obs_state, obs_simple_ta, obs_trend_ta, obs_temporal]).astype(np.float32)
        
        return np.nan_to_num(obs)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if done:
            return self._next_observation(), 0, done, False, {}
            
        row = self.df.iloc[self.current_step]
        current_price = row['close']
        current_low = row['low']
        current_high = row['high']
        reward = 0
        
        # --- Logika Stop Lossa (TSL lub stały) ---
        sl_price_to_check = self.stop_loss

        if self.tsl_active:
            sl_distance = self.entry_price * self.params['SL_PCT'] # Dystans TSL bazuje na początkowym SL
            if self.side == 1: # Long
                self.tsl_peak_price = max(self.tsl_peak_price, current_high)
                sl_price_to_check = self.tsl_peak_price - sl_distance
            else: # Short
                self.tsl_peak_price = min(self.tsl_peak_price, current_low)
                sl_price_to_check = self.tsl_peak_price + sl_distance

        # Sprawdzenie, czy SL nie został trafiony
        if (self.side == 1 and current_low <= sl_price_to_check) or \
           (self.side == -1 and current_high >= sl_price_to_check):
            reward = -1 # Maksymalna kara za SL
            done = True
        
        if not done:
            if action == 0: # Trzymaj
                # Nagroda za trzymanie zyskownych pozycji, kara za trzymanie stratnych
                reward = (current_price - self.df.iloc[self.current_step - 1]['close']) * self.side / self.entry_price
            
            elif action == 1 and not self.breakeven_set: # Przesuń SL na BE
                self.stop_loss = self.entry_price
                self.breakeven_set = True
                self.tsl_active = False # BE anuluje TSL
                reward = 0.1 # Mała nagroda za zabezpieczenie pozycji
            
            elif action == 2: # Zamknij teraz
                pnl = (current_price - self.entry_price) * self.side / self.entry_price
                reward = np.clip(pnl * 10, -1, 1) # Duża nagroda/kara za finalną decyzję
                done = True
                
            elif action == 3 and not self.tsl_active and not self.breakeven_set: # Aktywuj TSL
                unrealized_pnl = (current_price - self.entry_price) * self.side / self.entry_price
                if unrealized_pnl > 0: # Aktywuj TSL tylko jeśli pozycja jest na plusie
                    self.tsl_active = True
                    self.tsl_peak_price = current_high if self.side == 1 else current_low
                    reward = 0.2 # Większa nagroda za mądrą aktywację TSL
                else:
                    reward = -0.1 # Kara za próbę TSL na stratnej pozycji

        return self._next_observation(), reward, done, False, {}

def train_exit_manager(df, params, timesteps=75000, save_path="ai_exit_manager_model.zip"):
    print("🤖 Rozpoczynanie treningu Agenta AI (Wyjście, 16 cech, 4 akcje)...")
    df.ffill(inplace=True); df.bfill(inplace=True)
    env = ExitManagerEnv(df, params)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"✅ Trening Menedżera Wyjścia zakończony. Model zapisany jako '{save_path}'.")
    return model