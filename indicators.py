# indicators.py
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator

# <<< NOWA FUNKCJA: Dodawanie cech czasowych >>>
def add_temporal_features(df):
    """Dodaje cykliczne cechy czasowe (dzień tygodnia, godzina dnia) do DataFrame."""
    # Upewnij się, że indeks jest typu datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("  -> Ostrzeżenie: Indeks nie jest typu datetime. Pomijanie cech czasowych.")
        return df

    # Dzień tygodnia (0=Poniedziałek, 6=Niedziela)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Godzina dnia (0-23)
    df['hour_of_day'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    
    # Usuwamy kolumny pomocnicze
    df.drop(columns=['day_of_week', 'hour_of_day'], inplace=True, errors='ignore')
    return df

def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.copy(); gain[gain < 0] = 0
    loss = delta.copy(); loss[loss > 0] = 0; loss = abs(loss)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi; df[f'RSI_{period}'] = rsi
    return df

def add_wave_trend(df, n1=10, n2=21):
    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = abs(ap - esa).ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    df['WT1'] = wt1; df['WT2'] = wt2
    df[f'WT1_{n1}_{n2}'] = wt1; df[f'WT2_{n1}_{n2}'] = wt2
    return df

def add_atr(df, period=14):
    """Oblicza i dodaje do DataFrame wskaźnik ATR."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    df['atr'] = atr
    return df

def fit_trendline(prices):
    """Dopasowuje linię trendu do podanych cen za pomocą regresji liniowej."""
    x = np.arange(len(prices))
    coeffs = np.polyfit(x, prices, 1)
    slope, intercept = coeffs[0], coeffs[1]
    return slope, intercept

def add_trendline_features(df, lookback=72):
    """
    Oblicza i dodaje do DataFrame cechy oparte na liniach trendu.
    """
    if len(df) < lookback * 2: 
        return df 

    df_out = df.copy()
    feature_cols = ['res_slope_norm', 'res_error_norm', 'res_max_dist_norm', 'vol_norm', 'adx']
    
    if 'atr' not in df_out.columns:
        df_out = add_atr(df_out, period=14)
    
    try:
        adx_indicator = ADXIndicator(high=df_out['high'], low=df_out['low'], close=df_out['close'], window=lookback)
        df_out['adx'] = adx_indicator.adx()
    except IndexError as e:
        print(f"  -> OSTRZEŻENIE: Pomijanie cech ADX/Trendline (błąd: {e}). Dane prawdopodobnie zbyt krótkie lub nieprawidłowe.")
        return df 
    except Exception as e:
        print(f"  -> OSTRZEŻENIE: Inny błąd ADX: {e}. Ustawiam ADX na 0.")
        df_out['adx'] = 0.0

    df_out['vol_norm'] = df_out['volume'] / df_out['volume'].rolling(window=lookback).median()
    
    slopes, errors, max_dists = [], [], []
    close_prices = df_out['close'].values
    atr_values = df_out['atr'].values

    for i in range(len(df_out)):
        if i < lookback:
            slopes.append(np.nan); errors.append(np.nan); max_dists.append(np.nan)
            continue
        
        window = close_prices[i-lookback : i]
        slope, intercept = fit_trendline(window)
        x_window = np.arange(len(window))
        resistance_line = slope * x_window + intercept
        resistance_line += np.max(window - resistance_line)
        distances = resistance_line - window
        
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0
        
        slopes.append(slope / current_atr)
        errors.append(np.mean(distances) / current_atr)
        max_dists.append(np.max(distances) / current_atr)

    df_out['res_slope_norm'] = slopes
    df_out['res_error_norm'] = errors
    df_out['res_max_dist_norm'] = max_dists
    
    df_out.ffill(inplace=True)
    df_out.bfill(inplace=True)

    return df_out