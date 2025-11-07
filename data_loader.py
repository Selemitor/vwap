# data_loader.py
import os
import pandas as pd
import json
from pathlib import Path
from pycoingecko import CoinGeckoAPI
from config import DATA_PATH
import time

def discover_data_files():
    """Przeszukuje folder danych i zwraca słownik z dostępnymi parami i interwałami w formacie CSV."""
    files_structure = {}
    print("🔍 Przeszukiwanie folderu z danymi w formacie CSV...")
    # <<< ZMIANA: Szukamy plików .csv jako źródła prawdy >>>
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".csv"):
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                pair_name = f"{parts[0]}_{parts[1]}"
                timeframe = parts[2] if len(parts) > 2 else 'unknown'
                
                if pair_name not in files_structure:
                    files_structure[pair_name] = []
                files_structure[pair_name].append(timeframe)
    print(f"✅ Znaleziono {len(files_structure)} par krypto z plikami CSV.")
    return files_structure

def fetch_and_cache_market_caps(tickers, cache_file="mcap_cache.json", cache_duration_hours=24):
    # ... (ta funkcja pozostaje bez zmian) ...
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f: cache = json.load(f)
        if time.time() - cache.get('timestamp', 0) < cache_duration_hours * 3600:
            print("✅ Wczytano kapitalizację rynkową z pamięci podręcznej.")
            return cache['data']
    print("💰 Pobieranie aktualnej kapitalizacji rynkowej z CoinGecko...")
    cg = CoinGeckoAPI(); all_coins_list = cg.get_coins_list()
    symbol_to_id = {coin['symbol'].upper(): coin['id'] for coin in all_coins_list}
    mcap_data = {}; ids_to_query = []
    for ticker in tickers:
        base_currency = ticker.split('_')[0].upper()
        if base_currency in symbol_to_id: ids_to_query.append(symbol_to_id[base_currency])
    price_data = {}
    for i in range(0, len(ids_to_query), 250):
        batch_ids = ids_to_query[i:i+250]
        try:
            prices = cg.get_price(ids=batch_ids, vs_currencies='usd', include_market_cap='true')
            price_data.update(prices); time.sleep(1)
        except Exception as e: print(f"Błąd podczas pobierania danych z CoinGecko: {e}")
    for ticker in tickers:
        base_currency_upper = ticker.split('_')[0].upper(); coin_id = symbol_to_id.get(base_currency_upper)
        if coin_id and coin_id in price_data and 'usd_market_cap' in price_data[coin_id]:
            mcap_data[ticker] = price_data[coin_id]['usd_market_cap']
        else: mcap_data[ticker] = 0
    with open(cache_file, 'w') as f: json.dump({'timestamp': time.time(), 'data': mcap_data}, f)
    print("✅ Pobrano i zapisano kapitalizację rynkową.")
    return mcap_data

def preload_all_data(files_structure):
    """
    Inteligentnie wczytuje dane: szuka wersji Parquet, a jeśli jej nie ma,
    tworzy ją z pliku CSV.
    """
    print("⏳ Inteligentne wczytywanie danych (z użyciem cache Parquet)...")
    data_store = {}
    
    # <<< NOWA LOGIKA: Automatyczne tworzenie i wczytywanie z cache Parquet >>>
    cache_dir = DATA_PATH / ".cache_parquet"
    cache_dir.mkdir(exist_ok=True) # Stwórz folder cache, jeśli nie istnieje

    for pair, timeframes in files_structure.items():
        data_store[pair] = {}
        for tf in timeframes:
            parquet_path = cache_dir / f"{pair}_{tf}.parquet"
            csv_path = DATA_PATH / f"{pair}_{tf}.csv"
            
            df = None
            try:
                if parquet_path.exists():
                    # Jeśli wersja Parquet istnieje w cache, wczytaj ją (szybko)
                    df = pd.read_parquet(parquet_path)
                    print(f"  -> Wczytano z cache: {pair} {tf}")
                elif csv_path.exists():
                    # Jeśli nie, wczytaj CSV (wolno)
                    print(f"  -> Tworzenie cache dla: {pair} {tf} (wolniejsze pierwsze wczytanie)")
                    df_csv = pd.read_csv(csv_path)
                    
                    # Przekonwertuj timestamp
                    if "timestamp" in df_csv.columns:
                        if pd.api.types.is_numeric_dtype(df_csv["timestamp"]):
                            df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], unit='ms')
                        else:
                            df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"])
                        df_csv.set_index('timestamp', inplace=True)
                    
                    # Zapisz jako Parquet w folderze cache na przyszłość
                    df_csv.to_parquet(parquet_path)
                    df = df_csv
                else:
                    print(f"Ostrzeżenie: Nie znaleziono pliku CSV dla {pair} i interwału {tf}")
                    continue
                
                if df is not None:
                    data_store[pair][tf] = df

            except Exception as e:
                print(f"❌ Błąd podczas przetwarzania pliku dla {pair} {tf}: {e}")

    print("✅ Wszystkie dane załadowane.\n")
    return data_store