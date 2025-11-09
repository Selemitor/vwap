# data_loader.py
import os
import pandas as pd
import json
from pathlib import Path
from pycoingecko import CoinGeckoAPI
from config import DATA_PATH
import time

def discover_data_files():
    """
    <<< ZMIANA: Przeszukuje folder danych w formacie PARQUET >>>
    """
    files_structure = {}
    print("🔍 Przeszukiwanie folderu z danymi w formacie PARQUET...")
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".parquet"):
            parts = filename.replace('.parquet', '').split('_')
            if len(parts) >= 2:
                pair_name = f"{parts[0]}_{parts[1]}"
                timeframe = parts[2] if len(parts) > 2 else 'unknown'
                
                if pair_name not in files_structure:
                    files_structure[pair_name] = []
                files_structure[pair_name].append(timeframe)
    
    print(f"✅ Znaleziono {len(files_structure)} par krypto z plikami Parquet.")
    return files_structure

def fetch_and_cache_market_caps(tickers, cache_file="mcap_cache.json", cache_duration_hours=24):
    cache_file_path = DATA_PATH / cache_file
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f: cache = json.load(f)
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
    for i in range(0, len(ids_to_query), 400):
        batch_ids = ids_to_query[i:i+400]
        try:
            prices = cg.get_price(ids=batch_ids, vs_currencies='usd', include_market_cap='true')
            price_data.update(prices); time.sleep(0.5)
        except Exception as e: print(f"Błąd podczas pobierania danych z CoinGecko: {e}")
    for ticker in tickers:
        base_currency_upper = ticker.split('_')[0].upper(); coin_id = symbol_to_id.get(base_currency_upper)
        if coin_id and coin_id in price_data and 'usd_market_cap' in price_data[coin_id]:
            mcap_data[ticker] = price_data[coin_id]['usd_market_cap']
        else: mcap_data[ticker] = 0
    with open(cache_file_path, 'w') as f: json.dump({'timestamp': time.time(), 'data': mcap_data}, f)
    print("✅ Pobrano i zapisano kapitalizację rynkową.")
    return mcap_data

def preload_all_data(files_structure):
    """
    <<< ZMIANA: Wczytuje dane TYLKO z plików Parquet. >>>
    """
    print("⏳ Wczytywanie danych z plików Parquet...")
    data_store = {}
    
    for pair, timeframes in files_structure.items():
        data_store[pair] = {}
        for tf in timeframes:
            parquet_path = DATA_PATH / f"{pair}_{tf}.parquet"
            
            df = None
            try:
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    print(f"  -> Wczytano: {pair} {tf}")
                    
                    if "timestamp" in df.columns:
                        if pd.api.types.is_numeric_dtype(df["timestamp"]):
                            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                        else:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df.set_index('timestamp', inplace=True)
                    elif "time" in df.columns: 
                        if pd.api.types.is_numeric_dtype(df["time"]):
                            df["time"] = pd.to_datetime(df["time"], unit='ms')
                        else:
                            df["time"] = pd.to_datetime(df["time"])
                        df.set_index('time', inplace=True)

                    data_store[pair][tf] = df
                else:
                    print(f"Ostrzeżenie: Nie znaleziono pliku Parquet dla {pair} {tf}")
                    continue

            except Exception as e:
                print(f"❌ Błąd podczas przetwarzania pliku Parquet dla {pair} {tf}: {e}")

    print("✅ Wszystkie dane Parquet załadowane.\n")
    return data_store
