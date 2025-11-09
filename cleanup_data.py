import os
import pandas as pd
from pathlib import Path
import time
from config import DATA_PATH # Importuje poprawną ścieżkę ~/vwap/data
import sys

# --- KONFIGURACJA ---
MIN_CANDLE_COUNT = 150 

# <<< NOWA LOGIKA: Jedna "czarna lista" walut bazowych do usunięcia >>>
BLACKLISTED_BASE_CURRENCIES = [
    # Stablecoiny
    'USDT', 'USDC', 'DAI', 'TUSD', 'BUSD', 'FDUSD', 'USDP', 'PYUSD', 'PAX', 'GUSD', 'USDD', 'FRAX',
    
    # Główne Wrapped Tokens
    'WBTC',  # Wrapped BTC
    'WETH',  # Wrapped ETH
    'WBNB',  # Wrapped BNB
    'WAVAX', # Wrapped AVAX
    'WMATIC',# Wrapped MATIC
    'WFTM',  # Wrapped FTM
    
    # Popularne Liquid Staking / Bridged Tokens
    'STETH', # Lido Staked ETH
    'RETH',  # Rocket Pool ETH
    'CBETH', # Coinbase Wrapped Staked ETH
    'CETH',  # Compound ETH
    'CDAI',  # Compound DAI
    'CUSDC', # Compound USDC
    'AXLUSDC', # Axelar USDC
    'SOBTC'  # Solana Wrapped BTC
]
# ---

print(f"🧹 Rozpoczynanie INTELIGENTNEGO czyszczenia danych w: {DATA_PATH}")
print(f"Minimalna wymagana liczba świec na D1: {MIN_CANDLE_COUNT}")
print(f"Usuwanie: Krótkich historii, Delisted ORAZ {len(BLACKLISTED_BASE_CURRENCIES)} pozycji z czarnej listy.")
time.sleep(3)

# --- KROK 1: Odkryj wszystkie unikalne pary ---
all_files = os.listdir(DATA_PATH)
all_pairs = set()
files_to_scan = [] 

for filename in all_files:
    if filename.endswith(".parquet"):
        files_to_scan.append(filename)
        try:
            parts = filename.replace('.parquet', '').replace('_DELISTED', '').split('_')
            
            if len(parts) >= 2:
                pair_name = f"{parts[0]}_{parts[1]}"
                all_pairs.add(pair_name)
        except Exception as e:
            print(f"Błąd parsowania nazwy pliku: {filename} - {e}")

print(f"Znaleziono {len(all_pairs)} unikalnych par do weryfikacji...")

# --- KROK 2: Znajdź pary do usunięcia (Logika połączona) ---
pairs_to_remove = set()
pairs_checked = 0

for pair in all_pairs:
    pairs_checked += 1
    remove_this_pair = False
    reason = ""

    try:
        base_currency = pair.split('_')[0].upper()
        
        # <<< NOWY FILTR 1: Sprawdź Czarną Listę (Stablecoiny i Wrapped) >>>
        if base_currency in BLACKLISTED_BASE_CURRENCIES:
            remove_this_pair = True
            reason = "Blacklisted (Stable/Wrapped)"
        
        # FILTR 2: Długość (Tylko jeśli przeszedł filtr 1)
        if not remove_this_pair:
            d1_file_name = f"{pair}_1d.parquet"
            d1_file_path = DATA_PATH / d1_file_name
            
            if not d1_file_path.exists():
                d1_delisted_path = DATA_PATH / f"{pair}_1d_DELISTED.parquet"
                if not d1_delisted_path.exists():
                    remove_this_pair = True
                    reason = "Brak pliku 1d"
                else:
                    d1_file_path = d1_delisted_path
            
            if not remove_this_pair: 
                try:
                    df_1d = pd.read_parquet(d1_file_path)
                    if len(df_1d) < MIN_CANDLE_COUNT:
                        remove_this_pair = True
                        reason = f"Plik 1d za krótki ({len(df_1d)} < {MIN_CANDLE_COUNT})"
                except Exception as e:
                    remove_this_pair = True
                    reason = f"Błąd odczytu pliku 1d ({e})"
    
    except Exception as e:
        remove_this_pair = True
        reason = f"Błąd parsowania nazwy pary ({e})"

    if remove_this_pair:
        print(f"  -> Oznaczono do usunięcia: {pair} (Powód: {reason})")
        pairs_to_remove.add(pair)
        
    if pairs_checked % 100 == 0:
        print(f"  ...Sprawdzono {pairs_checked}/{len(all_pairs)} par...")

print(f"\n--- KROK 3: Usuwanie {len(pairs_to_remove)} złych par ORAZ wszystkich plików DELISTED ---")

files_removed_count = 0
for filename in files_to_scan: 
    try:
        file_path_to_remove = DATA_PATH / filename
        should_remove = False
        reason_for_removal = ""

        # FILTR 3: Usuń, jeśli 'DELISTED' jest w nazwie
        if "DELISTED" in filename.upper():
            should_remove = True
            reason_for_removal = "DELISTED"
        
        # REGUŁA 2: Sprawdź listę logiczną (jeśli jeszcze nie oznaczono do usunięcia)
        if not should_remove:
            parts = filename.replace('.parquet', '').replace('_DELISTED', '').split('_')
            if len(parts) >= 2:
                pair_name = f"{parts[0]}_{parts[1]}"
                if pair_name in pairs_to_remove:
                    should_remove = True
                    reason_for_removal = "Logika (Krótki/Stable/Wrapped)"

        # Wykonaj usunięcie
        if should_remove and file_path_to_remove.exists():
            os.remove(file_path_to_remove)
            files_removed_count += 1
            print(f"  -> Usunięto (lokalnie): {filename} (Powód: {reason_for_removal})")

    except Exception as e:
        print(f"Błąd podczas usuwania {filename}: {e}")

print("\n" + "="*50)
print("✅ Inteligentne czyszczenie zakończone.")
print(f"  Usunięto par (łącznie): {len(pairs_to_remove)}")
print(f"  Łącznie usunięto plików (wszystkie interwały): {files_removed_count}")
print("="*50)
print("\nMożesz teraz uruchomić główny skrypt. Będzie on używał tylko przefiltrowanych danych lokalnych.")
