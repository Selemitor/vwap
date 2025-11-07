# config.py
from pathlib import Path

# Znajdź ścieżkę do folderu, w którym znajduje się ten plik (czyli D:\Skrypty\VWAP\)
SCRIPT_DIR = Path(__file__).resolve().parent

# Ustaw ścieżkę danych jako podfolder "data" wewnątrz folderu projektu
DATA_PATH = SCRIPT_DIR / "data"
