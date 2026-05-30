"""
01_data_exploration.py
Eksploracja danych i analiza surowego zbioru Steam Games Dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import io

# UTF-8 encoding dla poprawnego wyświetlania znaków w konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_raw_data():
    """Ładuje surowe dane z pliku CSV"""
    data_dir = Path(__file__).parent / "data"
    
    csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                       if f.stem != "games_cleaned" and f.stem != "games_engineered"])
    if not csv_files:
        print("Brak plików CSV w katalogu data/. Spróbuję pobrać dane...")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("data_collection", Path(__file__).parent / "01_data_collection.py")
            data_collection = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_collection)
        except Exception as e:
            print(f"Nie udało się załadować skryptu 01_data_collection: {e}")
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ i nie można pobrać danych.")

        try:
            data_collection.download_and_move_dataset()
        except Exception as e:
            print(f"Błąd podczas pobierania danych: {e}")
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ i pobieranie zakończyło się błędem.")

        csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                            if f.stem != "games_cleaned" and f.stem != "games_engineered"])

        if not csv_files:
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ po próbie pobrania danych")
    
    latest_file = csv_files[-1]
    print(f"Ladowanie danych z: {latest_file.name}")
    
    df = pd.read_csv(latest_file, index_col=False)
    print(f"\n[OK] Zaladowano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn\n")
    return df

def get_columns_info(df):
    """Analiza informacji o kolumnach"""
    print("=" * 80)
    print("INFORMACJE O KOLUMNACH")
    print("=" * 80)
    
    selected_columns = [
        'AppID', 'Name', 'Release date', 'Price', 'Windows', 'Mac', 'Linux',
        'Metacritic score', 'Achievements', 'Developers', 'Publishers',
        'Categories', 'Genres', 'User score', 'Score rank', 'Positive',
        'Negative', 'Estimated owners'
    ]
    
    print(f"\nKolumny do preprocessingu ({len(selected_columns)}):")
    print("-" * 80)
    
    for col in selected_columns:
        if col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sample = df[col].dropna().mean() if missing < len(df) else "N/A"
                else:
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            except Exception as e:
                sample = "N/A"
            
            print(f"  * {col:25} | Type: {str(dtype):10} | Braki: {missing:7} ({missing_pct:5.1f}%) | Sample: {sample}")
        else:
            print(f"  * {col:25} | [!] KOLUMNA NIE ISTNIEJE")

def analyze_selected_data(df):
    """Analizuje szczegółowo wybrane kolumny"""
    print("\n" + "=" * 80)
    print("ANALIZA")
    print("=" * 80)
    
    # AppID
    print(f"\n[DATA] AppID:")
    print(f"  - Unikalne wartosci: {df['AppID'].nunique()}")
    print(f"  - Duplikaty: {df['AppID'].duplicated().sum()}")
    
    # Price
    print(f"\n[PRICE] Price:")
    print(f"  - Typ danych: {df['Price'].dtype}")
    print(f"  - Braki: {df['Price'].isna().sum()}")
    print(f"  - Bezplatne gry (0): {(df['Price'] == 0).sum()}")
    print(f"  - Min cena: ${df['Price'].min()}")
    print(f"  - Max cena: ${df['Price'].max()}")
    print(f"  - Srednia cena: ${df['Price'].mean():.2f}")
    
    # Scores
    print(f"\n[SCORE] Oceny:")
    metacritic = pd.to_numeric(df['Metacritic score'], errors='coerce')
    print(f"  - Metacritic score - srednia (nie-puste): {metacritic.mean():.1f}")
    user_score = pd.to_numeric(df['User score'], errors='coerce')
    print(f"  - User score - srednia (nie-puste): {user_score.mean():.2f}")
    
    # Reviews & Noise Analysis
    print(f"\n[REVIEW] Recenzje i Analiza Szumu:")
    pos = pd.to_numeric(df['Positive'], errors='coerce').fillna(0)
    neg = pd.to_numeric(df['Negative'], errors='coerce').fillna(0)
    total_rev = pos + neg
    
    print(f"  - Gry z całkowitym brakiem opinii (0 recenzji): {(total_rev == 0).sum()} ({(total_rev == 0).sum()/len(df)*100:.1f}%)")
    print(f"  - Gry o niskiej aktywności (1-4 recenzji): {((total_rev > 0) & (total_rev < 5)).sum()}")
    print(f"  - Gry aktywne rynkowo (>= 5 recenzji): {(total_rev >= 5).sum()} ({(total_rev >= 5).sum()/len(df)*100:.1f}%)")

def save_exploration_summary(df):
    """Zapisuje raport z eksploracji"""
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "selected_columns": 18,
        "missing_values": df.isna().sum().to_dict(),
        "column_types": df.dtypes.astype(str).to_dict()
    }
    
    output_file = output_dir / "01_exploration_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] Raport eksploracji zapisany: reports/01_exploration_summary.json")

def main():
    print("\n" + "=" * 80)
    print("EKSPLORACJA DANYCH - STEAM GAMES DATASET")
    print("=" * 80 + "\n")
    
    df = load_raw_data()
    get_columns_info(df)
    analyze_selected_data(df)
    save_exploration_summary(df)
    
    print("\n" + "=" * 80)
    print("[OK] EKSPLORACJA UKONCZONA")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()