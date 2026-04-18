"""
02_data_exploration.py
Eksploracja i analiza struktury danych

Ten skrypt:
- Ładuje dane z pliku CSV
- Analizuje statystykę opisową
- Sprawdza brakujące wartości
- Generuje podstawowe wizualizacje
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import io

# UTF-8 encoding
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_raw_data():
    """Ładuje surowe dane z pliku CSV"""
    data_dir = Path(__file__).parent / "data"
    
    # Load only raw data (NOT processed files like games_cleaned.csv or games_engineered.csv)
    csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                       if f.stem != "games_cleaned" and f.stem != "games_engineered"])
    if not csv_files:
        raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/")
    
    latest_file = csv_files[-1]
    print(f"Ladowanie danych z: {latest_file.name}")
    
    df = pd.read_csv(latest_file, index_col=False)
    print(f"\n[OK] Zaladowano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn\n")
    
    return df

def get_columns_info(df):
    """Analizuje informacje o kolumnach"""
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
            
            # Determine sample value based on dtype
            try:
                import pandas as pd
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
    print("ANALIZA SZCZEGOLOWA")
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
    print(f"  - Mediana ceny: ${df['Price'].median():.2f}")
    
    # Platform support
    print(f"\n[PLATFORM] Obsluga platform:")
    print(f"  - Windows: {df['Windows'].notna().sum()} ({(df['Windows'].notna().sum()/len(df)*100):.1f}%)")
    print(f"  - Mac: {df['Mac'].notna().sum()} ({(df['Mac'].notna().sum()/len(df)*100):.1f}%)")
    print(f"  - Linux: {df['Linux'].notna().sum()} ({(df['Linux'].notna().sum()/len(df)*100):.1f}%)")
    
    # Scores
    print(f"\n[SCORE] Oceny:")
    print(f"  - Metacritic score braki: {df['Metacritic score'].isna().sum()}")
    metacritic = pd.to_numeric(df['Metacritic score'], errors='coerce')
    print(f"  - Metacritic score - srednia: {metacritic.mean():.1f}")
    print(f"  - User score braki: {df['User score'].isna().sum()}")
    user_score = pd.to_numeric(df['User score'], errors='coerce')
    print(f"  - User score - srednia: {user_score.mean():.2f}")
    
    # Reviews
    print(f"\n[REVIEW] Recenzje:")
    print(f"  - Positive braki: {df['Positive'].isna().sum()}")
    positive = pd.to_numeric(df['Positive'], errors='coerce')
    print(f"  - Positive - srednia: {positive.mean():.0f}")
    print(f"  - Negative braki: {df['Negative'].isna().sum()}")
    negative = pd.to_numeric(df['Negative'], errors='coerce')
    print(f"  - Negative - srednia: {negative.mean():.0f}")
    
    # Metadata
    print(f"\n[META] Metadane:")
    print(f"  - Developers braki: {df['Developers'].isna().sum()}")
    print(f"  - Publishers braki: {df['Publishers'].isna().sum()}")
    print(f"  - Categories braki: {df['Categories'].isna().sum()}")
    print(f"  - Genres braki: {df['Genres'].isna().sum()}")
    print(f"  - Achievements braki: {df['Achievements'].isna().sum()}")

def save_exploration_summary(df):
    """Zapisuje raport z eksploracji"""
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "selected_columns": 18,
        "missing_values": df.isna().sum().to_dict(),
        "column_types": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.describe().to_dict()
    }
    
    output_file = output_dir / "01_exploration_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[OK] Raport eksploracji zapisany: reports/01_exploration_summary.json")

def main():
    print("\n" + "=" * 80)
    print("EKSPLORACJA DANYCH - STEAM GAMES DATASET")
    print("=" * 80 + "\n")
    
    # Zaladuj dane
    df = load_raw_data()
    
    # Analizuj kolumny
    get_columns_info(df)
    
    # Szczegolowa analiza
    analyze_selected_data(df)
    
    # Zapisz raport
    save_exploration_summary(df)
    
    print("\n" + "=" * 80)
    print("[OK] EKSPLORACJA UKONCZNA")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
