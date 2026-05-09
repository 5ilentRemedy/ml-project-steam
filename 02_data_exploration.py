# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do oper operacji na ścieżkach plików
import json # Do pracy z formatem JSON
import sys # Do interakcji z systemem operacyjnym
import io # Do obsługi strumieni wejścia/wyjścia

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Definicja funkcji do ładowania surowych danych
def load_raw_data():
    # Ustalenie ścieżki do katalogu 'data'
    data_dir = Path(__file__).parent / "data"
    
    # Wyszukiwanie plików CSV z surowymi danymi, ignorując pliki przetworzone
    csv_files = sorted([f for f in data_dir.glob("games_*.csv") if f.stem != "games_cleaned" and f.stem != "games_engineered"])
    
    # Sprawdzenie, czy znaleziono pliki CSV
    if not csv_files:
        # Jeśli nie znaleziono plików, próba pobrania danych za pomocą skryptu 01_data_collection.py
        print("Brak plikow CSV w katalogu data/. Sprobuje pobrac dane...")
        try:
            # Dynamiczne ładowanie skryptu 01_data_collection.py
            import importlib.util
            spec = importlib.util.spec_from_file_location("data_collection", Path(__file__).parent / "01_data_collection.py")
            data_collection = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_collection)
        except Exception as e:
            print(f"Nie udalo sie zaladowac skryptu 01_data_collection: {e}")
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ i nie mozna pobrac danych.")

        try:
            # Wywołanie funkcji pobierającej dane
            data_collection.download_and_move_dataset()
        except Exception as e:
            print(f"Blad podczas pobierania danych: {e}")
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ i pobieranie zakonczylo sie bledem.")

        # Ponowne wyszukanie plików CSV po próbie pobrania
        csv_files = sorted([f for f in data_dir.glob("games_*.csv") if f.stem != "games_cleaned" and f.stem != "games_engineered"])

        # Ostateczne sprawdzenie, czy pliki zostały pobrane
        if not csv_files:
            raise FileNotFoundError("Nie znaleziono pliku CSV w katalogu data/ po probie pobrania danych")
    
    # Wybór najnowszego pliku CSV
    latest_file = csv_files[-1]
    print(f"Ladowanie danych z: {latest_file.name}")
    
    # Wczytanie danych do DataFrame
    df = pd.read_csv(latest_file, index_col=False)
    print(f"\nOK Zaladowano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn\n")
    
    return df

# Definicja funkcji do analizy informacji o kolumnach
def get_columns_info(df):
    print("=" * 80)
    print("INFORMACJE O KOLUMNACH")
    print("=" * 80)
    
    # Lista kolumn, które będą brane pod uwagę w dalszym przetwarzaniu
    selected_columns = [
        'AppID', 'Name', 'Release date', 'Price', 'Windows', 'Mac', 'Linux',
        'Metacritic score', 'Achievements', 'Developers', 'Publishers',
        'Categories', 'Genres', 'User score', 'Score rank', 'Positive',
        'Negative', 'Estimated owners'
    ]
    
    print(f"\nKolumny do preprocessingu ({len(selected_columns)}):")
    print("-" * 80)
    
    # Iteracja przez wybrane kolumny i wyświetlanie ich podstawowych informacji
    for col in selected_columns:
        if col in df.columns:
            dtype = df[col].dtype # Typ danych kolumny
            missing = df[col].isna().sum() # Liczba brakujących wartości
            missing_pct = (missing / len(df)) * 100 # Procent brakujących wartości
            
            try:
                # Próba pobrania przykładowej wartości (średnia dla numerycznych, pierwsza dla pozostałych)
                import pandas as pd
                if pd.api.types.is_numeric_dtype(df[col]):
                    sample = df[col].dropna().mean() if missing < len(df) else "N/A"
                else:
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            except Exception as e:
                sample = "N/A" # W przypadku błędu ustawienie na "N/A"
            
            # Wyświetlenie sformatowanych informacji o kolumnie
            print(f"  * {col:25} | Type: {str(dtype):10} | Braki: {missing:7} ({missing_pct:5.1f}%) | Sample: {sample}")
        else:
            # Informacja o kolumnach, które nie istnieją w DataFrame
            print(f"  * {col:25} | ! KOLUMNA NIE ISTNIEJE")

# Definicja funkcji do szczegółowej analizy wybranych danych
def analyze_selected_data(df):
    print("\n" + "=" * 80)
    print("ANALIZA")
    print("=" * 80)
    
    # Analiza kolumny AppID
    print(f"\nDATA AppID:")
    print(f"  - Unikalne wartosci: {df['AppID'].nunique()}")
    print(f"  - Duplikaty: {df['AppID'].duplicated().sum()}")
    
    # Analiza kolumny Price
    print(f"\nPRICE Price:")
    print(f"  - Typ danych: {df['Price'].dtype}")
    print(f"  - Braki: {df['Price'].isna().sum()}")
    print(f"  - Bezplatne gry (0): {(df['Price'] == 0).sum()}")
    print(f"  - Min cena: ${df['Price'].min()}")
    print(f"  - Max cena: ${df['Price'].max()}")
    print(f"  - Srednia cena: ${df['Price'].mean():.2f}")
    print(f"  - Mediana ceny: ${df['Price'].median():.2f}")
    
    # Analiza obsługi platform
    print(f"\nPLATFORM Obsluga platform:")
    print(f"  - Windows: {df['Windows'].notna().sum()} ({(df['Windows'].notna().sum()/len(df)*100):.1f}%)")
    print(f"  - Mac: {df['Mac'].notna().sum()} ({(df['Mac'].notna().sum()/len(df)*100):.1f}%)")
    print(f"  - Linux: {df['Linux'].notna().sum()} ({(df['Linux'].notna().sum()/len(df)*100):.1f}%)")
    
    # Analiza ocen (Metacritic, User score)
    print(f"\nSCORE Oceny:")
    print(f"  - Metacritic score braki: {df['Metacritic score'].isna().sum()}")
    metacritic = pd.to_numeric(df['Metacritic score'], errors='coerce') # Konwersja na typ numeryczny, błędy jako NaN
    print(f"  - Metacritic score - srednia: {metacritic.mean():.1f}")
    print(f"  - User score braki: {df['User score'].isna().sum()}")
    user_score = pd.to_numeric(df['User score'], errors='coerce') # Konwersja na typ numeryczny, błędy jako NaN
    print(f"  - User score - srednia: {user_score.mean():.2f}")
    
    # Analiza recenzji (Positive, Negative)
    print(f"\nREVIEW Recenzje:")
    print(f"  - Positive braki: {df['Positive'].isna().sum()}")
    positive = pd.to_numeric(df['Positive'], errors='coerce') # Konwersja na typ numeryczny, błędy jako NaN
    print(f"  - Positive - srednia: {positive.mean():.0f}")
    print(f"  - Negative braki: {df['Negative'].isna().sum()}")
    negative = pd.to_numeric(df['Negative'], errors='coerce') # Konwersja na typ numeryczny, błędy jako NaN
    print(f"  - Negative - srednia: {negative.mean():.0f}")
    
    # Analiza metadanych
    print(f"\nMETA Metadane:")
    print(f"  - Developers braki: {df['Developers'].isna().sum()}")
    print(f"  - Publishers braki: {df['Publishers'].isna().sum()}")
    print(f"  - Categories braki: {df['Categories'].isna().sum()}")
    print(f"  - Genres braki: {df['Genres'].isna().sum()}")
    print(f"  - Achievements braki: {df['Achievements'].isna().sum()}")

# Definicja funkcji do zapisywania podsumowania eksploracji
def save_exploration_summary(df):
    # Ustalenie ścieżki do katalogu 'reports' i utworzenie go, jeśli nie istnieje
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    
    # Tworzenie słownika z podsumowaniem danych
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "selected_columns": 18, # Liczba kolumn wybranych do dalszego przetwarzania
        "missing_values": df.isna().sum().to_dict(), # Liczba brakujących wartości dla każdej kolumny
        "column_types": df.dtypes.astype(str).to_dict(), # Typy danych dla każdej kolumny
        "numeric_summary": df.describe().to_dict() # Statystyki opisowe dla kolumn numerycznych
    }
    
    # Zapisanie podsumowania do pliku JSON
    output_file = output_dir / "01_exploration_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nOK Raport eksploracji zapisany: reports/01_exploration_summary.json")

# Główna funkcja skryptu
def main():
    print("\n" + "=" * 80)
    print("EKSPLORACJA DANYCH - STEAM GAMES DATASET")
    print("=" * 80 + "\n")
    
    # Załadowanie surowych danych
    df = load_raw_data()
    
    # Wyświetlenie informacji o kolumnach
    get_columns_info(df)
    
    # Przeprowadzenie szczegółowej analizy wybranych danych
    analyze_selected_data(df)
    
    # Zapisanie podsumowania eksploracji do pliku JSON
    save_exploration_summary(df)
    
    print("\n" + "=" * 80)
    print("OK EKSPLORACJA UKONCZNA")
    print("=" * 80 + "\n")

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
