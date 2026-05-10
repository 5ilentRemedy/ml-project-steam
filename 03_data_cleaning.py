# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import logging # Do logowania informacji
import sys, io # Do obsługi strumieni wejścia/wyjścia

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do czyszczenia danych
class DataCleaner:
    # Inicjalizacja obiektu DataCleaner
    def __init__(self):
        self.df = None # Ramka danych, która będzie czyszczona
        # Lista kolumn, które zostaną wybrane do dalszego przetwarzania
        self.selected_columns = [
            'AppID', 'Name', 'Release date', 'Price', 'Windows', 'Mac', 'Linux',
            'Metacritic score', 'Achievements', 'Developers', 'Publishers',
            'Categories', 'Genres', 'User score', 'Score rank', 'Positive',
            'Negative', 'Estimated owners'
        ]
    
    # Metoda do ładowania surowych danych
    def load_data(self):
        # Ustalenie ścieżki do katalogu 'data'
        data_dir = Path(__file__).parent / "data"
        # Wyszukiwanie plików CSV z surowymi danymi, ignorując pliki już przetworzone
        csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                           if f.stem != "games_cleaned" and f.stem != "games_engineered"])
        # Sprawdzenie, czy znaleziono pliki CSV
        if not csv_files:
            raise FileNotFoundError("Nie znaleziono pliku CSV surowych danych")
        # Wczytanie najnowszego pliku CSV do DataFrame
        self.df = pd.read_csv(csv_files[-1], index_col=False)
        logger.info(f"Zaladowano dane: {self.df.shape} z {csv_files[-1].name}")
        return self.df
    
    # Metoda do wybierania określonych kolumn
    def select_columns(self):
        logger.info("Wybieranie kolumn...")
        # Wybranie tylko kolumn z listy selected_columns
        self.df = self.df[self.selected_columns].copy()
        logger.info(f"Wybrano {len(self.selected_columns)} kolumn")
        return self.df
    
    # Metoda do obsługi duplikatów
    def handle_duplicates(self):
        logger.info("Sprawdzanie duplikatow...")
        # Zliczenie duplikatów na podstawie kolumny 'AppID'
        duplicates = self.df['AppID'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Znaleziono {duplicates} duplikujacych AppID - usuwanie...")
            # Usunięcie duplikatów, zachowując pierwszy napotkany rekord
            self.df = self.df.drop_duplicates(subset=['AppID'], keep='first')
        logger.info(f"Dane po usunieciu duplikatow: {self.df.shape}")
        return self.df
    
    # Metoda do czyszczenia danych w kolumnie 'Price'
    def clean_price(self):
        logger.info("Czyszczenie cen...")
        # Konwersja kolumny 'Price' na typ numeryczny, błędy jako NaN
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        # Wypełnienie brakujących wartości (NaN) zerami
        self.df['Price'] = self.df['Price'].fillna(0)
        # Obcięcie wartości poniżej zera do zera
        self.df['Price'] = self.df['Price'].clip(lower=0)
        # Zaokrąglenie cen do dwóch miejsc po przecinku
        self.df['Price'] = self.df['Price'].round(2)
        logger.info(f"Ceny: min={self.df['Price'].min()}, max={self.df['Price'].max()}")
        return self.df
    
    # Metoda do czyszczenia danych w kolumnie 'Release date'
    def clean_release_date(self):
        logger.info("Czyszczenie dat...")
        # Konwersja kolumny 'Release date' na format datetime, błędy jako NaN
        self.df['Release date'] = pd.to_datetime(self.df['Release date'], format='%b %d, %Y', errors='coerce')
        initial_count = len(self.df) # Początkowa liczba wierszy
        # Usunięcie wierszy z brakującymi datami wydania
        self.df = self.df[self.df['Release date'].notna()]
        removed = initial_count - len(self.df) # Liczba usuniętych wierszy
        if removed > 0:
            logger.warning(f"Usunieto {removed} gier bez daty wydania")
        logger.info(f"Daty: od {self.df['Release date'].min().date()} do {self.df['Release date'].max().date()}")
        return self.df
    
    # Metoda do czyszczenia danych dotyczących platform
    def clean_platforms(self):
        logger.info("Czyszczenie platform...")
        # TODO #1 pozostawienie platform w osobnych kolumnach i oznaczenie ich jako True/False
        # Konwersja kolumn platform na typ boolean (True/False) w zależności od istnienia wartości
        for platform in ['Windows', 'Mac', 'Linux']:
            self.df[platform] = self.df[platform].notna().astype(bool)
        logger.info(f"Windows: {self.df['Windows'].sum()} gier")
        logger.info(f"Mac: {self.df['Mac'].sum()} gier")
        logger.info(f"Linux: {self.df['Linux'].sum()} gier")
        return self.df
    
    # Metoda do czyszczenia danych dotyczących ocen
    def clean_scores(self):
        logger.info("Czyszczenie ocen...")
        # Czyszczenie 'Metacritic score'
        if 'Metacritic score' in self.df.columns:
            self.df['Metacritic score'] = pd.to_numeric(self.df['Metacritic score'], errors='coerce')
            self.df['Metacritic score'] = self.df['Metacritic score'].clip(0, 100) # Obcięcie do zakresu 0-100
        # Czyszczenie 'User score'
        if 'User score' in self.df.columns:
            self.df['User score'] = pd.to_numeric(self.df['User score'], errors='coerce')
            self.df['User score'] = self.df['User score'].clip(0, 10) # Obcięcie do zakresu 0-10
        # Czyszczenie 'Score rank'
        if 'Score rank' in self.df.columns:
            self.df['Score rank'] = pd.to_numeric(self.df['Score rank'], errors='coerce')
            self.df['Score rank'] = self.df['Score rank'].clip(lower=0) # Obcięcie wartości poniżej zera do zera
        logger.info(f"Metacritic score - braki: {self.df['Metacritic score'].isna().sum()}")
        logger.info(f"User score - braki: {self.df['User score'].isna().sum()}")
        return self.df
    
    # Metoda do czyszczenia danych dotyczących recenzji
    def clean_reviews(self):
        logger.info("Czyszczenie recenzji...")
        # Czyszczenie kolumn 'Positive' i 'Negative'
        for col in ['Positive', 'Negative']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce') # Konwersja na numeryczny
            self.df[col] = self.df[col].fillna(0).astype(int) # Wypełnienie NaN zerami i konwersja na int
            self.df[col] = self.df[col].clip(lower=0) # Obcięcie wartości poniżej zera do zera
        # Obliczenie całkowitej liczby recenzji
        total_reviews = self.df['Positive'] + self.df['Negative']
        # Obliczenie współczynnika recenzji pozytywnych, unikając dzielenia przez zero
        self.df['Review_ratio'] = np.where(total_reviews > 0, self.df['Positive'] / total_reviews, np.nan)
        logger.info(f"Positive - srednia: {self.df['Positive'].mean():.0f}")
        logger.info(f"Negative - srednia: {self.df['Negative'].mean():.0f}")
        return self.df
    
    # Metoda do czyszczenia pól tekstowych
    def clean_text_fields(self):
        logger.info("Czyszczenie pol tekstowych...")
        # Iteracja przez wybrane kolumny tekstowe
        for col in ['Name', 'Developers', 'Publishers', 'Categories', 'Genres']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('') # Wypełnienie brakujących wartości pustym ciągiem
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].str.strip() # Usunięcie białych znaków z początku i końca
        return self.df
    
    # Metoda do czyszczenia pól numerycznych
    def clean_numeric_fields(self):
        logger.info("Czyszczenie pol numerycznych...")
        # Iteracja przez wybrane kolumny numeryczne
        for col in ['Achievements', 'Estimated owners']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce') # Konwersja na numeryczny
                self.df[col] = self.df[col].fillna(0) # Wypełnienie brakujących wartości zerami
                self.df[col] = self.df[col].astype(int).clip(lower=0) # Konwersja na int i obcięcie do wartości nieujemnych
        logger.info(f"Achievements - srednia: {self.df['Achievements'].mean():.0f}")
        logger.info(f"Estimated owners - srednia: {self.df['Estimated owners'].mean():.0f}")
        return self.df
    
    # Metoda do walidacji danych po czyszczeniu
    def validate_cleaned_data(self):
        logger.info("\nWERYFIKACJA CZYSZCZENIA:")
        logger.info("-" * 50)
        missing = self.df.isna().sum() # Zliczenie brakujących wartości
        if missing.sum() > 0:
            logger.warning(f"Brakujace wartosci:\n{missing[missing > 0]}")
        else:
            logger.info("OK Brak wartosci brakujacych")
        dups = self.df['AppID'].duplicated().sum() # Zliczenie duplikatów AppID
        if dups == 0:
            logger.info("OK Brak duplikatow AppID")
        logger.info(f"\nTypy danych:\n{self.df.dtypes}")
        return self.df
    
    # Metoda do zapisywania oczyszczonych danych
    def save_cleaned_data(self):
        # Ustalenie ścieżki do katalogu 'data'
        output_dir = Path(__file__).parent / "data"
        # Definiowanie nazwy pliku wyjściowego
        output_file = output_dir / "games_cleaned.csv"
        # Zapisanie DataFrame do pliku CSV
        self.df.to_csv(output_file, index=False)
        logger.info(f"\nOK Oczyszczone dane zapisane: {output_file.name}")
        logger.info(f"  Rozmiar: {self.df.shape}")
        return self.df
    
    # Metoda uruchamiająca cały proces czyszczenia danych
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("CZYSZCZENIE DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data() # Załadowanie danych
        self.select_columns() # Wybranie kolumn
        self.handle_duplicates() # Obsługa duplikatów
        self.clean_price() # Czyszczenie cen
        self.clean_release_date() # Czyszczenie dat wydania
        self.clean_platforms() # Czyszczenie danych platform
        self.clean_scores() # Czyszczenie ocen
        self.clean_reviews() # Czyszczenie recenzji
        self.clean_text_fields() # Czyszczenie pól tekstowych
        self.clean_numeric_fields() # Czyszczenie pól numerycznych
        self.validate_cleaned_data() # Walidacja oczyszczonych danych
        self.save_cleaned_data() # Zapisanie oczyszczonych danych
        logger.info("\n" + "=" * 80)
        logger.info("OK CZYSZCZENIE UKONCZONE")
        logger.info("=" * 80 + "\n")
        return self.df

# Główna funkcja programu
def main():
    cleaner = DataCleaner() # Utworzenie instancji klasy DataCleaner
    cleaner.run() # Uruchomienie procesu czyszczenia

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
