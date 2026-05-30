"""
03_data_cleaning.py
Czyszczenie danych, usuwanie szumu rynkowego oraz inżynieria zmiennej docelowej.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import io

# UTF-8 encoding dla poprawnego wyświetlania znaków w konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.df = None
        self.selected_columns = [
            'AppID', 'Name', 'Release date', 'Price', 'Windows', 'Mac', 'Linux',
            'Metacritic score', 'Achievements', 'Developers', 'Publishers',
            'Categories', 'Genres', 'User score', 'Score rank', 'Positive',
            'Negative', 'Estimated owners'
        ]
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                           if f.stem != "games_cleaned" and f.stem != "games_engineered"])
        if not csv_files:
            raise FileNotFoundError("Nie znaleziono pliku CSV surowych danych")
        self.df = pd.read_csv(csv_files[-1], index_col=False)
        logger.info(f"Załadowano surowe dane: {self.df.shape} z pliku {csv_files[-1].name}")
        return self.df
    
    def select_columns(self):
        logger.info("Wybieranie istotnych kolumn...")
        self.df = self.df[self.selected_columns].copy()
        return self.df
    
    def handle_duplicates(self):
        logger.info("Sprawdzanie duplikatów identyfikatorów AppID...")
        duplicates = self.df['AppID'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Znaleziono {duplicates} duplikujących się AppID - usuwanie...")
            self.df = self.df.drop_duplicates(subset=['AppID'], keep='first')
        return self.df
    
    def clean_price(self):
        logger.info("Czyszczenie formatów cenowych...")
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df['Price'] = self.df['Price'].fillna(0).clip(lower=0).round(2)
        return self.df
    
    def clean_release_date(self):
        logger.info("Czyszczenie i konwersja dat wydania...")
        self.df['Release date'] = pd.to_datetime(self.df['Release date'], format='%b %d, %Y', errors='coerce')
        initial_count = len(self.df)
        self.df = self.df[self.df['Release date'].notna()]
        removed = initial_count - len(self.df)
        if removed > 0:
            logger.warning(f"Usunięto {removed} rekordów ze względu na brakującą datę")
        return self.df
    
    def clean_platforms(self):
        logger.info("Konwersja flag systemów operacyjnych na typ logiczny...")
        for platform in ['Windows', 'Mac', 'Linux']:
            self.df[platform] = self.df[platform].notna().astype(bool)
        return self.df
    
    def clean_scores(self):
        logger.info("Czyszczenie i standaryzacja ocen krytyków oraz graczy...")
        if 'Metacritic score' in self.df.columns:
            self.df['Metacritic score'] = pd.to_numeric(self.df['Metacritic score'], errors='coerce')
        if 'User score' in self.df.columns:
            self.df['User score'] = pd.to_numeric(self.df['User score'], errors='coerce')
        return self.df
    
    def clean_reviews(self):
        logger.info("Przetwarzanie statystyk recenzji użytkowników...")
        for col in ['Positive', 'Negative']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int).clip(lower=0)
        
        self.df['Total_Reviews'] = self.df['Positive'] + self.df['Negative']
        self.df['Review_ratio'] = np.where(self.df['Total_Reviews'] > 0, self.df['Positive'] / self.df['Total_Reviews'], np.nan)
        return self.df

    def engineer_market_success(self):
        """🔥 NOWOŚĆ: Dynamiczny podział relatywny na podstawie kwantyli Twoich danych"""
        logger.info("Generowanie zbalansowanej segmentacji 'Sukces Rynkowy'...")
        
        # Odrzucamy martwe gry (mniej niż 3 opinie) jako czysty szum informacyjny
        initial_count = len(self.df)
        self.df = self.df[self.df['Total_Reviews'] >= 3].copy()
        logger.info(f"Odrzucono {initial_count - len(self.df)} martwych gier z zerowym ruchem")

        # Wyliczamy mediany i punkty podziału dynamicznie z bazy danych
        median_reviews = self.df['Total_Reviews'].median()
        median_ratio = self.df['Review_ratio'].median()
        
        logger.info(f"[PROGI DYNAMICZNE] Mediana recenzji: {median_reviews}, Mediana ratio pozytywów: {median_ratio:.2f}")

        # Dynamiczne warunki podziału geometrii rynku
        conditions = [
            (self.df['Total_Reviews'] >= median_reviews) & (self.df['Review_ratio'] >= median_ratio), # Zrozumiały Hit
            (self.df['Total_Reviews'] < median_reviews) & (self.df['Review_ratio'] >= median_ratio),  # Ukryty Diament
            (self.df['Review_ratio'] >= 0.40) & (self.df['Review_ratio'] < median_ratio)              # Przeciętniak
        ]
        choices = ['Zrozumiały Hit', 'Ukryty Diament', 'Przeciętniak']
        self.df['Market_Success'] = np.select(conditions, choices, default='Klapa rynkowa')
        
        # Logowanie zbalansowania klas docelowych
        counts = self.df['Market_Success'].value_counts()
        logger.info("Statystyki zbalansowania klas po eliminacji szumu:")
        for label, count in counts.items():
            logger.info(f"  - Klasa [{label}]: {count} gier")
        return self.df
    
    def clean_text_fields(self):
        logger.info("Czyszczenie i usuwanie białych znaków z pól tekstowych...")
        for col in ['Name', 'Developers', 'Publishers', 'Categories', 'Genres']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str).str.strip()
        return self.df
    
    def clean_numeric_fields(self):
        logger.info("Czyszczenie pozostałych pól numerycznych...")
        for col in ['Achievements', 'Estimated owners']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int).clip(lower=0)
        return self.df
    
    def validate_cleaned_data(self):
        logger.info("\nWERYFIKACJA KOŃCOWA PROCESU CZYSZCZENIA:")
        logger.info("-" * 60)
        missing = self.df[['Price', 'Market_Success']].isna().sum()
        logger.info(f"Brakujące wartości w kluczowych kolumnach: \n{missing}")
        logger.info(f"Rozmiar końcowy gotowego zbioru danych: {self.df.shape}")
        return self.df
    
    def save_cleaned_data(self):
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "games_cleaned.csv"
        self.df.to_csv(output_file, index=False)
        logger.info(f"[OK] Zapisano zbalansowany plik do: {output_file.name}")
        return self.df
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("START POTOKU CZYSZCZENIA DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.select_columns()
        self.handle_duplicates()
        self.clean_price()
        self.clean_release_date()
        self.clean_platforms()
        self.clean_scores()
        self.clean_reviews()
        self.engineer_market_success()
        self.clean_text_fields()
        self.clean_numeric_fields()
        self.validate_cleaned_data()
        self.save_cleaned_data()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] POTOK CZYSZCZENIA UKOŃCZONY POMYŚLNIE")
        logger.info("=" * 80 + "\n")
        return self.df

def main():
    cleaner = DataCleaner()
    cleaner.run()

if __name__ == "__main__":
    main()