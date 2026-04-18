"""
03_data_cleaning.py - Czyszczenie i standaryzacja danych
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys, io

if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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
        # Load raw data (games_YYYYMMDD_HHMMSS.csv) - NOT processed files
        csv_files = sorted([f for f in data_dir.glob("games_*.csv") 
                           if f.stem != "games_cleaned" and f.stem != "games_engineered"])
        if not csv_files:
            raise FileNotFoundError("Nie znaleziono pliku CSV surowych danych")
        self.df = pd.read_csv(csv_files[-1], index_col=False)
        logger.info(f"Zaladowano dane: {self.df.shape} z {csv_files[-1].name}")
        return self.df
    
    def select_columns(self):
        logger.info("Wybieranie kolumn...")
        self.df = self.df[self.selected_columns].copy()
        logger.info(f"Wybrano {len(self.selected_columns)} kolumn")
        return self.df
    
    def handle_duplicates(self):
        logger.info("Sprawdzanie duplikatow...")
        duplicates = self.df['AppID'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Znaleziono {duplicates} duplikujacych AppID - usuwanie...")
            self.df = self.df.drop_duplicates(subset=['AppID'], keep='first')
        logger.info(f"Dane po usunięciu duplikatow: {self.df.shape}")
        return self.df
    
    def clean_price(self):
        logger.info("Czyszczenie cen...")
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df['Price'] = self.df['Price'].fillna(0)
        self.df['Price'] = self.df['Price'].clip(lower=0)
        self.df['Price'] = self.df['Price'].round(2)
        logger.info(f"Ceny: min={self.df['Price'].min()}, max={self.df['Price'].max()}")
        return self.df
    
    def clean_release_date(self):
        logger.info("Czyszczenie dat...")
        self.df['Release date'] = pd.to_datetime(self.df['Release date'], format='%b %d, %Y', errors='coerce')
        initial_count = len(self.df)
        self.df = self.df[self.df['Release date'].notna()]
        removed = initial_count - len(self.df)
        if removed > 0:
            logger.warning(f"Usunieto {removed} gier bez daty wydania")
        logger.info(f"Daty: od {self.df['Release date'].min().date()} do {self.df['Release date'].max().date()}")
        return self.df
    
    def clean_platforms(self):
        logger.info("Czyszczenie platform...")
        for platform in ['Windows', 'Mac', 'Linux']:
            self.df[platform] = self.df[platform].notna().astype(bool)
        logger.info(f"Windows: {self.df['Windows'].sum()} gier")
        logger.info(f"Mac: {self.df['Mac'].sum()} gier")
        logger.info(f"Linux: {self.df['Linux'].sum()} gier")
        return self.df
    
    def clean_scores(self):
        logger.info("Czyszczenie ocen...")
        if 'Metacritic score' in self.df.columns:
            self.df['Metacritic score'] = pd.to_numeric(self.df['Metacritic score'], errors='coerce')
            self.df['Metacritic score'] = self.df['Metacritic score'].clip(0, 100)
        if 'User score' in self.df.columns:
            self.df['User score'] = pd.to_numeric(self.df['User score'], errors='coerce')
            self.df['User score'] = self.df['User score'].clip(0, 10)
        if 'Score rank' in self.df.columns:
            self.df['Score rank'] = pd.to_numeric(self.df['Score rank'], errors='coerce')
            self.df['Score rank'] = self.df['Score rank'].clip(lower=0)
        logger.info(f"Metacritic score - braki: {self.df['Metacritic score'].isna().sum()}")
        logger.info(f"User score - braki: {self.df['User score'].isna().sum()}")
        return self.df
    
    def clean_reviews(self):
        logger.info("Czyszczenie recenzji...")
        for col in ['Positive', 'Negative']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(0).astype(int)
            self.df[col] = self.df[col].clip(lower=0)
        total_reviews = self.df['Positive'] + self.df['Negative']
        self.df['Review_ratio'] = np.where(total_reviews > 0, self.df['Positive'] / total_reviews, np.nan)
        logger.info(f"Positive - srednia: {self.df['Positive'].mean():.0f}")
        logger.info(f"Negative - srednia: {self.df['Negative'].mean():.0f}")
        return self.df
    
    def clean_text_fields(self):
        logger.info("Czyszczenie pol tekstowych...")
        for col in ['Name', 'Developers', 'Publishers', 'Categories', 'Genres']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].str.strip()
        return self.df
    
    def clean_numeric_fields(self):
        logger.info("Czyszczenie pol numerycznych...")
        for col in ['Achievements', 'Estimated owners']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(0)
                self.df[col] = self.df[col].astype(int).clip(lower=0)
        logger.info(f"Achievements - srednia: {self.df['Achievements'].mean():.0f}")
        logger.info(f"Estimated owners - srednia: {self.df['Estimated owners'].mean():.0f}")
        return self.df
    
    def validate_cleaned_data(self):
        logger.info("\nWERYFIKACJA CZYSZCZENIA:")
        logger.info("-" * 50)
        missing = self.df.isna().sum()
        if missing.sum() > 0:
            logger.warning(f"Brakujace wartosci:\n{missing[missing > 0]}")
        else:
            logger.info("[OK] Brak wartosci brakujacych")
        dups = self.df['AppID'].duplicated().sum()
        if dups == 0:
            logger.info("[OK] Brak duplikatow AppID")
        logger.info(f"\nTypy danych:\n{self.df.dtypes}")
        return self.df
    
    def save_cleaned_data(self):
        output_dir = Path(__file__).parent / "data"
        output_file = output_dir / "games_cleaned.csv"
        self.df.to_csv(output_file, index=False)
        logger.info(f"\n[OK] Oczyszczone dane zapisane: {output_file.name}")
        logger.info(f"  Rozmiar: {self.df.shape}")
        return self.df
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("CZYSZCZENIE DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.select_columns()
        self.handle_duplicates()
        self.clean_price()
        self.clean_release_date()
        self.clean_platforms()
        self.clean_scores()
        self.clean_reviews()
        self.clean_text_fields()
        self.clean_numeric_fields()
        self.validate_cleaned_data()
        self.save_cleaned_data()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] CZYSZCZENIE UKONCZNA")
        logger.info("=" * 80 + "\n")
        return self.df

def main():
    cleaner = DataCleaner()
    cleaner.run()

if __name__ == "__main__":
    main()
