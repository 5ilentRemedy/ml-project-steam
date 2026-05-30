"""
03_feature_engineering.py
Inżynieria cech (Feature Engineering) dla modelu klasyfikacji sukcesu rynkowego.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import sys
import io

# UTF-8 encoding dla poprawnego wyświetlania znaków w konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        input_file = data_dir / "games_cleaned.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {input_file}")
            
        self.df = pd.read_csv(input_file, index_col=False)
        self.df['Release date'] = pd.to_datetime(self.df['Release date'])
        logger.info(f"Załadowano dane do inżynierii cech: {self.df.shape}")
        return self.df
    
    def create_date_features(self):
        logger.info("Tworzenie cech z daty wydania...")
        self.df['Release_year'] = self.df['Release date'].dt.year
        self.df['Release_month'] = self.df['Release date'].dt.month
        self.df['Release_quarter'] = self.df['Release date'].dt.quarter
        
        # Używamy stałego punktu odniesienia (rok 2026), aby uniknąć zmian w cechach przy uruchomieniu w przyszłości
        reference_date = pd.Timestamp('2026-05-30')
        self.df['Days_since_release'] = (reference_date - self.df['Release date']).dt.days
        self.df['Is_recent'] = (self.df['Days_since_release'] < 365).astype(int)
        return self.df
    
    def create_platform_features(self):
        logger.info("Tworzenie cech systemów operacyjnych...")
        self.df['Platform_count'] = self.df['Windows'].astype(int) + self.df['Mac'].astype(int) + self.df['Linux'].astype(int)
        self.df['Has_windows'] = self.df['Windows'].astype(int)
        self.df['Has_mac'] = self.df['Mac'].astype(int)
        self.df['Has_linux'] = self.df['Linux'].astype(int)
        self.df['Is_multiplatform'] = (self.df['Platform_count'] > 1).astype(int)
        return self.df
    
    def create_review_features(self):
        logger.info("Transformacja logarytmiczna wolumenu recenzji...")
        # Kolumna Total_Reviews została już poprawnie wyliczona w kroku 02_data_cleaning
        if 'Total_Reviews' not in self.df.columns:
            self.df['Total_Reviews'] = self.df['Positive'] + self.df['Negative']
            
        self.df['Review_ratio'] = self.df['Review_ratio'].fillna(0.5)
        self.df['Log_total_reviews'] = np.log1p(self.df['Total_Reviews'])
        return self.df
    
    def create_score_features(self):
        logger.info("Kategoryzacja ocen zewnętrznych...")
        self.df['Has_metacritic'] = (~self.df['Metacritic score'].isna()).astype(int)
        
        # Zastępujemy wartości NaN medianą lub zerem przed podziałem na koszyki
        temp_score = self.df['Metacritic score'].fillna(0)
        self.df['Metacritic_category'] = pd.cut(
            temp_score, bins=[-1, 0, 50, 70, 80, 100],
            labels=['None', 'Poor', 'Fair', 'Good', 'Excellent'], include_lowest=True
        ).astype(str)
        
        if 'User score' in self.df.columns:
            self.df['User_score_normalized'] = self.df['User score'].fillna(0) / 10.0
        return self.df
    
    def create_content_features(self):
        logger.info("Tworzenie cech zawartości i metadanych gry...")
        self.df['Log_achievements'] = np.log1p(self.df['Achievements'].fillna(0))
        self.df['Has_achievements'] = (self.df['Achievements'] > 0).astype(int)
        self.df['Log_owners'] = np.log1p(self.df['Estimated owners'].fillna(0))
        
        # Liczenie tagów/gatunków na podstawie przecinków
        self.df['Genre_count'] = self.df['Genres'].fillna('').str.count(',') + 1
        self.df['Genre_count'] = np.where(self.df['Genres'].isna() | (self.df['Genres'] == ''), 0, self.df['Genre_count'])
        
        self.df['Category_count'] = self.df['Categories'].fillna('').str.count(',') + 1
        self.df['Category_count'] = np.where(self.df['Categories'].isna() | (self.df['Categories'] == ''), 0, self.df['Category_count'])
        return self.df
    
    def create_price_features(self):
        logger.info("Przetwarzanie i logarytmowanie cech cenowych...")
        self.df['Is_free'] = (self.df['Price'] == 0).astype(int)
        self.df['Price_category'] = pd.cut(
            self.df['Price'], bins=[-0.01, 0, 9.99, 19.99, 49.99, float('inf')],
            labels=['Free', 'Budget', 'Standard', 'Premium', 'AAA'], include_lowest=True
        ).astype(str)
        
        paid_mask = self.df['Price'] > 0
        self.df['Log_price'] = 0.0
        self.df.loc[paid_mask, 'Log_price'] = np.log1p(self.df.loc[paid_mask, 'Price'])
        return self.df
    
    def encode_categorical_features(self):
        logger.info("Kodowanie zmiennych kategorycznych (One-Hot Encoding)...")
        categorical_cols = ['Metacritic_category', 'Price_category']
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                self.df = pd.concat([self.df, dummies], axis=1)
                logger.info(f"  -> Utworzono zmienne dummy dla: {col}")
        return self.df
    
    def normalize_numeric_features(self):
        logger.info("Standaryzacja (Z-score) cech numerycznych...")
        numeric_features = ['Price', 'Metacritic score', 'Positive', 'Negative', 'Review_ratio', 'Achievements', 'Days_since_release']
        available_features = [col for col in numeric_features if col in self.df.columns]
        
        normalized_data = self.scaler.fit_transform(self.df[available_features].fillna(0))
        for i, col in enumerate(available_features):
            self.df[f'{col}_normalized'] = normalized_data[:, i]
        return self.df
    
    def create_interaction_features(self):
        logger.info("Generowanie cech interakcji...")
        # Wypełniamy braki w Metacritic medianą na potrzeby interakcji
        meta_filled = self.df['Metacritic score'].fillna(self.df['Metacritic score'].median() if self.df['Metacritic score'].notna().any() else 60)
        
        self.df['Price_Rating_ratio'] = self.df['Price'] / (meta_filled + 1)
        self.df['Rating_Review_score'] = (meta_filled / 100 * self.df['Review_ratio'])
        self.df['Owners_Review_ratio'] = np.log1p(self.df['Estimated owners'].fillna(0)) * self.df['Review_ratio']
        return self.df
    
    def save_engineered_data(self):
        output_dir = Path(__file__).parent / "data"
        output_file = output_dir / "games_engineered.csv"
        self.df.to_csv(output_file, index=False)
        logger.info(f"[OK] Dane z cechami zapisane w: {output_file.name}")
        logger.info(f"     Struktura końcowa: {self.df.shape[0]} wierszy, {self.df.shape[1]} kolumn")
        return self.df
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("START PROCESU INŻYNIERII CECH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.create_date_features()
        self.create_platform_features()
        self.create_review_features()
        self.create_score_features()
        self.create_content_features()
        self.create_price_features()
        self.encode_categorical_features()
        self.normalize_numeric_features()
        self.create_interaction_features()
        self.save_engineered_data()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] INŻYNIERIA CECH UKOŃCZONA POMYŚLNIE")
        logger.info("=" * 80 + "\n")
        return self.df

def main():
    engineer = FeatureEngineer()
    engineer.run()

if __name__ == "__main__":
    main()