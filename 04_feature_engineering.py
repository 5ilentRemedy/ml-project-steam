import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import sys, io

if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        input_file = data_dir / "games_cleaned.csv"
        self.df = pd.read_csv(input_file, index_col=False)
        self.df['Release date'] = pd.to_datetime(self.df['Release date'])
        logger.info(f"Zaladowano dane: {self.df.shape}")
        return self.df
    
    def create_date_features(self):
        logger.info("Tworzenie cech z daty...")
        self.df['Release_year'] = self.df['Release date'].dt.year
        self.df['Release_month'] = self.df['Release date'].dt.month
        self.df['Release_quarter'] = self.df['Release date'].dt.quarter
        self.df['Days_since_release'] = (pd.Timestamp.now() - self.df['Release date']).dt.days
        self.df['Is_recent'] = (self.df['Days_since_release'] < 365).astype(int)
        logger.info("[OK] Cechy daty utworzone")
        return self.df
    
    def create_platform_features(self):
        logger.info("Tworzenie cech platform...")
        self.df['Platform_count'] = self.df['Windows'].astype(int) + self.df['Mac'].astype(int) + self.df['Linux'].astype(int)
        self.df['Has_windows'] = self.df['Windows'].astype(int)
        self.df['Has_mac'] = self.df['Mac'].astype(int)
        self.df['Has_linux'] = self.df['Linux'].astype(int)
        self.df['Is_multiplatform'] = (self.df['Platform_count'] > 1).astype(int)
        logger.info(f"Srednia liczba platform: {self.df['Platform_count'].mean():.2f}")
        logger.info("[OK] Cechy platform utworzone")
        return self.df
    
    def create_review_features(self):
        logger.info("Tworzenie cech recenzji...")
        self.df['Total_reviews'] = self.df['Positive'] + self.df['Negative']
        self.df['Review_ratio'] = self.df['Review_ratio'].fillna(0.5)
        self.df['Is_heavily_reviewed'] = (self.df['Total_reviews'] > self.df['Total_reviews'].quantile(0.75)).astype(int)
        self.df['Log_total_reviews'] = np.log1p(self.df['Total_reviews'])
        logger.info(f"Srednia recenzji: {self.df['Total_reviews'].mean():.0f}")
        logger.info("[OK] Cechy recenzji utworzone")
        return self.df
    
    def create_score_features(self):
        logger.info("Tworzenie cech ocen...")
        self.df['Has_metacritic'] = (~self.df['Metacritic score'].isna()).astype(int)
        self.df['Metacritic_category'] = pd.cut(
            self.df['Metacritic score'], bins=[0, 50, 70, 80, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent'], include_lowest=True
        ).astype(str)
        self.df['Metacritic_category'] = self.df['Metacritic_category'].fillna('Unknown')
        self.df['Is_highly_rated'] = (self.df['Metacritic score'] >= 75).astype(int)
        self.df['User_score_normalized'] = self.df['User score'] / 10.0
        logger.info("[OK] Cechy ocen utworzone")
        return self.df
    
    def create_content_features(self):
        logger.info("Tworzenie cech zawartosci...")
        self.df['Log_achievements'] = np.log1p(self.df['Achievements'])
        self.df['Has_achievements'] = (self.df['Achievements'] > 0).astype(int)
        self.df['Log_owners'] = np.log1p(self.df['Estimated owners'])
        self.df['Genre_count'] = self.df['Genres'].str.count(',') + 1
        self.df['Genre_count'] = self.df['Genre_count'].fillna(0).astype(int)
        self.df['Category_count'] = self.df['Categories'].str.count(',') + 1
        self.df['Category_count'] = self.df['Category_count'].fillna(0).astype(int)
        logger.info(f"Srednia genre'ow: {self.df['Genre_count'].mean():.2f}")
        logger.info(f"Srednia kategorii: {self.df['Category_count'].mean():.2f}")
        logger.info("[OK] Cechy zawartosci utworzone")
        return self.df
    
    def create_price_features(self):
        logger.info("Tworzenie cech ceny...")
        self.df['Is_free'] = (self.df['Price'] == 0).astype(int)
        self.df['Price_category'] = pd.cut(
            self.df['Price'], bins=[-0.01, 0, 9.99, 19.99, 49.99, float('inf')],
            labels=['Free', 'Budget', 'Standard', 'Premium', 'AAA'], include_lowest=True
        ).astype(str)
        self.df['Price_category'] = self.df['Price_category'].fillna('Unknown')
        paid_mask = self.df['Price'] > 0
        self.df['Log_price'] = 0.0
        self.df.loc[paid_mask, 'Log_price'] = np.log1p(self.df.loc[paid_mask, 'Price'])
        logger.info(f"Bezplatne gry: {self.df['Is_free'].sum()}")
        logger.info(f"Srednia cena: ${self.df['Price'].mean():.2f}")
        logger.info("[OK] Cechy ceny utworzone")
        return self.df
    
    def encode_categorical_features(self):
        logger.info("Kodowanie zmiennych kategorycznych...")
        categorical_cols = ['Metacritic_category', 'Price_category']
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                self.df = pd.concat([self.df, dummies], axis=1)
                logger.info(f"  [OK] {col} zakodowana ({dummies.shape[1]} kolumn)")
        return self.df
    
    def normalize_numeric_features(self):
        logger.info("Normalizowanie cech numerycznych...")
        numeric_features = ['Price', 'Metacritic score', 'User score', 'Positive', 'Negative', 'Review_ratio', 'Achievements', 'Days_since_release']
        numeric_features = [col for col in numeric_features if col in self.df.columns]
        normalized_data = self.scaler.fit_transform(self.df[numeric_features].fillna(0))
        for i, col in enumerate(numeric_features):
            self.df[f'{col}_normalized'] = normalized_data[:, i]
        logger.info(f"  Znormalizowano {len(numeric_features)} cech")
        return self.df
    
    def create_interaction_features(self):
        logger.info("Tworzenie cech interakcji...")
        self.df['Price_Rating_ratio'] = self.df['Price'] / (self.df['Metacritic score'] + 1)
        self.df['Rating_Review_score'] = (self.df['Metacritic score'] / 100 * self.df['Review_ratio'])
        self.df['Owners_Review_ratio'] = np.log1p(self.df['Estimated owners']) * self.df['Review_ratio']
        logger.info("[OK] Cechy interakcji utworzone")
        return self.df
    
    def save_engineered_data(self):
        output_dir = Path(__file__).parent / "data"
        output_file = output_dir / "games_engineered.csv"
        self.df.to_csv(output_file, index=False)
        logger.info(f"\n[OK] Dane z cechami zapisane: {output_file.name}")
        logger.info(f"  Rozmiar: {self.df.shape}")
        logger.info(f"  Kolumn: {self.df.shape[1]}")
        return self.df
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("INZYNIERIA CECH")
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
        logger.info("[OK] INZYNIERIA CECH UKOŃCZONA")
        logger.info("=" * 80 + "\n")
        return self.df

def main():
    engineer = FeatureEngineer()
    engineer.run()

if __name__ == "__main__":
    main()
