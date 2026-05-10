# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
from sklearn.preprocessing import StandardScaler # Do skalowania cech numerycznych
import logging # Do logowania informacji
import sys, io # Do obsługi strumieni wejścia/wyjścia

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do inżynierii cech
class FeatureEngineer:
    # Inicjalizacja obiektu FeatureEngineer
    def __init__(self):
        self.df = None # Ramka danych, która będzie przetwarzana
        self.scaler = StandardScaler() # Obiekt do skalowania cech numerycznych
        self.label_encoders = {} # Słownik do przechowywania LabelEncoderów (nieużywane w obecnej wersji, ale zachowane)
    
    # Metoda do ładowania oczyszczonych danych
    def load_data(self):
        # Ustalenie ścieżki do katalogu 'data'
        data_dir = Path(__file__).parent / "data"
        # Definiowanie ścieżki do pliku z oczyszczonymi danymi
        input_file = data_dir / "games_cleaned.csv"
        # Wczytanie danych do DataFrame
        self.df = pd.read_csv(input_file, index_col=False)
        # Konwersja kolumny 'Release date' na typ datetime
        self.df['Release date'] = pd.to_datetime(self.df['Release date'])
        logger.info(f"Zaladowano dane: {self.df.shape}")
        return self.df
    
    # Metoda do tworzenia cech związanych z datą wydania
    def create_date_features(self):
        logger.info("Tworzenie cech z daty...")
        # Wyodrębnienie roku wydania
        self.df['Release_year'] = self.df['Release date'].dt.year
        # Wyodrębnienie miesiąca wydania
        self.df['Release_month'] = self.df['Release date'].dt.month
        # Wyodrębnienie kwartału wydania
        self.df['Release_quarter'] = self.df['Release date'].dt.quarter
        # Obliczenie liczby dni od daty wydania do dzisiaj
        self.df['Days_since_release'] = (pd.Timestamp.now() - self.df['Release date']).dt.days
        # Utworzenie cechy binarnej 'Is_recent' (czy gra została wydana w ciągu ostatniego roku)
        self.df['Is_recent'] = (self.df['Days_since_release'] < 365).astype(int)
        logger.info("OK Cechy daty utworzone")
        return self.df
    
    # Metoda do tworzenia cech związanych z platformami
    def create_platform_features(self):
        logger.info("Tworzenie cech platform...")
        # Obliczenie liczby platform, na których dostępna jest gra
        self.df['Platform_count'] = self.df['Windows'].astype(int) + self.df['Mac'].astype(int) + self.df['Linux'].astype(int)
        # Utworzenie cech binarnych dla każdej platformy
        self.df['Has_windows'] = self.df['Windows'].astype(int)
        self.df['Has_mac'] = self.df['Mac'].astype(int)
        self.df['Has_linux'] = self.df['Linux'].astype(int)
        # Utworzenie cechy binarnej 'Is_multiplatform' (czy gra jest dostępna na więcej niż jednej platformie)
        self.df['Is_multiplatform'] = (self.df['Platform_count'] > 1).astype(int)
        logger.info(f"Srednia liczba platform: {self.df['Platform_count'].mean():.2f}")
        logger.info("OK Cechy platform utworzone")
        return self.df
    
    # Metoda do tworzenia cech związanych z recenzjami
    def create_review_features(self):
        logger.info("Tworzenie cech recenzji...")
        # Obliczenie całkowitej liczby recenzji
        self.df['Total_reviews'] = self.df['Positive'] + self.df['Negative']
        # Wypełnienie brakujących wartości w 'Review_ratio' medianą (0.5)
        self.df['Review_ratio'] = self.df['Review_ratio'].fillna(0.5)
        # Utworzenie cechy binarnej 'Is_heavily_reviewed' (czy gra ma dużo recenzji - powyżej 75. percentyla)
        self.df['Is_heavily_reviewed'] = (self.df['Total_reviews'] > self.df['Total_reviews'].quantile(0.75)).astype(int)
        # Logarytmiczna transformacja całkowitej liczby recenzji (log1p(x) = log(1+x))
        self.df['Log_total_reviews'] = np.log1p(self.df['Total_reviews'])
        logger.info(f"Srednia recenzji: {self.df['Total_reviews'].mean():.0f}")
        logger.info("OK Cechy recenzji utworzone")
        return self.df
    
    # Metoda do tworzenia cech związanych z ocenami
    def create_score_features(self):
            logger.info("Tworzenie cech ocen...")
            # Utworzenie cechy binarnej 'Has_metacritic' (czy gra ma ocenę Metacritic)
            self.df['Has_metacritic'] = (~self.df['Metacritic score'].isna()).astype(int)
            # Kategoryzacja ocen Metacritic na podstawie przedziałów
            self.df['Metacritic_category'] = pd.cut(
                self.df['Metacritic score'], bins=[0, 50, 70, 80, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent'], include_lowest=True
            ).astype(str)
            # Wypełnienie brakujących kategorii 'Unknown'
            self.df['Metacritic_category'] = self.df['Metacritic_category'].fillna('Unknown')
            
            # Definicja sukcesu gry: 
            # Kryterium 1: Minimum 20 recenzji i co najmniej 70% pozytywnych
            # Kryterium 2: Metacritic score >= 75
            criterion1 = (self.df['Review_ratio'] >= 0.70) & (self.df['Total_reviews'] >= 20)
            criterion2 = (self.df['Metacritic score'] >= 75)
            self.df['Is_highly_rated'] = (criterion1 | criterion2).astype(int)
            
            # Normalizacja oceny użytkowników do zakresu 0-1
            self.df['User_score_normalized'] = self.df['User score'] / 10.0
            
            # Logowanie liczby gier uznanych za sukces
            success_count = self.df['Is_highly_rated'].sum()
            logger.info(f"INFO Gier ze statusem 'Sukces' (Is_highly_rated=1): {success_count} na {len(self.df)}")
            logger.info("OK Cechy ocen utworzone")
            return self.df
    
    # Metoda do tworzenia cech związanych z zawartością gry
    def create_content_features(self):
        logger.info("Tworzenie cech zawartosci...")
        # Logarytmiczna transformacja liczby osiągnięć
        self.df['Log_achievements'] = np.log1p(self.df['Achievements'])
        # Utworzenie cechy binarnej 'Has_achievements' (czy gra ma osiągnięcia)
        self.df['Has_achievements'] = (self.df['Achievements'] > 0).astype(int)
        # Logarytmiczna transformacja liczby szacowanych właścicieli
        self.df['Log_owners'] = np.log1p(self.df['Estimated owners'])
        # Obliczenie liczby gatunków dla każdej gry
        self.df['Genre_count'] = self.df['Genres'].str.count(',') + 1
        self.df['Genre_count'] = self.df['Genre_count'].fillna(0).astype(int)
        # Obliczenie liczby kategorii dla każdej gry
        self.df['Category_count'] = self.df['Categories'].str.count(',') + 1
        self.df['Category_count'] = self.df['Category_count'].fillna(0).astype(int)
        logger.info(f"Srednia genre'ow: {self.df['Genre_count'].mean():.2f}")
        logger.info(f"Srednia kategorii: {self.df['Category_count'].mean():.2f}")
        logger.info("OK Cechy zawartosci utworzone")
        return self.df
    
    # Metoda do tworzenia cech związanych z ceną
    def create_price_features(self):
        logger.info("Tworzenie cech ceny...")
        # Utworzenie cechy binarnej 'Is_free' (czy gra jest darmowa)
        self.df['Is_free'] = (self.df['Price'] == 0).astype(int)
        # Kategoryzacja cen na podstawie przedziałów
        self.df['Price_category'] = pd.cut(
            self.df['Price'], bins=[-0.01, 0, 9.99, 19.99, 49.99, float('inf')],
            labels=['Free', 'Budget', 'Standard', 'Premium', 'AAA'], include_lowest=True
        ).astype(str)
        # Wypełnienie brakujących kategorii 'Unknown'
        self.df['Price_category'] = self.df['Price_category'].fillna('Unknown')
        # Maska dla gier płatnych
        paid_mask = self.df['Price'] > 0
        # Logarytmiczna transformacja ceny dla gier płatnych
        self.df['Log_price'] = 0.0
        self.df.loc[paid_mask, 'Log_price'] = np.log1p(self.df.loc[paid_mask, 'Price'])
        logger.info(f"Bezplatne gry: {self.df['Is_free'].sum()}")
        logger.info(f"Srednia cena: ${self.df['Price'].mean():.2f}")
        logger.info("OK Cechy ceny utworzone")
        return self.df
    
    # Metoda do kodowania zmiennych kategorycznych (One-Hot Encoding)
    def encode_categorical_features(self):
        logger.info("Kodowanie zmiennych kategorycznych...")
        categorical_cols = ['Metacritic_category', 'Price_category'] # Kolumny do zakodowania
        for col in categorical_cols:
            if col in self.df.columns:
                # Tworzenie zmiennych dummy (One-Hot Encoding)
                dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                # Łączenie nowych kolumn z oryginalną ramką danych
                self.df = pd.concat([self.df, dummies], axis=1)
                logger.info(f"  OK {col} zakodowana ({dummies.shape[1]} kolumn)")
        return self.df
    
    # Metoda do normalizacji cech numerycznych
    def normalize_numeric_features(self):
        logger.info("Normalizowanie cech numerycznych...")
        # Lista cech numerycznych do normalizacji
        numeric_features = ['Price', 'Metacritic score', 'User score', 'Positive', 'Negative', 'Review_ratio', 'Achievements', 'Days_since_release']
        # Filtrowanie listy, aby zawierała tylko istniejące kolumny
        numeric_features = [col for col in numeric_features if col in self.df.columns]
        # Skalowanie danych za pomocą StandardScaler
        normalized_data = self.scaler.fit_transform(self.df[numeric_features].fillna(0))
        # Dodanie znormalizowanych kolumn do DataFrame
        for i, col in enumerate(numeric_features):
            self.df[f'{col}_normalized'] = normalized_data[:, i]
        logger.info(f"  Znormalizowano {len(numeric_features)} cech")
        return self.df
    
    # Metoda do tworzenia cech interakcji
    def create_interaction_features(self):
        logger.info("Tworzenie cech interakcji...")
        # Utworzenie cechy 'Price_Rating_ratio'
        self.df['Price_Rating_ratio'] = self.df['Price'] / (self.df['Metacritic score'] + 1)
        # Utworzenie cechy 'Rating_Review_score'
        self.df['Rating_Review_score'] = (self.df['Metacritic score'] / 100 * self.df['Review_ratio'])
        # Utworzenie cechy 'Owners_Review_ratio'
        self.df['Owners_Review_ratio'] = np.log1p(self.df['Estimated owners']) * self.df['Review_ratio']
        logger.info("OK Cechy interakcji utworzone")
        return self.df
    
    # Metoda do zapisywania danych po inżynierii cech
    def save_engineered_data(self):
        # Ustalenie ścieżki do katalogu 'data'
        output_dir = Path(__file__).parent / "data"
        # Definiowanie nazwy pliku wyjściowego
        output_file = output_dir / "games_engineered.csv"
        # Zapisanie DataFrame do pliku CSV
        self.df.to_csv(output_file, index=False)
        logger.info(f"\nOK Dane z cechami zapisane: {output_file.name}")
        logger.info(f"  Rozmiar: {self.df.shape}")
        logger.info(f"  Kolumn: {self.df.shape[1]}")
        return self.df
    
    # Metoda uruchamiająca cały proces inżynierii cech
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("INZYNIERIA CECH")
        logger.info("=" * 80 + "\n")
        self.load_data() # Załadowanie danych
        self.create_date_features() # Tworzenie cech z daty
        self.create_platform_features() # Tworzenie cech platform
        self.create_review_features() # Tworzenie cech recenzji
        self.create_score_features() # Tworzenie cech ocen
        self.create_content_features() # Tworzenie cech zawartości
        self.create_price_features() # Tworzenie cech ceny
        self.encode_categorical_features() # Kodowanie zmiennych kategorycznych
        self.normalize_numeric_features() # Normalizacja cech numerycznych
        self.create_interaction_features() # Tworzenie cech interakcji
        self.save_engineered_data() # Zapisanie danych po inżynierii cech
        logger.info("\n" + "=" * 80)
        logger.info("OK INZYNIERIA CECH UKONCZONA")
        logger.info("=" * 80 + "\n")
        return self.df

# Główna funkcja programu
def main():
    engineer = FeatureEngineer() # Utworzenie instancji klasy FeatureEngineer
    engineer.run() # Uruchomienie procesu inżynierii cech

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
