# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import logging # Do logowania informacji
import json # Do pracy z formatem JSON
import sys # Do interakcji z systemem operacyjnym
import io # Do obsługi strumieni wejścia/wyjścia
import time # Do mierzenia czasu wykonania
import joblib # Do serializacji i deserializacji obiektów Pythona (np. modeli)

# Importowanie modeli i metryk z scikit-learn
from sklearn.linear_model import LogisticRegression # Model regresji logistycznej
from sklearn.ensemble import RandomForestClassifier # Model lasu losowego
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # Metryki ewaluacji

# Próba importu zaawansowanych algorytmów gradient boosting (LightGBM, XGBoost)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do trenowania modeli
class ModelTrainer:
    # Inicjalizacja obiektu ModelTrainer
    def __init__(self):
        self.project_dir = Path(__file__).parent # Ustalenie katalogu projektu
        self.processed_dir = self.project_dir / "data" / "processed" # Ścieżka do przetworzonych danych
        self.models_dir = self.project_dir / "models" # Ścieżka do katalogu modeli
        self.reports_dir = self.project_dir / "reports" # Ścieżka do katalogu raportów
        
        # Utworzenie katalogu na modele, jeśli nie istnieje
        self.models_dir.mkdir(exist_ok=True)
        
        self.train_df = None # Ramka danych dla zbioru treningowego
        self.test_df = None # Ramka danych dla zbioru testowego
        self.target_col = 'Is_highly_rated'  # Kolumna docelowa (do przewidzenia)
        
        self.models = {} # Słownik do przechowywania zdefiniowanych modeli
        self.results = {} # Słownik do przechowywania wyników ewaluacji modeli
        self.best_model_name = None # Nazwa najlepszego modelu

    # Metoda do ładowania danych treningowych i testowych
    def load_data(self):
        logger.info("Ladowanie danych treningowych i testowych...")
        # Ścieżki do plików CSV zbiorów treningowego i testowego
        train_path = self.processed_dir / "games_train.csv"
        test_path = self.processed_dir / "games_test.csv"
        
        # Sprawdzenie, czy pliki istnieją
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Nie znaleziono plikow train/test w {self.processed_dir}. Uruchom skrypt 06_data_export.py.")
            
        # Wczytanie danych do DataFrame
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        logger.info(f"OK Zbior treningowy: {self.train_df.shape[0]} wierszy")
        logger.info(f"OK Zbior testowy: {self.test_df.shape[0]} wierszy")

    # Metoda do przygotowania danych (cechy X i cel y) do trenowania
    def prepare_data(self):
        logger.info(f"Przygotowanie danych. Cel (Target): '{self.target_col}'")
        
        # Kolumny do odrzucenia z zestawu cech (identyfikatory, tekst, kolumna docelowa)
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        
        # Filtrowanie kolumn do odrzucenia, aby usunąć tylko te, które faktycznie istnieją
        drop_train = [c for c in cols_to_drop if c in self.train_df.columns]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        # Podział danych na cechy (X) i zmienną docelową (y) dla zbioru treningowego
        self.X_train = self.train_df.drop(columns=drop_train)
        self.y_train = self.train_df[self.target_col]
        
        # Podział danych na cechy (X) i zmienną docelową (y) dla zbioru testowego
        self.X_test = self.test_df.drop(columns=drop_test)
        self.y_test = self.test_df[self.target_col]
        
        # Wypełnienie brakujących wartości w cechach zerami (prosta imputacja)
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        logger.info(f"OK Cechy gotowe. Liczba cech: {self.X_train.shape[1]}")
        logger.info(f"Lista cech: {list(self.X_train.columns)}")

    # Metoda do definiowania modeli uczenia maszynowego do przetestowania
    def define_models(self):
        logger.info("Inicjalizacja modeli ML...")
        
        # 1. Regresja Logistyczna (model bazowy)
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, # Maksymalna liczba iteracji dla algorytmu optymalizacyjnego
            random_state=42, # Ziarno losowości dla powtarzalności wyników
            class_weight='balanced' # Automatyczne dostosowanie wag klas, aby radzić sobie z niezbalansowanymi klasami
        )
        
        # 2. Random Forest (Las Losowy)
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, # Liczba drzew w lesie
            max_depth=15, # Maksymalna głębokość drzewa
            random_state=42, # Ziarno losowości
            n_jobs=-1, # Użycie wszystkich dostępnych rdzeni procesora
            class_weight='balanced' # Dostosowanie wag klas
        )
        
        # 3. LightGBM (jeśli biblioteka jest zainstalowana)
        if HAS_LGBM:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, # Liczba estymatorów (drzew)
                learning_rate=0.1, # Współczynnik uczenia
                random_state=42, # Ziarno losowości
                n_jobs=-1, # Użycie wszystkich dostępnych rdzeni procesora
                class_weight='balanced' # Dostosowanie wag klas
            )
            logger.info("  Dodano model LightGBM")
        # 4. XGBoost (jeśli biblioteka jest zainstalowana )
        elif HAS_XGB:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, # Liczba estymatorów (drzew)
                learning_rate=0.1, # Współczynnik uczenia
                random_state=42, # Ziarno losowości
                n_jobs=-1, # Użycie wszystkich dostępnych rdzeni procesora
                eval_metric='logloss' # Metryka ewaluacji dla wczesnego zatrzymania
            )
            logger.info("  Dodano model XGBoost")
        else:
            logger.warning("  LightGBM ani XGBoost nie sa zainstalowane. Zostana uzyte tylko modele Sklearn.")
            logger.warning("      Aby uzyskac lepsze wyniki, zainstaluj: pip install lightgbm xgboost")

    # Metoda do trenowania i ewaluacji wszystkich zdefiniowanych modeli
    def train_and_evaluate(self):
        logger.info("\n" + "-" * 60)
        logger.info("TRENOWANIE I EWALUACJA MODELI")
        logger.info("-" * 60)
        
        # Iteracja przez każdy model w słowniku models
        for name, model in self.models.items():
            logger.info(f"Trenowanie modelu: {name}...")
            start_time = time.time() # Rozpoczęcie pomiaru czasu trenowania
            
            # Trenowanie modelu na danych treningowych
            model.fit(self.X_train, self.y_train)
            
            # Generowanie predykcji na zbiorze testowym
            y_pred = model.predict(self.X_test)
            # Generowanie prawdopodobieństw predykcji (jeśli model to wspiera)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            train_time = time.time() - start_time # Obliczenie czasu trenowania
            
            # Obliczanie metryk ewaluacji
            acc = accuracy_score(self.y_test, y_pred) # Dokładność
            prec = precision_score(self.y_test, y_pred, zero_division=0) # Precyzja
            rec = recall_score(self.y_test, y_pred, zero_division=0) # Czułość (Recall)
            f1 = f1_score(self.y_test, y_pred, zero_division=0) # F1-Score
            roc_auc = roc_auc_score(self.y_test, y_proba) # ROC-AUC
            
            # Zapisywanie wyników do słownika results
            self.results[name] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'training_time_seconds': float(train_time)
            }
            
            logger.info(f"  OK {name} | Czas: {train_time:.1f}s | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

    # Metoda do zapisywania raportów i najlepszego modelu
    def save_reports_and_models(self):
        # Znalezienie najlepszego modelu na podstawie metryki ROC-AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        logger.info("\n" + "=" * 60)
        logger.info(f"NAJLEPSZY MODEL: {self.best_model_name} (ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f})")
        logger.info("=" * 60)
        
        # Zapisanie najlepszego modelu na dysk za pomocą joblib
        best_model = self.models[self.best_model_name]
        model_file_path = self.models_dir / "best_model.joblib"
        joblib.dump(best_model, model_file_path)
        logger.info(f"OK Zapisano najlepszy model do: {model_file_path.relative_to(self.project_dir)}")
        
        # Eksport ważności cech (feature importance), jeśli model to obsługuje
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = {
                feature: float(imp) for feature, imp in zip(self.X_train.columns, importances)
            }
            # Sortowanie cech według ważności (od najważniejszej)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            logger.info("\nKluczowe cechy (Top 5):")
            for i, (feat, imp) in enumerate(list(feature_importance.items())[:5]):
                logger.info(f"  {i+1}. {feat}: {imp:.4f}")
        
        # Przygotowanie raportu z treningu
        report = {
            "target_column": self.target_col,
            "feature_count": len(self.X_train.columns),
            "features_used": list(self.X_train.columns),
            "best_model": self.best_model_name,
            "model_results": self.results,
            "feature_importance": feature_importance
        }
        
        # Zapisanie raportu do pliku JSON
        report_file = self.reports_dir / "04_model_training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\nOK Raport z treningu zapisany: {report_file.relative_to(self.project_dir)}")

    # Metoda uruchamiająca cały proces trenowania modeli
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("MACHINE LEARNING - TRENOWANIE MODELI")
        logger.info("=" * 80 + "\n")
        
        self.load_data() # Załadowanie danych
        self.prepare_data() # Przygotowanie danych (X i y)
        self.define_models() # Definiowanie modeli do przetestowania
        self.train_and_evaluate() # Trenowanie i ewaluacja modeli
        self.save_reports_and_models() # Zapisanie raportów i najlepszego modelu
        
        logger.info("\n" + "=" * 80)
        logger.info("OK TRENOWANIE ZAKONCZONE")
        logger.info("=" * 80 + "\n")

# Główna funkcja programu
def main():
    trainer = ModelTrainer() # Utworzenie instancji klasy ModelTrainer
    trainer.run() # Uruchomienie procesu trenowania modeli

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()