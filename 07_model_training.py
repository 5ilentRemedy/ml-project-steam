"""
07_model_training.py - Trenowanie i ewaluacja modeli Machine Learning
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
import io
import time
import joblib

# Modele i metryki
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Próba importu zaawansowanych algorytmów (Gradient Boosting)
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

# Kodowanie UTF-8 dla konsoli (zgodność z Twoim pipeline'em)
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.processed_dir = self.project_dir / "data" / "processed"
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports"
        
        # Utworzenie katalogu na modele, jeśli nie istnieje
        self.models_dir.mkdir(exist_ok=True)
        
        self.train_df = None
        self.test_df = None
        self.target_col = 'Is_highly_rated'  # Nasz cel: przewidzenie wysokiej oceny
        
        self.models = {}
        self.results = {}
        self.best_model_name = None

    def load_data(self):
        """Ładuje podzielone dane z poprzedniego kroku (exportu)"""
        logger.info("Ładowanie danych treningowych i testowych...")
        train_path = self.processed_dir / "games_train.csv"
        test_path = self.processed_dir / "games_test.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Nie znaleziono plików train/test w {self.processed_dir}. Uruchom skrypt 06_data_export.py.")
            
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        logger.info(f"[OK] Zbiór treningowy: {self.train_df.shape[0]} wierszy")
        logger.info(f"[OK] Zbiór testowy: {self.test_df.shape[0]} wierszy")

    def prepare_data(self):
        """Przygotowuje X (cechy) i y (cel) do trenowania"""
        logger.info(f"Przygotowanie danych. Cel (Target): '{self.target_col}'")
        
        # Kolumny do odrzucenia z cech (identyfikatory i tekst których model bezpośrednio nie zrozumie)
        # Genres odrzucamy, bo to surowy tekst (np. "Action, RPG"). Ewentualnie można by je zakodować (One-Hot).
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        
        # Upewnienie się, że usuwamy tylko istniejące kolumny
        drop_train = [c for c in cols_to_drop if c in self.train_df.columns]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_train = self.train_df.drop(columns=drop_train)
        self.y_train = self.train_df[self.target_col]
        
        self.X_test = self.test_df.drop(columns=drop_test)
        self.y_test = self.test_df[self.target_col]
        
        # Zamiana braków danych (jeśli jakieś zostały) na 0 lub medianę
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        logger.info(f"[OK] Cechy gotowe. Liczba cech: {self.X_train.shape[1]}")
        logger.info(f"Lista cech: {list(self.X_train.columns)}")

    def define_models(self):
        """Definiuje modele do przetestowania"""
        logger.info("Inicjalizacja modeli ML...")
        
        # 1. Baseline: Regresja Logistyczna (szybka, prosta, interpretowalna)
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight='balanced'
        )
        
        # 2. Random Forest (Las Losowy) - bardzo solidny algorytm na dane tabelaryczne
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 3. LightGBM (jeśli zainstalowany) - zazwyczaj daje najlepsze wyniki
        if HAS_LGBM:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            logger.info("  ✓ Dodano model LightGBM")
        elif HAS_XGB:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            logger.info("  ✓ Dodano model XGBoost")
        else:
            logger.warning("  [!] LightGBM ani XGBoost nie są zainstalowane. Zostaną użyte tylko modele Sklearn.")
            logger.warning("      Aby uzyskać lepsze wyniki, zainstaluj: pip install lightgbm xgboost")

    def train_and_evaluate(self):
        """Trenuje wszystkie zdefiniowane modele i mierzy ich skuteczność"""
        logger.info("\n" + "-" * 60)
        logger.info("TRENOWANIE I EWALUACJA MODELI")
        logger.info("-" * 60)
        
        for name, model in self.models.items():
            logger.info(f"Trenowanie modelu: {name}...")
            start_time = time.time()
            
            # Trenowanie
            model.fit(self.X_train, self.y_train)
            
            # Predykcje
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            train_time = time.time() - start_time
            
            # Metryki
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            self.results[name] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'training_time_seconds': float(train_time)
            }
            
            logger.info(f"  [OK] {name} | Czas: {train_time:.1f}s | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

    def save_reports_and_models(self):
        """Zapisuje wyniki do pliku JSON i eksportuje najlepszy model"""
        # Znalezienie najlepszego modelu na podstawie ROC-AUC (świetna metryka dla niezbalansowanych klas)
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        logger.info("\n" + "=" * 60)
        logger.info(f"🏆 NAJLEPSZY MODEL: {self.best_model_name} (ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f})")
        logger.info("=" * 60)
        
        # Zapisz model na dysk używając joblib
        best_model = self.models[self.best_model_name]
        model_file_path = self.models_dir / "best_model.joblib"
        joblib.dump(best_model, model_file_path)
        logger.info(f"[OK] Zapisano najlepszy model do: {model_file_path.relative_to(self.project_dir)}")
        
        # Eksport feature importance (ważności cech) jeśli model to obsługuje (Random Forest / LightGBM)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = {
                feature: float(imp) for feature, imp in zip(self.X_train.columns, importances)
            }
            # Sortuj od najważniejszej
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            logger.info("\nKluczowe cechy (Top 5):")
            for i, (feat, imp) in enumerate(list(feature_importance.items())[:5]):
                logger.info(f"  {i+1}. {feat}: {imp:.4f}")
        
        # Przygotowanie raportu
        report = {
            "target_column": self.target_col,
            "feature_count": len(self.X_train.columns),
            "features_used": list(self.X_train.columns),
            "best_model": self.best_model_name,
            "model_results": self.results,
            "feature_importance": feature_importance
        }
        
        report_file = self.reports_dir / "04_model_training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\n[OK] Raport z treningu zapisany: {report_file.relative_to(self.project_dir)}")

    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("MACHINE LEARNING - TRENOWANIE MODELI")
        logger.info("=" * 80 + "\n")
        
        self.load_data()
        self.prepare_data()
        self.define_models()
        self.train_and_evaluate()
        self.save_reports_and_models()
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] TRENOWANIE ZAKOŃCZONE")
        logger.info("=" * 80 + "\n")


def main():
    trainer = ModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()