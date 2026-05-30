"""
07_model_training.py
Trenowanie i ewaluacja wieloklasowych modeli uczenia maszynowego z zabezpieczeniem pustych klas.
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

# Modele i metryki wieloklasowe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Kodowanie UTF-8 dla konsoli Windows
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
        
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.train_df = None
        self.test_df = None
        self.target_col = 'Market_Success'
        
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.label_encoder = None

    def load_data(self):
        """Ładuje podzielone dane wygenerowane przez DataExporter"""
        logger.info("Ładowanie zsynchronizowanych zbiorów train/test...")
        train_path = self.processed_dir / "games_train.csv"
        test_path = self.processed_dir / "games_test.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Brak zbiorów w {self.processed_dir}. Uruchom najpierw 06_data_export.py.")
            
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        logger.info(f"[OK] Wczytano zbiór treningowy: {self.train_df.shape[0]} wierszy")
        logger.info(f"[OK] Wczytano zbiór testowy: {self.test_df.shape[0]} wierszy")

    def prepare_data(self):
        """Przygotowuje X i y zapobiegając wyciekowi danych (Data Leakage)"""
        logger.info(f"Przygotowanie macierzy cech dla targetu: '{self.target_col}'")
        
        # Eliminacja cech bezpośrednio powiązanych z regułą budowy targetu
        cols_to_drop = [
            'AppID', 'Name', 'Genres', 'Market_Success',
            'Positive', 'Negative', 'Total_reviews', 'Total_Reviews', 'Review_ratio', 'Log_total_reviews'
        ]
        
        drop_train = [c for c in cols_to_drop if c in self.train_df.columns]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_train = self.train_df.drop(columns=drop_train).fillna(0)
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        
        self.y_train = self.train_df[self.target_col]
        self.y_test = self.test_df[self.target_col]
        
        logger.info(f"[OK] Bezpieczne cechy uczące przygotowane. Liczba cech: {self.X_train.shape[1]}")

    def define_models(self):
        """Definiuje klasyfikatory wieloklasowe ze zbalansowanymi wagami"""
        logger.info("Inicjalizacja wieloklasowych modeli maszynowych...")
        
        # 1. Regresja Logistyczna (Bez przestarzałego parametru multi_class)
        self.models['Logistic Regression'] = LogisticRegression(
            solver='lbfgs',
            max_iter=2000, 
            random_state=42, 
            class_weight='balanced'
        )
        
        # 2. Las Losowy (Zbalansowany klasyfikator drzewiasty)
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 3. Zaawansowane algorytmy boostingowe
        if HAS_LGBM:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=4,
                n_estimators=100,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            logger.info("  ✓ Pomyślnie zaimportowano model LightGBM")
        elif HAS_XGB:
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            
            self.models['XGBoost'] = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=4,
                n_estimators=100,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            logger.info("  ✓ Pomyślnie zaimportowano model XGBoost")

    def train_and_evaluate(self):
        """Uruchamia proces uczenia i wylicza metryki wieloklasowe Macro"""
        logger.info("\n" + "-" * 60)
        logger.info("ROZPOCZĘCIE ETAPU TRENOWANIA MODELI ML")
        logger.info("-" * 60)
        
        for name, model in self.models.items():
            logger.info(f"Trenowanie i optymalizacja: {name}...")
            start_time = time.time()
            
            if name == 'XGBoost' and HAS_XGB:
                model.fit(self.X_train, self.y_train_encoded)
                y_pred_raw = model.predict(self.X_test)
                y_pred = self.label_encoder.inverse_transform(y_pred_raw)
                y_proba = model.predict_proba(self.X_test)
                model_classes = self.label_encoder.classes_
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                model_classes = model.classes_
                
            elapsed_time = time.time() - start_time
            
            # Obliczanie podstawowych metryk
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
            
            # 🔥 POPRAWKA ZABEZPIECZAJĄCA: Liczymy ROC-AUC OvR tylko dla klas realnie obecnych w y_test
            unique_test_classes = np.unique(self.y_test)
            
            if len(unique_test_classes) > 1:
                # Mapujemy indeksy obecnych klas w tablicy prawdopodobieństw modelu
                class_indices = [list(model_classes).index(cls) for cls in unique_test_classes]
                filtered_proba = y_proba[:, class_indices]
                
                # Normalizujemy prawdopodobieństwa, aby sumowały się do 1 dla filtrowanych podklas
                if filtered_proba.sum(axis=1).min() > 0:
                    filtered_proba = filtered_proba / filtered_proba.sum(axis=1, keepdims=True)
                
                # Binarize y_test względem rzeczywiście obecnych klas, aby dopasować kształt do filtered_proba
                y_test_binarized = label_binarize(self.y_test, classes=unique_test_classes)

                # Jeśli test zawiera dokładnie 2 klasy, roc_auc_score oczekuje 1D y_score (prawdopodobieństwo klasy pozytywnej)
                if filtered_proba.shape[1] == 2:
                    # Przy binarnym przypadku `label_binarize` może zwrócić macierz 1-kolumnową
                    # (kolumna odpowiada klasie `unique_test_classes[1]`). Dopasowujemy mapowanie.
                    if y_test_binarized.shape[1] == 1:
                        y_true_bin = y_test_binarized[:, 0]
                        y_score_pos = filtered_proba[:, 1]
                    else:
                        y_true_bin = y_test_binarized[:, 1]
                        y_score_pos = filtered_proba[:, 1]
                    roc_auc = roc_auc_score(y_true_bin, y_score_pos)
                else:
                    roc_auc = roc_auc_score(y_test_binarized, filtered_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = 0.5  # Wartość domyślna w przypadku braku różnorodności klasowej testu

            self.results[name] = {
                'accuracy': float(acc),
                'precision_macro': float(prec),
                'recall_macro': float(rec),
                'f1_score_macro': float(f1),
                'roc_auc_macro': float(roc_auc),
                'training_time_seconds': float(elapsed_time)
            }
            logger.info(f"  [OK] {name} | Czas: {elapsed_time:.1f}s | Multi-ROC-AUC (Obecne klasy): {roc_auc:.4f} | F1-Macro: {f1:.4f}")

    def save_reports_and_models(self):
        """Eksportuje model joblib i buduje raport końcowy"""
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['roc_auc_macro'])
        logger.info("\n" + "=" * 60)
        logger.info(f"🏆 NAJLEPSZY KLASYFIKATOR: {self.best_model_name} (ROC-AUC: {self.results[self.best_model_name]['roc_auc_macro']:.4f})")
        logger.info("=" * 60)
        
        best_model = self.models[self.best_model_name]
        joblib.dump(best_model, self.models_dir / "best_model.joblib")
        if self.label_encoder:
            joblib.dump(self.label_encoder, self.models_dir / "label_encoder.joblib")
            
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = {feat: float(imp) for feat, imp in zip(self.X_train.columns, importances)}
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            
            logger.info("\nNajważniejsze zmienne decyzyjne klasyfikacji (Top 5):")
            for i, (feat, imp) in enumerate(list(feature_importance.items())[:5]):
                logger.info(f"  {i+1}. {feat}: {imp:.4f}")
        
        report = {
            "target_column": self.target_col,
            "feature_count": len(self.X_train.columns),
            "features_used": list(self.X_train.columns),
            "best_model": self.best_model_name,
            "model_results": self.results,
            "feature_importance": feature_importance
        }
        
        with open(self.reports_dir / "04_model_training_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"\n[OK] Raport z uczenia pomyślnie zapisany w reports/04_model_training_report.json")

    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("START ETAPU TRENOWANIA MODELI")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.prepare_data()
        self.define_models()
        self.train_and_evaluate()
        self.save_reports_and_models()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] PROCES UCZENIA I EKSPORTU MODELU ZAKOŃCZONY")
        logger.info("=" * 80 + "\n")

def main():
    trainer = ModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()