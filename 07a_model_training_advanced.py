# ============================================================================
# TRENOWANIE MODELI ML - WERSJA ZAAWANSOWANA
# Zawiera: Logistic Regression, Random Forest, Decision Tree, Neural Network, 
# LightGBM, XGBoost
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
import io
import time
import warnings

# Importowanie modeli z scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Metryki
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# Keras/TensorFlow
try:
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

# Importowanie zaawansowanych algorytmów boosting
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

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie kodowania UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """Trenowanie zaawansowanych modeli ML z pełnym zestawem algorytmów"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.processed_dir = self.project_dir / "data" / "processed"
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        # Tworzenie katalogów
        self.models_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.target_col = 'Is_highly_rated'
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.training_times = {}
    
    def load_data(self):
        """Załaduj dane treningowe, walidacyjne i testowe"""
        logger.info("=" * 80)
        logger.info("ZAAWANSOWANE TRENOWANIE MODELI ML")
        logger.info("=" * 80)
        logger.info("\nŁadowanie danych...")
        
        train_path = self.processed_dir / "games_train.csv"
        val_path = self.processed_dir / "games_val.csv"
        test_path = self.processed_dir / "games_test.csv"
        
        if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
            raise FileNotFoundError(f"Brakuje plików train/val/test. Uruchom 06_data_export.py")
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        logger.info(f"✓ Train: {len(self.train_df)} wierszy")
        logger.info(f"✓ Val:   {len(self.val_df)} wierszy")
        logger.info(f"✓ Test:  {len(self.test_df)} wierszy")
    
    def prepare_data(self):
        """Przygotuj cechy i zmienną docelową"""
        logger.info("\nPrzygotowanie danych...")
        
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        
        drop_train = [c for c in cols_to_drop if c in self.train_df.columns]
        drop_val = [c for c in cols_to_drop if c in self.val_df.columns]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_train = self.train_df.drop(columns=drop_train).fillna(0)
        self.y_train = self.train_df[self.target_col]
        
        self.X_val = self.val_df.drop(columns=drop_val).fillna(0)
        self.y_val = self.val_df[self.target_col]
        
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        self.y_test = self.test_df[self.target_col]
        
        logger.info(f"✓ Liczba cech: {self.X_train.shape[1]}")
        logger.info(f"✓ Cechy: {list(self.X_train.columns)[:5]}... (wyświetlono 5 z {len(self.X_train.columns)})")
        
        # Skalowanie danych
        logger.info("\nSkalowanie danych (StandardScaler)...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        logger.info("✓ Dane skalowane")
        
        # Oblicz sample weights dla balansowania klas
        self.sample_weights = compute_sample_weight('balanced', self.y_train)
        logger.info(f"✓ Wagi próbek obliczone dla balansowania klas")
        
        # Sprawdzenie rozkładu klas
        class_dist = self.y_train.value_counts()
        class_dist_pct = self.y_train.value_counts(normalize=True) * 100
        logger.info(f"\nRozkład klas w zbiorze treningowym:")
        logger.info(f"  Klasa 0: {class_dist[0]} ({class_dist_pct[0]:.1f}%)")
        logger.info(f"  Klasa 1: {class_dist[1]} ({class_dist_pct[1]:.1f}%)")
    
    def define_models(self):
        """Zdefiniuj wszystkie modele"""
        logger.info("\n" + "-" * 60)
        logger.info("DEFINICJA MODELI")
        logger.info("-" * 60)
        
        # 1. Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        logger.info("✓ Logistic Regression")
        
        # 2. Decision Tree Classifier
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        logger.info("✓ Decision Tree Classifier")
        
        # 3. Random Forest Classifier
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        logger.info("✓ Random Forest Classifier")
        
        # 4. LightGBM
        if HAS_LGBM:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            logger.info("✓ LightGBM")
        else:
            logger.warning("⚠ LightGBM nie zainstalowany")
        
        # 5. XGBoost
        if HAS_XGB:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            logger.info("✓ XGBoost")
        else:
            logger.warning("⚠ XGBoost nie zainstalowany")
    
    def train_and_evaluate_models(self):
        """Trenuj i ewaluuj modele"""
        logger.info("\n" + "-" * 60)
        logger.info("TRENOWANIE I EWALUACJA")
        logger.info("-" * 60)
        
        for name, model in self.models.items():
            logger.info(f"\n[{name}] Trenowanie...")
            
            start_time = time.time()
            
            # Trenowanie
            if name in ['LightGBM', 'XGBoost']:
                model.fit(self.X_train, self.y_train,
                         sample_weight=self.sample_weights,
                         eval_set=[(self.X_val, self.y_val)],
                         verbose=0)
            else:
                model.fit(self.X_train, self.y_train,
                         sample_weight=self.sample_weights)
            
            train_time = time.time() - start_time
            self.training_times[name] = train_time
            
            # Predykcje
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
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
            
            logger.info(f"  ✓ Czas trenowania: {train_time:.2f}s")
            logger.info(f"  ✓ Accuracy:  {acc:.4f}")
            logger.info(f"  ✓ Precision: {prec:.4f}")
            logger.info(f"  ✓ Recall:    {rec:.4f}")
            logger.info(f"  ✓ F1-Score:  {f1:.4f}")
            logger.info(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
    
    def train_neural_network(self):
        """Trenuj prostą sieci neuronową"""
        if not HAS_KERAS:
            logger.warning("⚠ TensorFlow/Keras nie zainstalowany - pomijanie Neural Network")
            return
        
        logger.info(f"\n[Neural Network] Trenowanie...")
        
        start_time = time.time()
        
        # Budowa modelu
        model = Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Trenowanie
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            self.X_train_scaled, self.y_train,
            sample_weight=self.sample_weights,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        train_time = time.time() - start_time
        self.training_times['Neural Network'] = train_time
        
        # Predykcje
        y_proba = model.predict(self.X_test_scaled, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        
        # Metryki
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        self.results['Neural Network'] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'training_time_seconds': float(train_time)
        }
        
        logger.info(f"  ✓ Czas trenowania: {train_time:.2f}s")
        logger.info(f"  ✓ Accuracy:  {acc:.4f}")
        logger.info(f"  ✓ Precision: {prec:.4f}")
        logger.info(f"  ✓ Recall:    {rec:.4f}")
        logger.info(f"  ✓ F1-Score:  {f1:.4f}")
        logger.info(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
        
        # Zapisz model
        model_path = self.models_dir / "neural_network.h5"
        model.save(model_path)
        logger.info(f"  ✓ Model zapisany: {model_path.name}")
    
    def visualize_results(self):
        """Wizualizuj wyniki"""
        logger.info("\n" + "-" * 60)
        logger.info("WIZUALIZACJA WYNIKÓW")
        logger.info("-" * 60)
        
        # Konwertuj wyniki na DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        # 1. Porównanie metryk
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            results_df[metric].plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('Score')
            for i, v in enumerate(results_df[metric]):
                ax.text(v, i, f' {v:.4f}', va='center')
        
        # Czas trenowania
        ax = axes[1, 2]
        training_times_df = pd.Series(self.training_times)
        training_times_df = training_times_df[results_df.index]
        training_times_df.plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_title('CZAS TRENOWANIA (sekundy)')
        ax.set_xlabel('Czas [s]')
        for i, v in enumerate(training_times_df):
            ax.text(v, i, f' {v:.2f}s', va='center')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "models_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: models_comparison.png")
        
        # 2. Szczegółowe porównanie ROC-AUC
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(results_df))
        bars = ax.bar(x_pos, results_df['roc_auc'], color=['green' if x == max(results_df['roc_auc']) else 'skyblue' for x in results_df['roc_auc']])
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title('Porównanie Modeli - Metrika ROC-AUC')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        for i, (bar, val) in enumerate(zip(bars, results_df['roc_auc'])):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "roc_auc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: roc_auc_comparison.png")
    
    def save_best_model(self):
        """Zapisz najlepszy model"""
        logger.info("\n" + "-" * 60)
        logger.info("WYBÓR I ZAPIS NAJLEPSZEGO MODELU")
        logger.info("-" * 60)
        
        best_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        best_score = self.results[best_name]['roc_auc']
        
        logger.info(f"\n✓ NAJLEPSZY MODEL: {best_name}")
        logger.info(f"  ROC-AUC: {best_score:.4f}")
        logger.info(f"  Czas trenowania: {self.training_times[best_name]:.2f}s")
        
        if best_name == 'Neural Network':
            logger.info("  (Model Neural Network został już zapisany)")
        else:
            best_model = self.models[best_name]
            model_path = self.models_dir / "best_model.joblib"
            joblib.dump(best_model, model_path)
            logger.info(f"  ✓ Model zapisany: {model_path.name}")
    
    def save_results_report(self):
        """Zapisz raport wyników"""
        logger.info("\nZapisywanie raportu...")
        
        # Konwertuj do formatu serializowalnego
        results_for_json = {}
        for model_name, metrics in self.results.items():
            results_for_json[model_name] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                           for k, v in metrics.items()}
        
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_info': {
                'train_size': len(self.train_df),
                'val_size': len(self.val_df),
                'test_size': len(self.test_df),
                'n_features': self.X_train.shape[1]
            },
            'class_distribution': {
                'class_0': int(self.y_train.value_counts()[0]),
                'class_1': int(self.y_train.value_counts()[1])
            },
            'models_performance': results_for_json
        }
        
        report_path = self.reports_dir / "07_model_training_advanced.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Raport zapisany: {report_path.name}")
        
        # Wyświetl podsumowanie
        logger.info("\n" + "=" * 60)
        logger.info("PODSUMOWANIE WYNIKÓW")
        logger.info("=" * 60)
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        print("\n" + results_df.to_string())
        print("\n")
    
    def run(self):
        """Uruchom całe pipeline'u"""
        self.load_data()
        self.prepare_data()
        self.define_models()
        self.train_and_evaluate_models()
        self.train_neural_network()
        self.visualize_results()
        self.save_best_model()
        self.save_results_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ TRENOWANIE MODELI ZAKOŃCZONE")
        logger.info("=" * 80 + "\n")


def main():
    trainer = AdvancedModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
