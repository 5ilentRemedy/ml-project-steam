"""
07_model_evaluation.py
Zaawansowana ewaluacja wieloklasowa modelu ML z rysowaniem krzywych ROC/PR typu One-vs-Rest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import io
import joblib
import json

# Narzędzia do wizualizacji i metryk wieloklasowych
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Kodowanie UTF-8 dla konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.processed_dir = self.project_dir / "data" / "processed"
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_df = None
        self.model = None
        self.target_col = 'Market_Success'  # Zsynchronizowany target
        
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.classes = ['Zrozumiały Hit', 'Ukryty Diament', 'Przeciętniak', 'Klapa rynkowa']

    def load_data_and_model(self):
        """Ładuje zbiór testowy oraz najlepszy zapisany model"""
        logger.info("Ładowanie danych testowych i modelu...")
        
        test_path = self.processed_dir / "games_test.csv"
        model_path = self.models_dir / "best_model.joblib"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Brak pliku danych testowych: {test_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}. Uruchom najpierw 05_model_training.py.")
            
        self.test_df = pd.read_csv(test_path)
        self.model = joblib.load(model_path)
        
        logger.info(f"[OK] Zbiór testowy: {self.test_df.shape[0]} wierszy")
        logger.info(f"[OK] Model załadowany: {type(self.model).__name__}")
        
        # 🔥 POPRAWKA WYCIEKU: Identyczne czyszczenie cech jak w skrypcie 05
        cols_to_drop = [
            'AppID', 'Name', 'Genres', 'Market_Success',
            'Positive', 'Negative', 'Total_reviews', 'Total_Reviews', 'Review_ratio', 'Log_total_reviews'
        ]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        self.y_test = self.test_df[self.target_col]

    def generate_predictions(self):
        """Generuje predykcje i prawdopodobieństwa wieloklasowe"""
        logger.info("Generowanie predykcji wieloklasowych na zbiorze testowym...")
        
        # Obsługa specyfiki XGBoost, jeśli został zapisany LabelEncoder
        encoder_path = self.models_dir / "label_encoder.joblib"
        if encoder_path.exists() and type(self.model).__name__ == 'XGBClassifier':
            le = joblib.load(encoder_path)
            y_pred_raw = self.model.predict(self.X_test)
            self.y_pred = le.inverse_transform(y_pred_raw)
        else:
            self.y_pred = self.model.predict(self.X_test)
            
        self.y_proba = self.model.predict_proba(self.X_test)
        
        acc = accuracy_score(self.y_test, self.y_pred)
        logger.info(f"[OK] Ogólna dokładność (Accuracy) modelu na teście: {acc:.4f}")

    def evaluate_metrics(self):
        """Generuje pełny raport klasyfikacji wieloklasowej i zapisuje do JSON"""
        logger.info("\n" + "-" * 60)
        logger.info("WIELOKLASOWY RAPORT KLASYFIKACJI (METRYKI KOŃCOWE)")
        logger.info("-" * 60)
        
        # Użyj tylko klas, które są rzeczywiście obecne w zbiorze testowym
        model_classes = list(getattr(self.model, 'classes_', self.classes))
        present_classes = [c for c in model_classes if c in np.unique(self.y_test)]

        report = classification_report(self.y_test, self.y_pred, labels=present_classes, target_names=present_classes)
        print(report)

        # Zapis zgodny z oczekiwaniami orchestratora (tylko obecne klasy)
        report_dict = classification_report(self.y_test, self.y_pred, labels=present_classes, output_dict=True)
        with open(self.reports_dir / "05_evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def plot_roc_curve(self):
        """⭐ WARTOŚCIOWA WIZUALIZACJA: Wieloklasowe krzywe ROC i PR (One-vs-Rest)"""
        logger.info("📊 Generowanie wieloklasowych krzywych ROC i Precision-Recall...")
        
        try:
            # Binaryzacja etykiet tekstowych na potrzeby wyliczeń krzywych OvR
            model_classes = list(getattr(self.model, 'classes_', self.classes))
            y_test_bin = label_binarize(self.y_test, classes=model_classes)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']

            # Pętla rysująca linie osobno dla każdej klasy zwróconej przez model
            for i, class_name in enumerate(model_classes):
                if i >= self.y_proba.shape[1] or i >= y_test_bin.shape[1]:
                    continue
                # Ścieżka ROC
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

                # Ścieżka Precision-Recall
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], self.y_proba[:, i])
                pr_auc = auc(recall, precision)
                ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
            
            # Formatowanie wykresu ROC
            ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate (1 - Specyficzność)')
            ax1.set_ylabel('True Positive Rate (Czułość)')
            ax1.set_title('Wieloklasowa krzywa ROC (One-vs-Rest)', fontsize=11, fontweight='bold')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.2)
            
            # Formatowanie wykresu Precision-Recall
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall (Pełność)')
            ax2.set_ylabel('Precision (Precyzja)')
            ax2.set_title('Wieloklasowa krzywa Precision-Recall', fontsize=11, fontweight='bold')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.2)
            
            out_path = self.figures_dir / "roc_pr_curves.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print("✓ Zapisano zsynchronizowany wykres: roc_pr_curves.png")
        except Exception as e:
            logger.error(f"✗ Błąd podczas rysowania krzywych wieloklasowych OvR: {e}")

    def plot_feature_importance(self):
        """⭐ KLUCZOWA WIZUALIZACJA: Ważność cech modelu klasyfikacji"""
        logger.info("📊 Generowanie wykresu ważności cech wejściowych...")
        try:
            importances = None
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])
                
            if importances is not None:
                feat_imp = pd.DataFrame({
                    'Feature': self.X_test.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                plt.figure(figsize=(10, 6))
                plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal', alpha=0.8)
                plt.title(f'Istotność cech w predykcji sukcesu rynkowego ({type(self.model).__name__})', fontsize=12, fontweight='bold')
                plt.xlabel('Współczynnik ważności algorytmu')
                plt.grid(axis='x', alpha=0.3)
                
                out_path = self.figures_dir / "feature_importance.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print("✓ Zapisano wykres istotności: feature_importance.png")
        except Exception as e:
            logger.error(f"✗ Błąd przy tworzeniu wykresu ważności cech: {e}")

    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("PEŁNA EWALUACJA ZBALANSOWANEGO MODELU ML")
        logger.info("=" * 80 + "\n")
        
        try:
            self.load_data_and_model()
            self.generate_predictions()
            self.evaluate_metrics()
            self.plot_roc_curve()
            self.plot_feature_importance()
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PROCES EWALUACJI ZAKOŃCZONY POMYŚLNIE")
            logger.info(f"  - Wszystkie raporty i krzywe wieloklasowe zostały wygenerowane.")
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"✗ Błąd krytyczny podczas ewaluacji: {e}")
            raise

def main():
    evaluator = ModelEvaluator()
    evaluator.run()

if __name__ == "__main__":
    main()