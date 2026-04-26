"""
08_model_evaluation.py - Głęboka ewaluacja zapisanego modelu i generowanie wizualizacji
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import io
import joblib
import json

# Narzędzia do wizualizacji i metryk
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

# Próba importu SHAP do wyjaśnialnego AI (XAI)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Kodowanie UTF-8 dla konsoli
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
        
        # Utworzenie katalogu na wykresy
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_df = None
        self.model = None
        self.target_col = 'Is_highly_rated'
        
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None

    def load_data_and_model(self):
        """Ładuje zbiór testowy oraz najlepszy zapisany model"""
        logger.info("Ładowanie danych testowych i modelu...")
        
        test_path = self.processed_dir / "games_test.csv"
        model_path = self.models_dir / "best_model.joblib"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Brak pliku danych testowych: {test_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}. Uruchom najpierw 07_model_training.py.")
            
        self.test_df = pd.read_csv(test_path)
        self.model = joblib.load(model_path)
        
        logger.info(f"[OK] Zbiór testowy: {self.test_df.shape[0]} wierszy")
        logger.info(f"[OK] Model załadowany: {type(self.model).__name__}")
        
        # Przygotowanie X i y (musi być identyczne jak w skrypcie 07)
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        self.y_test = self.test_df[self.target_col]

    def generate_predictions(self):
        """Generuje predykcje i prawdopodobieństwa"""
        logger.info("Generowanie predykcji na zbiorze testowym...")
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            # Fallback jeśli model nie wspiera predict_proba (np. niezmodyfikowany SVC)
            self.y_proba = self.y_pred
            
        acc = accuracy_score(self.y_test, self.y_pred)
        logger.info(f"[OK] Dokładność (Accuracy) na teście: {acc:.4f}")

    def evaluate_metrics(self):
        """Generuje pełny raport klasyfikacji"""
        logger.info("\n" + "-" * 50)
        logger.info("RAPORT KLASYFIKACJI")
        logger.info("-" * 50)
        
        report = classification_report(self.y_test, self.y_pred, target_names=['Niska/Średnia Ocena', 'Wysoka Ocena (>=75)'])
        print(report)
        
        # Zapis do pliku
        report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
        with open(self.reports_dir / "05_evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def plot_confusion_matrix(self):
        """Rysuje macierz pomyłek"""
        logger.info("Generowanie macierzy pomyłek...")
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Przewidziano: Nie', 'Przewidziano: Tak'],
                    yticklabels=['Faktycznie: Nie', 'Faktycznie: Tak'])
        plt.title('Macierz Pomyłek (Confusion Matrix)')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        
        out_path = self.figures_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"  [OK] Zapisano: {out_path.name}")

    def plot_roc_curve(self):
        """Rysuje krzywą ROC oraz Precision-Recall"""
        logger.info("Generowanie krzywej ROC i PR...")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        out_path = self.figures_dir / "roc_pr_curves.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"  [OK] Zapisano: {out_path.name}")

    def plot_feature_importance(self):
        """Rysuje ważność cech, jeśli model to obsługuje"""
        logger.info("Generowanie wykresu ważności cech...")
        
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            
        if importances is not None:
            feat_imp = pd.DataFrame({
                'Feature': self.X_test.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True).tail(15) # Top 15 cech
            
            plt.figure(figsize=(10, 8))
            plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
            plt.title(f'Ważność Cech (Top 15) - {type(self.model).__name__}')
            plt.xlabel('Waga / Znaczenie')
            
            out_path = self.figures_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"  [OK] Zapisano: {out_path.name}")
        else:
            logger.info("  [-] Ten model nie udostępnia atrybutu ważności cech.")

    def run_shap_analysis(self):
        """Wykonuje analizę SHAP (Explainable AI), jeśli biblioteka jest dostępna"""
        if not HAS_SHAP:
            logger.warning("Biblioteka SHAP nie jest zainstalowana. Pomijanie zaawansowanej interpretacji.")
            logger.warning("Zainstaluj używając: pip install shap")
            return
            
        logger.info("Generowanie analizy SHAP (Explainable AI)... Może to chwilę potrwać.")
        try:
            # Używamy mniejszej próbki do analizy SHAP, aby przyspieszyć działanie
            X_sample = shap.sample(self.X_test, 1000)
            
            # W zależności od modelu dobieramy odpowiedni explainer
            if type(self.model).__name__ in ['RandomForestClassifier', 'LGBMClassifier', 'XGBClassifier']:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                
                # Zabezpieczenie dla modeli zwracających klasyfikację binarną w postaci tablicy 3D
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] 
            else:
                explainer = shap.Explainer(self.model, X_sample)
                shap_values = explainer(X_sample).values
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            
            out_path = self.figures_dir / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  [OK] Zapisano: {out_path.name}")
            
        except Exception as e:
            logger.error(f"  [BŁĄD] Nie udało się wygenerować wykresu SHAP: {str(e)}")

    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("EWALUACJA MODELU ML")
        logger.info("=" * 80 + "\n")
        
        self.load_data_and_model()
        self.generate_predictions()
        self.evaluate_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance()
        self.run_shap_analysis()
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] EWALUACJA ZAKOŃCZONA")
        logger.info(f"Wyniki zapisano w: {self.figures_dir.relative_to(self.project_dir)}")
        logger.info("=" * 80 + "\n")


def main():
    evaluator = ModelEvaluator()
    evaluator.run()

if __name__ == "__main__":
    main()