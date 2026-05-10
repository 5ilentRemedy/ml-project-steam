# ============================================================================
# ZAAWANSOWANA EWALUACJA MODELU - Wizualizacje i Szczegółowe Analizy
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import io
import json
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, accuracy_score
)
from scipy import stats

import joblib

# Ustawienie kodowania UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class AdvancedModelEvaluator:
    """Zaawansowana ewaluacja modelu z wizualizacjami"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.processed_dir = self.project_dir / "data" / "processed"
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_df = None
        self.model = None
        self.target_col = 'Is_highly_rated'
        
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
    
    def load_data_and_model(self):
        """Załaduj dane testowe i model"""
        logger.info("=" * 80)
        logger.info("ZAAWANSOWANA EWALUACJA MODELU")
        logger.info("=" * 80)
        logger.info("\nŁadowanie danych i modelu...")
        
        test_path = self.processed_dir / "games_test.csv"
        model_path = self.models_dir / "best_model.joblib"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Brak pliku testowego: {test_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Brak modelu: {model_path}")
        
        self.test_df = pd.read_csv(test_path)
        self.model = joblib.load(model_path)
        
        logger.info(f"✓ Dane testowe: {self.test_df.shape}")
        logger.info(f"✓ Model załadowany: {type(self.model).__name__}")
        
        # Przygotowanie danych
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        self.y_test = self.test_df[self.target_col]
    
    def generate_predictions(self):
        """Generuj predykcje"""
        logger.info("\nGenerowanie predykcji...")
        
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, 'predict_proba'):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.y_proba = self.y_pred
        
        acc = accuracy_score(self.y_test, self.y_pred)
        logger.info(f"✓ Accuracy: {acc:.4f}")
    
    def plot_confusion_matrix(self):
        """Wizualizuj macierz pomyłek"""
        logger.info("\nGenerowanie macierzy pomyłek...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Przewidziano: 0', 'Przewidziano: 1'],
                   yticklabels=['Faktycznie: 0', 'Faktycznie: 1'])
        plt.title('Macierz Pomyłek (Confusion Matrix)', fontsize=14, fontweight='bold')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        
        # Dodaj statystyki
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        stats_text = f'Czułość (Recall): {sensitivity:.3f}\nSpecyficzność: {specificity:.3f}'
        plt.text(1.5, -0.35, stats_text, fontsize=10, ha='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix_advanced.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: confusion_matrix_advanced.png")
    
    def plot_roc_and_pr_curves(self):
        """Wizualizuj krzywe ROC i Precision-Recall"""
        logger.info("\nGenerowanie krzywych ROC i PR...")
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Krzywa ROC
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Krzywa Precision-Recall
        axes[1].plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "roc_pr_curves_advanced.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: roc_pr_curves_advanced.png")
    
    def plot_feature_importance(self):
        """Wizualizuj ważność cech"""
        logger.info("\nGenerowanie wykresu ważności cech...")
        
        importances = None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        
        if importances is not None:
            feat_imp = pd.DataFrame({
                'Feature': self.X_test.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            top_n = min(15, len(feat_imp))
            feat_imp = feat_imp.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue')
            
            # Koloruj najwyższą wartość innym kolorem
            bars[0].set_color('darkgreen')
            
            ax.set_title(f'Top {top_n} Cech - Ważność', fontsize=14, fontweight='bold')
            ax.set_xlabel('Ważność / Znaczenie')
            
            # Dodaj wartości na słupkach
            for i, (bar, val) in enumerate(zip(bars, feat_imp['Importance'])):
                ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
                       va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / "feature_importance_advanced.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✓ Zapisano: feature_importance_advanced.png")
        else:
            logger.warning("⚠ Model nie wspiera ważności cech")
    
    def plot_prediction_distribution(self):
        """Wizualizuj rozkład predykcji prawdopodobieństwa"""
        logger.info("\nGenerowanie rozkładu predykcji...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rozkład prawdopodobieństw dla klasy 0
        y_proba_0 = self.y_proba[self.y_test == 0]
        axes[0, 0].hist(y_proba_0, bins=30, edgecolor='black', color='red', alpha=0.7)
        axes[0, 0].set_title('Rozkład Predykcji - Klasa 0 (Niska/Średnia Ocena)')
        axes[0, 0].set_xlabel('Prawdopodobieństwo Klasy 1')
        axes[0, 0].set_ylabel('Częstość')
        axes[0, 0].axvline(0.5, color='black', linestyle='--', label='Próg (0.5)')
        axes[0, 0].legend()
        
        # Rozkład prawdopodobieństw dla klasy 1
        y_proba_1 = self.y_proba[self.y_test == 1]
        axes[0, 1].hist(y_proba_1, bins=30, edgecolor='black', color='green', alpha=0.7)
        axes[0, 1].set_title('Rozkład Predykcji - Klasa 1 (Wysoka Ocena)')
        axes[0, 1].set_xlabel('Prawdopodobieństwo Klasy 1')
        axes[0, 1].set_ylabel('Częstość')
        axes[0, 1].axvline(0.5, color='black', linestyle='--', label='Próg (0.5)')
        axes[0, 1].legend()
        
        # Łączny rozkład
        axes[1, 0].hist(self.y_proba[self.y_test == 0], bins=30, alpha=0.5, label='Klasa 0', color='red', edgecolor='black')
        axes[1, 0].hist(self.y_proba[self.y_test == 1], bins=30, alpha=0.5, label='Klasa 1', color='green', edgecolor='black')
        axes[1, 0].set_title('Łączny Rozkład Predykcji')
        axes[1, 0].set_xlabel('Prawdopodobieństwo Klasy 1')
        axes[1, 0].set_ylabel('Częstość')
        axes[1, 0].legend()
        axes[1, 0].axvline(0.5, color='black', linestyle='--', label='Próg (0.5)')
        
        # Box plot
        data_to_plot = [y_proba_0, y_proba_1]
        bp = axes[1, 1].boxplot(data_to_plot, labels=['Klasa 0', 'Klasa 1'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'green']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_title('Box Plot - Rozkład Predykcji')
        axes[1, 1].set_ylabel('Prawdopodobieństwo Klasy 1')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "prediction_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: prediction_distributions.png")
    
    def plot_error_analysis(self):
        """Analiza błędów klasyfikacji"""
        logger.info("\nGenerowanie analizy błędów...")
        
        errors = (self.y_pred != self.y_test).astype(int)
        error_proba = self.y_proba.copy()
        
        # Separuj błędy False Positive i False Negative
        false_positive = (self.y_pred == 1) & (self.y_test == 0)
        false_negative = (self.y_pred == 0) & (self.y_test == 1)
        true_positive = (self.y_pred == 1) & (self.y_test == 1)
        true_negative = (self.y_pred == 0) & (self.y_test == 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Macierz błędów wizualna
        cm = confusion_matrix(self.y_test, self.y_pred)
        labels = ['TN', 'FP', 'FN', 'TP']
        values = [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
        
        ax = axes[0, 0]
        bars = ax.bar(range(4), values, color=['green', 'red', 'red', 'green'])
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Liczba Próbek')
        ax.set_title('Matryca Błędów - Liczby Bezwzględne')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val, str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Rozkład prawdopodobieństw dla każdego typu
        ax = axes[0, 1]
        data_types = {
            'TP': self.y_proba[true_positive],
            'TN': self.y_proba[true_negative],
            'FP': self.y_proba[false_positive],
            'FN': self.y_proba[false_negative]
        }
        
        bp = ax.boxplot([data_types[k] for k in ['TP', 'TN', 'FP', 'FN']], 
                        labels=['TP', 'TN', 'FP', 'FN'],
                        patch_artist=True)
        
        colors = ['green', 'lightgreen', 'red', 'darkred']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Prawdopodobieństwo Predykcji')
        ax.set_title('Rozkład Prawdopodobieństw - Typy Błędów')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statystyki błędów
        ax = axes[1, 0]
        ax.axis('off')
        
        total = len(self.y_test)
        tp_count = true_positive.sum()
        tn_count = true_negative.sum()
        fp_count = false_positive.sum()
        fn_count = false_negative.sum()
        
        accuracy = (tp_count + tn_count) / total
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0
        
        stats_text = f"""
        METRYKI GŁÓWNE:
        Accuracy: {accuracy:.4f}
        Precision: {precision:.4f}
        Recall (Czułość): {recall:.4f}
        Specificity: {specificity:.4f}
        
        LICZBY BEZWZGLĘDNE:
        True Positive (TP): {tp_count}
        True Negative (TN): {tn_count}
        False Positive (FP): {fp_count}
        False Negative (FN): {fn_count}
        
        RAZEM: {total}
        """
        
        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11, 
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Macierz normalizowana (procenty)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[1, 1]
        sns.heatmap(cm_norm, annot=np.array([[f'{v:.1%}' for v in row] for row in cm_norm]),
                   fmt='', cmap='Blues', ax=ax, 
                   xticklabels=['Przewidziano: 0', 'Przewidziano: 1'],
                   yticklabels=['Faktycznie: 0', 'Faktycznie: 1'],
                   cbar_kws={'label': 'Procent'})
        ax.set_title('Macierz Pomyłek - Znormalizowana (%)')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Zapisano: error_analysis.png")
    
    def generate_classification_report(self):
        """Generuj raport klasyfikacji"""
        logger.info("\nGenerowanie raportu klasyfikacji...")
        
        report = classification_report(self.y_test, self.y_pred,
                                      target_names=['Klasa 0 (Niska/Średnia)', 'Klasa 1 (Wysoka)'],
                                      output_dict=True)
        
        print("\n" + "=" * 70)
        print("RAPORT KLASYFIKACJI")
        print("=" * 70)
        print(classification_report(self.y_test, self.y_pred,
                                   target_names=['Klasa 0 (Niska/Średnia)', 'Klasa 1 (Wysoka)']))
        
        report_path = self.reports_dir / "08_model_evaluation_advanced.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Raport zapisany: {report_path.name}")
    
    def run(self):
        """Uruchom całą ewaluację"""
        self.load_data_and_model()
        self.generate_predictions()
        self.plot_confusion_matrix()
        self.plot_roc_and_pr_curves()
        self.plot_feature_importance()
        self.plot_prediction_distribution()
        self.plot_error_analysis()
        self.generate_classification_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ZAAWANSOWANA EWALUACJA ZAKOŃCZONA")
        logger.info(f"Wyniki zapisano w: {self.figures_dir.relative_to(self.project_dir)}")
        logger.info("=" * 80 + "\n")


def main():
    evaluator = AdvancedModelEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
