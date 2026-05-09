# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import logging # Do logowania informacji
import sys # Do interakcji z systemem operacyjnym
import io # Do obsługi strumieni wejścia/wyjścia
import joblib # Do serializacji i deserializacji obiektów Pythona (np. modeli)
import json # Do pracy z formatem JSON

# Narzędzia do wizualizacji i metryk z scikit-learn
import matplotlib.pyplot as plt # Do tworzenia wykresów
import seaborn as sns # Do tworzenia atrakcyjnych wykresów statystycznych
from sklearn.metrics import (
    classification_report, # Raport klasyfikacji
    confusion_matrix, # Macierz pomyłek
    roc_curve, # Krzywa ROC
    auc, # Pole pod krzywą (Area Under Curve)
    precision_recall_curve, # Krzywa Precision-Recall
    accuracy_score # Dokładność
)

# Próba importu biblioteki SHAP do wyjaśnialnego AI (XAI)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    
# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do ewaluacji modelu
class ModelEvaluator:
    # Inicjalizacja obiektu ModelEvaluator
    def __init__(self):
        self.project_dir = Path(__file__).parent # Ustalenie katalogu projektu
        self.processed_dir = self.project_dir / "data" / "processed" # Ścieżka do przetworzonych danych
        self.models_dir = self.project_dir / "models" # Ścieżka do katalogu modeli
        self.reports_dir = self.project_dir / "reports" # Ścieżka do katalogu raportów
        self.figures_dir = self.reports_dir / "figures" # Ścieżka do katalogu wykresów
        
        # Utworzenie katalogu na wykresy, jeśli nie istnieje
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_df = None # Ramka danych dla zbioru testowego
        self.model = None # Załadowany model
        self.target_col = 'Is_highly_rated' # Kolumna docelowa
        
        self.X_test = None # Cechy zbioru testowego
        self.y_test = None # Zmienna docelowa zbioru testowego
        self.y_pred = None # Predykcje klas na zbiorze testowym
        self.y_proba = None # Prawdopodobieństwa predykcji na zbiorze testowym

    # Metoda do ładowania zbioru testowego i najlepszego zapisanego modelu
    def load_data_and_model(self):
        logger.info("Ladowanie danych testowych i modelu...")
        
        # Ścieżki do plików danych testowych i modelu
        test_path = self.processed_dir / "games_test.csv"
        model_path = self.models_dir / "best_model.joblib"
        
        # Sprawdzenie, czy pliki istnieją
        if not test_path.exists():
            raise FileNotFoundError(f"Brak pliku danych testowych: {test_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}. Uruchom najpierw 07_model_training.py.")
            
        # Wczytanie danych testowych do DataFrame
        self.test_df = pd.read_csv(test_path)
        # Załadowanie modelu za pomocą joblib
        self.model = joblib.load(model_path)
        
        logger.info(f"OK Zbior testowy: {self.test_df.shape[0]} wierszy")
        logger.info(f"OK Model zaladowany: {type(self.model).__name__}")
        
        # Przygotowanie cech (X) i zmiennej docelowej (y) - musi być identyczne jak w skrypcie 07
        cols_to_drop = ['AppID', 'Name', 'Genres', self.target_col]
        drop_test = [c for c in cols_to_drop if c in self.test_df.columns]
        
        # Podział danych na cechy (X_test) i zmienną docelową (y_test)
        self.X_test = self.test_df.drop(columns=drop_test).fillna(0)
        self.y_test = self.test_df[self.target_col]

    # Metoda do generowania predykcji i prawdopodobieństw
    def generate_predictions(self):
        logger.info("Generowanie predykcji na zbiorze testowym...")
        # Generowanie predykcji klas
        self.y_pred = self.model.predict(self.X_test)
        
        # Generowanie prawdopodobieństw predykcji, jeśli model to wspiera
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            # Fallback, jeśli model nie wspiera predict_proba (np. niektóre modele SVM)
            self.y_proba = self.y_pred
            
        # Obliczenie i zalogowanie dokładności
        acc = accuracy_score(self.y_test, self.y_pred)
        logger.info(f"OK Dokladnosc (Accuracy) na tescie: {acc:.4f}")

    # Metoda do generowania pełnego raportu klasyfikacji
    def evaluate_metrics(self):
        logger.info("\n" + "-" * 50)
        logger.info("RAPORT KLASYFIKACJI")
        logger.info("-" * 50)
        
        # Wygenerowanie i wydrukowanie raportu klasyfikacji
        report = classification_report(self.y_test, self.y_pred, target_names=['Niska/Srednia Ocena', 'Wysoka Ocena (>=75)'])
        print(report)
        
        # Konwersja raportu do słownika i zapisanie do pliku JSON
        report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
        with open(self.reports_dir / "05_evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # Metoda do rysowania macierzy pomyłek
    def plot_confusion_matrix(self):
        logger.info("Generowanie macierzy pomylek...")
        # Obliczenie macierzy pomyłek
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6)) # Ustawienie rozmiaru wykresu
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Przewidziano: Nie', 'Przewidziano: Tak'],
                    yticklabels=['Faktycznie: Nie', 'Faktycznie: Tak'])
        plt.title('Macierz Pomylek (Confusion Matrix)')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        
        # Zapisanie wykresu do pliku
        out_path = self.figures_dir / "confusion_matrix.png"
        plt.tight_layout() # Dopasowanie układu
        plt.savefig(out_path, dpi=300) # Zapisanie z wysoką rozdzielczością
        plt.close() # Zamknięcie wykresu
        logger.info(f"  OK Zapisano: {out_path.name}")

    # Metoda do rysowania krzywej ROC oraz Precision-Recall
    def plot_roc_curve(self):
        logger.info("Generowanie krzywej ROC i PR...")
        
        # Obliczenie krzywej ROC i AUC-ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12, 5)) # Ustawienie rozmiaru wykresu
        
        # Wykres krzywej ROC
        plt.subplot(1, 2, 1) # Pierwszy subplot w siatce 1x2
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})') # Rysowanie krzywej ROC
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Rysowanie linii losowego klasyfikatora
        plt.xlim([0.0, 1.0]) # Zakres osi X
        plt.ylim([0.0, 1.05]) # Zakres osi Y
        plt.xlabel('False Positive Rate') # Etykieta osi X
        plt.ylabel('True Positive Rate') # Etykieta osi Y
        plt.title('Receiver Operating Characteristic (ROC)') # Tytuł wykresu
        plt.legend(loc="lower right") # Legenda
        
        # Obliczenie krzywej Precision-Recall i AUC-PR
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        
        # Wykres krzywej Precision-Recall
        plt.subplot(1, 2, 2) # Drugi subplot w siatce 1x2
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})') # Rysowanie krzywej PR
        plt.xlabel('Recall') # Etykieta osi X
        plt.ylabel('Precision') # Etykieta osi Y
        plt.title('Precision-Recall Curve') # Tytuł wykresu
        plt.legend(loc="lower left") # Legenda
        
        # Zapisanie wykresu do pliku
        out_path = self.figures_dir / "roc_pr_curves.png"
        plt.tight_layout() # Dopasowanie układu
        plt.savefig(out_path, dpi=300) # Zapisanie z wysoką rozdzielczością
        plt.close() # Zamknięcie wykresu
        logger.info(f"  OK Zapisano: {out_path.name}")

    # Metoda do rysowania ważności cech, jeśli model to obsługuje
    def plot_feature_importance(self):
        logger.info("Generowanie wykresu waznosci cech...")
        
        importances = None # Zmienna do przechowywania ważności cech
        # Sprawdzenie, czy model posiada atrybut 'feature_importances_' (dla modeli drzewiastych)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        # Sprawdzenie, czy model posiada atrybut 'coef_' (dla modeli liniowych)
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            
        # Jeśli ważności cech są dostępne, wygeneruj wykres
        if importances is not None:
            # Utworzenie DataFrame z cechami i ich ważnością, posortowanie i wybranie top 15
            feat_imp = pd.DataFrame({
                'Feature': self.X_test.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True).tail(15)
            
            plt.figure(figsize=(10, 8)) # Ustawienie rozmiaru wykresu
            plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue') # Rysowanie wykresu słupkowego poziomego
            plt.title(f'Waznosc Cech (Top 15) - {type(self.model).__name__}') # Tytuł wykresu
            plt.xlabel('Waga / Znaczenie') # Etykieta osi X
            
            # Zapisanie wykresu do pliku
            out_path = self.figures_dir / "feature_importance.png"
            plt.tight_layout() # Dopasowanie układu
            plt.savefig(out_path, dpi=300) # Zapisanie z wysoką rozdzielczością
            plt.close() # Zamknięcie wykresu
            logger.info(f"  OK Zapisano: {out_path.name}")
        else:
            logger.info("  Ten model nie udostepnia atrybutu waznosci cech.") # Informacja, jeśli ważności cech nie są dostępne

    # Metoda do wykonywania analizy SHAP (Explainable AI)
    def run_shap_analysis(self):
        # Sprawdzenie, czy biblioteka SHAP jest zainstalowana
        if not HAS_SHAP:
            logger.warning("Biblioteka SHAP nie jest zainstalowana. Pomijanie zaawansowanej interpretacji.")
            logger.warning("Zainstaluj uzywajac: pip install shap")
            return
            
        logger.info("Generowanie analizy SHAP (Explainable AI)... Moze to chwile potrwac.")
        try:
            # Użycie mniejszej próbki danych do analizy SHAP w celu przyspieszenia obliczeń
            X_sample = shap.sample(self.X_test, 1000)
            
            # Wybór odpowiedniego explainera SHAP w zależności od typu modelu
            if type(self.model).__name__ in ['RandomForestClassifier', 'LGBMClassifier', 'XGBClassifier']:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                
                # Obsługa modeli zwracających wartości SHAP dla obu klas w postaci listy
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Wybranie wartości SHAP dla klasy pozytywnej
            else:
                explainer = shap.Explainer(self.model, X_sample)
                shap_values = explainer(X_sample).values
            
            plt.figure(figsize=(12, 8)) # Ustawienie rozmiaru wykresu
            shap.summary_plot(shap_values, X_sample, show=False) # Generowanie wykresu podsumowującego SHAP
            
            # Zapisanie wykresu do pliku
            out_path = self.figures_dir / "shap_summary.png"
            plt.tight_layout() # Dopasowanie układu
            plt.savefig(out_path, dpi=300, bbox_inches='tight') # Zapisanie z wysoką rozdzielczością
            plt.close() # Zamknięcie wykresu
            logger.info(f"  OK Zapisano: {out_path.name}")
            
        # Obsługa błędów podczas generowania analizy SHAP
        except Exception as e:
            logger.error(f"  BLAD Nie udalo sie wygenerowac wykresu SHAP: {str(e)}")

    # Metoda uruchamiająca cały proces ewaluacji modelu
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("EWALUACJA MODELU ML")
        logger.info("=" * 80 + "\n")
        
        self.load_data_and_model() # Załadowanie danych testowych i wytrenowanego modelu
        self.generate_predictions() # Generowanie predykcji i prawdopodobieństw
        self.evaluate_metrics() # Ocena metryk klasyfikacji i zapisanie raportu
        self.plot_confusion_matrix() # Wygenerowanie i zapisanie macierzy pomyłek
        self.plot_roc_curve() # Wygenerowanie i zapisanie krzywych ROC i Precision-Recall
        self.plot_feature_importance() # Wygenerowanie i zapisanie wykresu ważności cech
        self.run_shap_analysis() # Wykonanie i zapisanie analizy SHAP
        
        logger.info("\n" + "=" * 80)
        logger.info("OK EWALUACJA ZAKONCZONA") # Informacja o zakończeniu ewaluacji
        logger.info(f"Wyniki zapisano w: {self.figures_dir.relative_to(self.project_dir)}")
        logger.info("=" * 80 + "\n")

# Główna funkcja programu
def main():
    evaluator = ModelEvaluator() # Utworzenie instancji klasy ModelEvaluator
    evaluator.run() # Uruchomienie procesu ewaluacji

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
