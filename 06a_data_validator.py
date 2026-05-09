# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import logging # Do logowania informacji
import matplotlib.pyplot as plt # Do tworzenia wykresów
import seaborn as sns # Do tworzenia atrakcyjnych wykresów statystycznych
import sys, io # Do obsługi strumieni wejścia/wyjścia

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do walidacji danych
class DataValidator:
    # Inicjalizacja obiektu DataValidator
    def __init__(self):
        self.df = None # Ramka danych, która będzie walidowana
        # Ustalenie ścieżki do katalogu 'reports' i utworzenie go, jeśli nie istnieje
        self.reports_dir = Path(__file__).parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        # Ustalenie ścieżki do podkatalogu 'plots' w 'reports' i utworzenie go, jeśli nie istnieje
        self.plots_dir = self.reports_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # Metoda do ładowania danych z pliku Parquet
    def load_data(self, file_name="games_final.parquet"):
        # Ustalenie pełnej ścieżki do pliku danych
        data_path = Path(__file__).parent / "data" / "processed" / file_name
        # Sprawdzenie, czy plik danych istnieje
        if not data_path.exists():
            logger.error(f"Plik danych nie znaleziony: {data_path}")
            sys.exit(1) # Zakończenie działania skryptu z błędem
        
        try:
            # Wczytanie danych z pliku Parquet do DataFrame
            self.df = pd.read_parquet(data_path)
            logger.info(f"Zaladowano dane z '{file_name}': {self.df.shape}")
        except Exception as e:
            logger.error(f"Blad podczas ladowania pliku Parquet: {e}")
            sys.exit(1) # Zakończenie działania skryptu z błędem
        return self.df

    # Metoda do generowania histogramów dla istotnych cech numerycznych
    def generate_feature_histograms(self):
        logger.info("Generowanie histogramow dla istotnych cech numerycznych...")
        # Wybór kolumn numerycznych z DataFrame
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        # Lista kolumn do wykluczenia z generowania histogramów (np. identyfikatory, kolumny binarne)
        cols_to_exclude = ['AppID', 'Is_free', 'Has_achievements', 'Is_highly_rated']
        
        # Filtrowanie kolumn numerycznych, aby uzyskać listę cech do histogramów
        features_for_histograms = [col for col in numeric_cols if col not in cols_to_exclude]

        # Sprawdzenie, czy są cechy do wygenerowania histogramów
        if not features_for_histograms:
            logger.warning("Brak odpowiednich cech numerycznych do wygenerowania histogramow.")
            return

        # Iteracja przez wybrane cechy i generowanie histogramów
        for col in features_for_histograms:
            plt.figure(figsize=(10, 6)) # Ustawienie rozmiaru wykresu
            sns.histplot(self.df[col], kde=True, bins=30) # Tworzenie histogramu z estymacją gęstości jądrowej
            plt.title(f'Rozklad cechy: {col}') # Ustawienie tytułu wykresu
            plt.xlabel(col) # Etykieta osi X
            plt.ylabel('Czestosc') # Etykieta osi Y
            plt.grid(axis='y', alpha=0.75) # Dodanie siatki na osi Y
            output_file = self.plots_dir / f"histogram_{col}.png" # Definiowanie nazwy pliku wyjściowego
            plt.savefig(output_file) # Zapisanie wykresu do pliku
            plt.close() # Zamknięcie wykresu, aby zwolnić pamięć
            logger.info(f"  OK Histogram dla '{col}' zapisany do: {output_file}")

    # Metoda do analizy wartości korelacyjnych
    def analyze_correlations(self, significance_threshold=0.7):
        logger.info("Analiza wartosci korelacyjnych...")
        
        # Wybór kolumn numerycznych z DataFrame
        numeric_df = self.df.select_dtypes(include=np.number)
        # Sprawdzenie, czy są kolumny numeryczne do analizy
        if numeric_df.empty:
            logger.warning("Brak kolumn numerycznych do analizy korelacji.")
            return

        corr_matrix = numeric_df.corr() # Obliczenie macierzy korelacji
        
        # 1. Generowanie histogramu rozkładu korelacji
        # Wyodrębnienie górnego trójkąta macierzy korelacji (bez przekątnej)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_values = upper_tri.unstack().dropna() # Rozwinięcie macierzy do serii i usunięcie NaN

        # Sprawdzenie, czy są wartości korelacji do wygenerowania histogramu
        if not corr_values.empty:
            plt.figure(figsize=(10, 6)) # Ustawienie rozmiaru wykresu
            sns.histplot(corr_values, bins=30, kde=True) # Tworzenie histogramu
            plt.title('Rozklad Wartosci Korelacji Miedzy Cechami Numerycznymi') # Ustawienie tytułu
            plt.xlabel('Wartosc Korelacji') # Etykieta osi X
            plt.ylabel('Czestosc') # Etykieta osi Y
            plt.grid(axis='y', alpha=0.75) # Dodanie siatki
            output_file = self.plots_dir / "correlation_distribution_histogram.png" # Nazwa pliku
            plt.savefig(output_file) # Zapisanie wykresu
            plt.close() # Zamknięcie wykresu
            logger.info(f"OK Histogram rozkladu korelacji zapisany do: {output_file}")
        else:
            logger.warning("Brak wystarczajacych danych do wygenerowania histogramu korelacji.")

        # 2. Generowanie heatmapy korelacji
        plt.figure(figsize=(14, 12)) # Zwiększony rozmiar figury dla lepszej czytelności
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5) # Tworzenie heatmapy
        plt.title('Macierz Korelacji Cech Numerycznych') # Ustawienie tytułu
        plt.tight_layout() # Automatyczne dopasowanie układu, aby zapobiec ucinaniu elementów
        output_file_heatmap = self.plots_dir / "correlation_heatmap.png" # Nazwa pliku
        plt.savefig(output_file_heatmap) # Zapisanie wykresu
        plt.close() # Zamknięcie wykresu
        logger.info(f"OK Heatmapa korelacji zapisana do: {output_file_heatmap}")

        # 3. Raportowanie istotnych korelacji
        logger.info(f"Identyfikowanie istotnych korelacji (abs > {significance_threshold})...")
        significant_correlations = [] # Lista do przechowywania istotnych korelacji
        # Iteracja przez macierz korelacji (tylko górny trójkąt)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                # Sprawdzenie, czy wartość bezwzględna korelacji przekracza próg istotności
                if abs(correlation) >= significance_threshold:
                    significant_correlations.append({
                        'Cecha_1': col1,
                        'Cecha_2': col2,
                        'Korelacja': correlation,
                        'Abs_Korelacja': abs(correlation)
                    })
        
        # Zapisanie raportu istotnych korelacji, jeśli takie znaleziono
        if significant_correlations:
            significant_corr_df = pd.DataFrame(significant_correlations).sort_values(by='Abs_Korelacja', ascending=False) # Utworzenie DataFrame i sortowanie
            output_file_report = self.reports_dir / "significant_correlations_report.csv" # Nazwa pliku raportu
            significant_corr_df.to_csv(output_file_report, index=False) # Zapisanie do CSV
            logger.info(f"OK Raport istotnych korelacji zapisany do: {output_file_report}")
            logger.info("  Najsilniejsze korelacje:")
            logger.info(significant_corr_df.head()) # Wyświetlenie kilku najsilniejszych korelacji
        else:
            logger.info(f"INFO Brak korelacji o wartosci bezwzglednej wiekszej niz {significance_threshold}.")

    # Metoda do wykrywania outlierów za pomocą metody IQR
    def detect_outliers_iqr(self, threshold=1.5):
        logger.info(f"Wykrywanie outlierow za pomoca metody IQR (threshold={threshold})...")
        
        # Wybór kolumn numerycznych
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            logger.warning("Brak kolumn numerycznych do wykrywania outlierow.")
            return

        outlier_report = [] # Lista do przechowywania informacji o outlierach
        
        # Kolumny do wykluczenia z detekcji outlierów (np. binarne, identyfikatory)
        cols_to_exclude_from_outliers = ['AppID', 'Is_free', 'Has_achievements', 'Is_highly_rated']
        features_for_outlier_detection = [col for col in numeric_df.columns if col not in cols_to_exclude_from_outliers]

        # Sprawdzenie, czy są cechy do detekcji outlierów
        if not features_for_outlier_detection:
            logger.warning("Brak odpowiednich cech numerycznych do wykrywania outlierow.")
            return

        # Iteracja przez wybrane cechy w celu detekcji outlierów
        for column in features_for_outlier_detection:
            Q1 = numeric_df[column].quantile(0.25) # Pierwszy kwartyl
            Q3 = numeric_df[column].quantile(0.75) # Trzeci kwartyl
            IQR = Q3 - Q1 # Rozstęp międzykwartylowy
            
            lower_bound = Q1 - threshold * IQR # Dolna granica dla outlierów
            upper_bound = Q3 + threshold * IQR # Górna granica dla outlierów
            
            # Znalezienie outlierów
            outliers = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]
            
            # Zapisanie informacji o outlierach i generowanie boxplotu, jeśli outliery istnieją
            if not outliers.empty:
                num_outliers = len(outliers) # Liczba outlierów
                percentage_outliers = (num_outliers / len(self.df)) * 100 # Procent outlierów
                outlier_report.append({
                    'Kolumna': column,
                    'Liczba_Outlierow': num_outliers,
                    'Procent_Outlierow': f"{percentage_outliers:.2f}%",
                    'Dolna_Granica_IQR': lower_bound,
                    'Gorna_Granica_IQR': upper_bound
                })
                logger.info(f"  OK Kolumna '{column}': znaleziono {num_outliers} outlierow ({percentage_outliers:.2f}%)")

                # Generowanie boxplotu dla kolumny z outlierami
                plt.figure(figsize=(8, 6)) # Ustawienie rozmiaru wykresu
                sns.boxplot(y=numeric_df[column]) # Tworzenie boxplotu
                plt.title(f'Boxplot dla kolumny: {column} (z outlierami)') # Ustawienie tytułu
                plt.ylabel(column) # Etykieta osi Y
                boxplot_file = self.plots_dir / f"boxplot_outliers_{column}.png" # Nazwa pliku
                plt.savefig(boxplot_file) # Zapisanie wykresu
                plt.close() # Zamknięcie wykresu
                logger.info(f"  OK Boxplot dla '{column}' zapisany do: {boxplot_file}")
            else:
                logger.info(f"  INFO Kolumna '{column}': brak outlierow.")

        # Zapisanie raportu outlierów do pliku CSV
        if outlier_report:
            outlier_df = pd.DataFrame(outlier_report) # Utworzenie DataFrame z raportu
            output_file = self.reports_dir / "outlier_report_iqr.csv" # Nazwa pliku raportu
            outlier_df.to_csv(output_file, index=False) # Zapisanie do CSV
            logger.info(f"OK Raport outlierow zapisany do: {output_file}")
        else:
            logger.info("INFO Brak outlierow wykrytych w zadnej kolumnie.")

        return outlier_report

    # Metoda uruchamiająca cały proces walidacji i analizy danych
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("WALIDACJA I ANALIZA DANYCH")
        logger.info("=" * 80 + "\n")

        self.load_data() # Załadowanie danych
        if self.df is None:
            logger.error("Nie udalo sie zaladowac danych. Przerywam walidacje.")
            return

        # Generowanie histogramów dla istotnych cech numerycznych
        self.generate_feature_histograms()

        # Analiza wartości korelacyjnych
        self.analyze_correlations()

        # Analiza wartości odstających i zidentyfikowanie ich
        self.detect_outliers_iqr()

        logger.info("\n" + "=" * 80)
        logger.info("OK WALIDACJA DANYCH UKONCZONA")
        logger.info("=" * 80 + "\n")
        logger.info(f"Raporty i wykresy zapisane w katalogach: {self.reports_dir} i {self.plots_dir}")

# Główna funkcja programu
def main():
    validator = DataValidator() # Utworzenie instancji klasy DataValidator
    validator.run() # Uruchomienie procesu walidacji

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
