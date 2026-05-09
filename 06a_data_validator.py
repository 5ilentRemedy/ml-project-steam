import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys, io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.df = None
        self.reports_dir = Path(__file__).parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.reports_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_name="games_final.parquet"):
        data_path = Path(__file__).parent / "data" / "processed" / file_name
        if not data_path.exists():
            logger.error(f"Plik danych nie znaleziony: {data_path}")
            sys.exit(1)
        
        try:
            self.df = pd.read_parquet(data_path)
            logger.info(f"Zaladowano dane z '{file_name}': {self.df.shape}")
        except Exception as e:
            logger.error(f"Blad podczas ladowania pliku Parquet: {e}")
            sys.exit(1)
        return self.df

    def generate_feature_histograms(self):
        """
        Generuje histogramy dla wszystkich istotnych cech numerycznych.
        """
        logger.info("Generowanie histogramow dla istotnych cech numerycznych...")
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        # Wyklucz kolumny, które są identyfikatorami lub mają specyficzny rozkład (np. binarne)
        # 'AppID' jest identyfikatorem, 'Is_free', 'Has_achievements', 'Is_highly_rated' są binarne
        # 'Release_year' może być traktowany jako kategoryczny lub numeryczny, ale histogram jest ok
        cols_to_exclude = ['AppID', 'Is_free', 'Has_achievements', 'Is_highly_rated']
        
        features_for_histograms = [col for col in numeric_cols if col not in cols_to_exclude]

        if not features_for_histograms:
            logger.warning("Brak odpowiednich cech numerycznych do wygenerowania histogramow.")
            return

        for col in features_for_histograms:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'Rozklad cechy: {col}')
            plt.xlabel(col)
            plt.ylabel('Czestosc')
            plt.grid(axis='y', alpha=0.75)
            output_file = self.plots_dir / f"histogram_{col}.png"
            plt.savefig(output_file)
            plt.close()
            logger.info(f"  [OK] Histogram dla '{col}' zapisany do: {output_file}")

    def analyze_correlations(self, significance_threshold=0.7):
        logger.info("Analiza wartosci korelacyjnych...")
        
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            logger.warning("Brak kolumn numerycznych do analizy korelacji.")
            return

        corr_matrix = numeric_df.corr()
        
        # 1. Generowanie histogramu rozkładu korelacji
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_values = upper_tri.unstack().dropna()

        if not corr_values.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(corr_values, bins=30, kde=True)
            plt.title('Rozklad Wartosci Korelacji Miedzy Cechami Numerycznymi')
            plt.xlabel('Wartosc Korelacji')
            plt.ylabel('Czestosc')
            plt.grid(axis='y', alpha=0.75)
            output_file = self.plots_dir / "correlation_distribution_histogram.png"
            plt.savefig(output_file)
            plt.close()
            logger.info(f"[OK] Histogram rozkladu korelacji zapisany do: {output_file}")
        else:
            logger.warning("Brak wystarczajacych danych do wygenerowania histogramu korelacji.")

        # 2. Generowanie heatmapy korelacji
        plt.figure(figsize=(14, 12)) # Zwiększono rozmiar figury dla lepszej czytelności i uniknięcia ucinania
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Macierz Korelacji Cech Numerycznych')
        plt.tight_layout() # Automatyczne dopasowanie układu, aby zapobiec ucinaniu elementów
        output_file_heatmap = self.plots_dir / "correlation_heatmap.png"
        plt.savefig(output_file_heatmap)
        plt.close()
        logger.info(f"[OK] Heatmapa korelacji zapisana do: {output_file_heatmap}")

        # 3. Raportowanie istotnych korelacji
        logger.info(f"Identyfikowanie istotnych korelacji (abs > {significance_threshold})...")
        significant_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= significance_threshold:
                    significant_correlations.append({
                        'Cecha_1': col1,
                        'Cecha_2': col2,
                        'Korelacja': correlation,
                        'Abs_Korelacja': abs(correlation)
                    })
        
        if significant_correlations:
            significant_corr_df = pd.DataFrame(significant_correlations).sort_values(by='Abs_Korelacja', ascending=False)
            output_file_report = self.reports_dir / "significant_correlations_report.csv"
            significant_corr_df.to_csv(output_file_report, index=False)
            logger.info(f"[OK] Raport istotnych korelacji zapisany do: {output_file_report}")
            logger.info("  Najsilniejsze korelacje:")
            logger.info(significant_corr_df.head())
        else:
            logger.info(f"[INFO] Brak korelacji o wartosci bezwzglednej wiekszej niz {significance_threshold}.")

    def detect_outliers_iqr(self, threshold=1.5):
        """
        Wykrywa outliery w kolumnach numerycznych za pomocą metody IQR.
        Outliery to wartości poza zakresem [Q1 - threshold*IQR, Q3 + threshold*IQR].
        Generuje raport i boxploty dla kolumn z outlierami.
        """
        logger.info(f"Wykrywanie outlierow za pomoca metody IQR (threshold={threshold})...")
        
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            logger.warning("Brak kolumn numerycznych do wykrywania outlierow.")
            return

        outlier_report = []
        
        # Wyklucz kolumny binarne i identyfikatory, które nie mają sensu dla outlierów IQR
        cols_to_exclude_from_outliers = ['AppID', 'Is_free', 'Has_achievements', 'Is_highly_rated']
        features_for_outlier_detection = [col for col in numeric_df.columns if col not in cols_to_exclude_from_outliers]

        if not features_for_outlier_detection:
            logger.warning("Brak odpowiednich cech numerycznych do wykrywania outlierow.")
            return

        for column in features_for_outlier_detection:
            Q1 = numeric_df[column].quantile(0.25)
            Q3 = numeric_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Znajdź outliery
            outliers = numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)]
            
            if not outliers.empty:
                num_outliers = len(outliers)
                percentage_outliers = (num_outliers / len(self.df)) * 100
                outlier_report.append({
                    'Kolumna': column,
                    'Liczba_Outlierow': num_outliers,
                    'Procent_Outlierow': f"{percentage_outliers:.2f}%",
                    'Dolna_Granica_IQR': lower_bound,
                    'Gorna_Granica_IQR': upper_bound
                })
                logger.info(f"  [OK] Kolumna '{column}': znaleziono {num_outliers} outlierow ({percentage_outliers:.2f}%)")

                # Generowanie boxplotu dla kolumny z outlierami
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=numeric_df[column])
                plt.title(f'Boxplot dla kolumny: {column} (z outlierami)')
                plt.ylabel(column)
                boxplot_file = self.plots_dir / f"boxplot_outliers_{column}.png"
                plt.savefig(boxplot_file)
                plt.close()
                logger.info(f"  [OK] Boxplot dla '{column}' zapisany do: {boxplot_file}")
            else:
                logger.info(f"  [INFO] Kolumna '{column}': brak outlierow.")

        if outlier_report:
            outlier_df = pd.DataFrame(outlier_report)
            output_file = self.reports_dir / "outlier_report_iqr.csv"
            outlier_df.to_csv(output_file, index=False)
            logger.info(f"[OK] Raport outlierow zapisany do: {output_file}")
        else:
            logger.info("[INFO] Brak outlierow wykrytych w zadnej kolumnie.")

        return outlier_report

    def run(self):
        """
        Orkiestruje proces walidacji i analizy danych.
        """
        logger.info("\n" + "=" * 80)
        logger.info("WALIDACJA I ANALIZA DANYCH")
        logger.info("=" * 80 + "\n")

        self.load_data()
        if self.df is None:
            logger.error("Nie udalo sie zaladowac danych. Przerywam walidacje.")
            return

        # 1. Generowanie histogramów dla istotnych cech
        self.generate_feature_histograms()

        # 2. Analiza wartości korelacyjnych
        self.analyze_correlations()

        # 3. Analiza wartości odstających i zidentyfikowanie ich
        self.detect_outliers_iqr()

        logger.info("\n" + "=" * 80)
        logger.info("[OK] WALIDACJA DANYCH UKONCZONA")
        logger.info("=" * 80 + "\n")
        logger.info(f"Raporty i wykresy zapisane w katalogach: {self.reports_dir} i {self.plots_dir}")

def main():
    validator = DataValidator()
    validator.run()

if __name__ == "__main__":
    # Upewnij się, że katalog 'data/processed' istnieje i zawiera 'games_final.parquet'
    # Możesz uruchomić 06_data_export.py przed tym skryptem, aby wygenerować te pliki.
    
    # Przyklad uruchomienia:
    # python 06a_data_validator.py
    main()
