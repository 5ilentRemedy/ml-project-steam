# ============================================================================
# ZAAWANSOWANA ANALIZA DANYCH - Sprawdzenie Wszystkich Założeń
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import io
import json
import warnings

# Wizualizacja
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ustawienie kodowania UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ComprehensiveAnalysis:
    """Zaawansowana analiza danych - sprawdzenie wszystkich założeń"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir / "data"
        self.reports_dir = self.project_dir / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.analysis_report = {}
    
    def load_latest_data(self):
        """Załaduj najnowsze dane"""
        logger.info("=" * 80)
        logger.info("ZAAWANSOWANA ANALIZA DANYCH")
        logger.info("=" * 80)
        
        # Spróbuj załadować dane w kolejności: engineered -> cleaned -> raw
        for filename in ['games_engineered.csv', 'games_cleaned.csv']:
            filepath = self.data_dir / filename
            if filepath.exists():
                self.df = pd.read_csv(filepath, index_col=False)
                logger.info(f"✓ Załadowano dane: {filename}")
                logger.info(f"  Rozmiar: {self.df.shape}")
                return self.df
        
        raise FileNotFoundError("Nie znaleziono danych do analizy")
    
    def check_duplicates(self):
        """Sprawdź duplikaty"""
        logger.info("\n" + "-" * 60)
        logger.info("1. DUPLIKATY")
        logger.info("-" * 60)
        
        if 'AppID' not in self.df.columns:
            logger.warning("Brak kolumny AppID")
            self.analysis_report['duplicates'] = {'status': 'unknown'}
            return
        
        dup_count = self.df['AppID'].duplicated().sum()
        total_rows = len(self.df)
        
        if dup_count == 0:
            logger.info(f"✓ BRAK duplikatów (0 z {total_rows} wierszy)")
            self.analysis_report['duplicates'] = {'status': 'OK', 'count': 0}
        else:
            logger.warning(f"⚠ Znaleziono {dup_count} duplikatów")
            self.analysis_report['duplicates'] = {'status': 'WARNING', 'count': dup_count}
    
    def check_missing_data(self):
        """Sprawdź braki danych"""
        logger.info("\n" + "-" * 60)
        logger.info("2. BRAKI DANYCH")
        logger.info("-" * 60)
        
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df) * 100)
        
        has_missing = (missing > 0).any()
        
        if not has_missing:
            logger.info(f"✓ BRAK braków danych - wszystkie kolumny kompletne")
            self.analysis_report['missing_data'] = {'status': 'OK', 'count': 0}
        else:
            logger.warning(f"⚠ Znaleziono braki danych:")
            for col in self.df.columns[missing > 0]:
                logger.warning(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
            self.analysis_report['missing_data'] = {
                'status': 'WARNING', 
                'count': missing.sum(),
                'columns': {col: {'count': int(missing[col]), 'pct': float(missing_pct[col])} 
                           for col in self.df.columns[missing > 0]}
            }
    
    def check_numeric_encoding(self):
        """Sprawdź czy dane są zakodowane liczbowo"""
        logger.info("\n" + "-" * 60)
        logger.info("3. KODOWANIE LICZBOWE")
        logger.info("-" * 60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        object_cols = self.df.select_dtypes(include=['object']).columns
        
        logger.info(f"  Kolumny numeryczne: {len(numeric_cols)} ({len(numeric_cols)/len(self.df.columns)*100:.0f}%)")
        logger.info(f"  Kolumny tekstowe: {len(object_cols)} ({len(object_cols)/len(self.df.columns)*100:.0f}%)")
        
        if len(object_cols) > 0:
            logger.info(f"  Kolumny tekstowe: {list(object_cols)}")
            # Sprawdź czy to już one-hot encoded
            for col in object_cols[:5]:
                unique = self.df[col].nunique()
                logger.info(f"    {col}: {unique} unikalnych wartości")
        
        self.analysis_report['encoding'] = {
            'status': 'OK' if len(object_cols) <= 3 else 'WARNING',
            'numeric_cols': len(numeric_cols),
            'object_cols': len(object_cols),
            'object_columns': list(object_cols)
        }
    
    def check_standardization(self):
        """Sprawdź czy dane są wystandaryzowane"""
        logger.info("\n" + "-" * 60)
        logger.info("4. STANDARYZACJA DANYCH")
        logger.info("-" * 60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        standardized_count = 0
        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            
            # Dane wystandaryzowane mają mean ≈ 0 i std ≈ 1
            if abs(mean) < 0.1 and abs(std - 1) < 0.1:
                standardized_count += 1
        
        if standardized_count > len(numeric_cols) * 0.5:
            logger.info(f"✓ Większość cech wydaje się wystandaryzowana ({standardized_count}/{len(numeric_cols)})")
            status = 'OK'
        else:
            logger.warning(f"⚠ Cechy NIE są w pełni wystandaryzowane (standaryzowanych: {standardized_count}/{len(numeric_cols)})")
            logger.info("  Przykłady statystyk:")
            for col in list(numeric_cols)[:5]:
                logger.info(f"    {col}: mean={self.df[col].mean():.2f}, std={self.df[col].std():.2f}, "
                          f"min={self.df[col].min():.2f}, max={self.df[col].max():.2f}")
            status = 'WARNING'
        
        self.analysis_report['standardization'] = {
            'status': status,
            'standardized_cols': standardized_count,
            'total_cols': len(numeric_cols)
        }
    
    def check_data_split(self):
        """Sprawdź podział 70/15/15"""
        logger.info("\n" + "-" * 60)
        logger.info("5. PODZIAŁ DANYCH (70/15/15)")
        logger.info("-" * 60)
        
        processed_dir = self.data_dir / "processed"
        
        files_exist = {
            'train': (processed_dir / "games_train.csv").exists(),
            'val': (processed_dir / "games_val.csv").exists(),
            'test': (processed_dir / "games_test.csv").exists()
        }
        
        if all(files_exist.values()):
            train_df = pd.read_csv(processed_dir / "games_train.csv")
            val_df = pd.read_csv(processed_dir / "games_val.csv")
            test_df = pd.read_csv(processed_dir / "games_test.csv")
            
            total = len(train_df) + len(val_df) + len(test_df)
            train_pct = len(train_df) / total * 100
            val_pct = len(val_df) / total * 100
            test_pct = len(test_df) / total * 100
            
            logger.info(f"✓ Zbiory podzielone:")
            logger.info(f"  Train: {len(train_df)} ({train_pct:.1f}%)")
            logger.info(f"  Val:   {len(val_df)} ({val_pct:.1f}%)")
            logger.info(f"  Test:  {len(test_df)} ({test_pct:.1f}%)")
            
            self.analysis_report['data_split'] = {
                'status': 'OK',
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df),
                'ratios': {'train': train_pct, 'val': val_pct, 'test': test_pct}
            }
            return train_df, val_df, test_df
        else:
            logger.warning("⚠ Zbiory train/val/test nie istnieją")
            self.analysis_report['data_split'] = {'status': 'NOT_FOUND'}
            return None, None, None
    
    def analyze_correlations(self):
        """Analiza korelacji między cechami"""
        logger.info("\n" + "-" * 60)
        logger.info("6. KORELACJE MIĘDZY CECHAMI")
        logger.info("-" * 60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("Brak kolumn numerycznych do analizy korelacji")
            return
        
        # Oblicz macierz korelacji
        corr_matrix = self.df[numeric_cols].corr()
        
        # Znajdź najsilniejsze korelacje (bez autokorelacji)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
        
        # Silne korelacje (> 0.7)
        strong_corr = corr_df[corr_df['abs_corr'] > 0.7]
        
        logger.info(f"\n  Liczba par cech: {len(corr_df)}")
        logger.info(f"  Silne korelacje (|r| > 0.7): {len(strong_corr)}")
        
        if len(strong_corr) > 0:
            logger.info("  Top silne korelacje:")
            for idx, row in strong_corr.head(10).iterrows():
                logger.info(f"    {row['col1']} <-> {row['col2']}: {row['correlation']:.3f}")
        
        # Wizualizacja macierzy korelacji
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Macierz Korelacji - Wszystkie Cechy Numeryczne')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Zapisano: correlation_matrix.png")
        
        self.analysis_report['correlations'] = {
            'total_pairs': len(corr_df),
            'strong_correlations': len(strong_corr),
            'top_correlations': corr_df.head(10)[['col1', 'col2', 'correlation']].to_dict('records')
        }
    
    def analyze_distributions(self):
        """Analiza rozkładów zmiennych"""
        logger.info("\n" + "-" * 60)
        logger.info("7. ROZKŁADY ZMIENNYCH")
        logger.info("-" * 60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if 'Is_highly_rated' in self.df.columns:
            # Rozkład zmiennej docelowej
            target_counts = self.df['Is_highly_rated'].value_counts()
            target_pct = self.df['Is_highly_rated'].value_counts(normalize=True) * 100
            
            logger.info("\n  Rozkład zmiennej docelowej (Is_highly_rated):")
            for val in sorted(target_counts.index):
                logger.info(f"    {val}: {target_counts[val]} ({target_pct[val]:.1f}%)")
            
            # Wizualizacja rozkładu zmiennej docelowej
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogram zmiennej docelowej
            ax = axes[0, 0]
            target_counts.plot(kind='bar', ax=ax, color=['red', 'green'])
            ax.set_title('Rozkład Zmiennej Docelowej (Is_highly_rated)')
            ax.set_xlabel('Klasa')
            ax.set_ylabel('Liczba gier')
            ax.set_xticklabels(['Niska/Średnia', 'Wysoka'], rotation=0)
            
            # Pie chart
            ax = axes[0, 1]
            ax.pie(target_counts.values, labels=['Niska/Średnia', 'Wysoka'], 
                  autopct='%1.1f%%', colors=['red', 'green'])
            ax.set_title('Procent Rozkładu Klas')
            
            # Rozkłady wybranych cech
            selected_features = list(numeric_cols)[:6]
            for idx, col in enumerate(selected_features):
                if idx >= 2:
                    break
                ax = axes[1, idx]
                self.df[col].hist(bins=30, ax=ax, edgecolor='black')
                ax.set_title(f'Rozkład: {col}')
                ax.set_xlabel('Wartość')
                ax.set_ylabel('Częstość')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / "distributions_target.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Zapisano: distributions_target.png")
        
        # Rozkład kilku wybranych cech
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        selected_cols = list(numeric_cols)[:9]
        for idx, col in enumerate(selected_cols):
            ax = axes[idx]
            # Histogram
            self.df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            # Dodaj linię średniej
            ax.axvline(self.df[col].mean(), color='red', linestyle='--', label=f'Mean: {self.df[col].mean():.2f}')
            ax.axvline(self.df[col].median(), color='green', linestyle='--', label=f'Median: {self.df[col].median():.2f}')
            ax.set_title(f'{col}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "distributions_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Zapisano: distributions_features.png")
    
    def detect_outliers(self):
        """Detekcja wartości odstających (outlierów)"""
        logger.info("\n" + "-" * 60)
        logger.info("8. WARTOŚCI ODSTAJĄCE (OUTLIERS)")
        logger.info("-" * 60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outliers_summary = {}
        total_outliers = 0
        
        logger.info("\n  Metoda IQR (Interquartile Range):")
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_pct = outlier_count / len(self.df) * 100
                outliers_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }
                total_outliers += outlier_count
                
                if outlier_pct > 1:  # Wyświetl tylko jeśli > 1%
                    logger.info(f"    {col}: {outlier_count} outlierów ({outlier_pct:.1f}%)")
                    logger.info(f"      Zakresy: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    logger.info(f"      Wartości rzeczywiste: [{self.df[col].min():.2f}, {self.df[col].max():.2f}]")
        
        if total_outliers == 0:
            logger.info("✓ Brak znaczących outlierów")
        else:
            logger.warning(f"⚠ Znaleziono {total_outliers} instancji outlierów")
        
        self.analysis_report['outliers'] = outliers_summary
    
    def check_class_balance(self, train_df=None):
        """Sprawdź balansowanie klas w zbiorze treningowym"""
        logger.info("\n" + "-" * 60)
        logger.info("9. BALANSOWANIE KLAS")
        logger.info("-" * 60)
        
        if 'Is_highly_rated' not in self.df.columns:
            logger.warning("Brak kolumny Is_highly_rated")
            return
        
        # Sprawdź w danych ogólnych
        target_counts = self.df['Is_highly_rated'].value_counts()
        target_pct = self.df['Is_highly_rated'].value_counts(normalize=True) * 100
        
        logger.info("\n  Ogólnie (cały zbiór):")
        logger.info(f"    Klasa 0: {target_counts[0]} ({target_pct[0]:.1f}%)")
        logger.info(f"    Klasa 1: {target_counts[1]} ({target_pct[1]:.1f}%)")
        
        imbalance_ratio = max(target_pct) / min(target_pct)
        logger.info(f"    Stosunek nierównowagi: {imbalance_ratio:.2f}x")
        
        # Sprawdź w zbiorze treningowym jeśli dostępny
        if train_df is not None and 'Is_highly_rated' in train_df.columns:
            train_counts = train_df['Is_highly_rated'].value_counts()
            train_pct = train_df['Is_highly_rated'].value_counts(normalize=True) * 100
            
            logger.info("\n  Zbiór treningowy:")
            logger.info(f"    Klasa 0: {train_counts.get(0, 0)} ({train_pct.get(0, 0):.1f}%)")
            logger.info(f"    Klasa 1: {train_counts.get(1, 0)} ({train_pct.get(1, 0):.1f}%)")
            
            if abs(train_pct.get(1, 0) - 50) < 5:  # W przybliżeniu 50/50
                logger.info("✓ Klasy są zbliżone do 50/50 w zbiorze treningowym")
            else:
                logger.warning(f"⚠ Klasy NIE są 50/50 (różnica: {abs(train_pct.get(1, 0) - 50):.1f}%)")
        
        self.analysis_report['class_balance'] = {
            'class_0': target_counts[0],
            'class_1': target_counts[1],
            'class_0_pct': target_pct[0],
            'class_1_pct': target_pct[1],
            'imbalance_ratio': imbalance_ratio
        }
    
    def run(self):
        """Uruchom całą analizę"""
        self.load_latest_data()
        
        self.check_duplicates()
        self.check_missing_data()
        self.check_numeric_encoding()
        self.check_standardization()
        train_df, val_df, test_df = self.check_data_split()
        self.analyze_correlations()
        self.analyze_distributions()
        self.detect_outliers()
        self.check_class_balance(train_df)
        
        # Zapisz raport
        report_file = self.reports_dir / "comprehensive_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ANALIZA ZAKOŃCZONA")
        logger.info(f"Raport zapisany: {report_file.name}")
        logger.info("=" * 80)


def main():
    analyzer = ComprehensiveAnalysis()
    analyzer.run()


if __name__ == "__main__":
    main()
