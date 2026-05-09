# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import json # Do pracy z formatem JSON
import logging # Do logowania informacji
import sys, io # Do obsługi strumieni wejścia/wyjścia

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do walidacji danych
class DataValidator:
    # Inicjalizacja obiektu DataValidator
    def __init__(self):
        self.df = None # Ramka danych, która będzie walidowana
        self.validation_report = {} # Słownik do przechowywania wyników walidacji
    
    # Metoda do ładowania danych po inżynierii cech
    def load_data(self):
        # Ustalenie ścieżki do katalogu 'data'
        data_dir = Path(__file__).parent / "data"
        # Definiowanie ścieżki do pliku z danymi po inżynierii cech
        input_file = data_dir / "games_engineered.csv"
        # Wczytanie danych do DataFrame
        self.df = pd.read_csv(input_file, index_col=False)
        logger.info(f"Zaladowano dane do walidacji: {self.df.shape}")
        return self.df
    
    # Metoda do sprawdzania brakujących wartości
    def check_missing_values(self):
        logger.info("Sprawdzanie brakujacych wartosci...")
        missing = self.df.isna().sum() # Zliczenie brakujących wartości dla każdej kolumny
        missing_pct = (missing / len(self.df) * 100) # Procent brakujących wartości dla każdej kolumny
        issues = missing[missing > 0].to_dict() # Kolumny z brakującymi wartościami
        # Zapisanie informacji o brakujących wartościach do raportu walidacji
        self.validation_report['missing_values'] = {
            'count': int(missing.sum()),
            'columns_affected': len(issues),
            'details': {col: {'count': int(count), 'percentage': float(missing_pct[col])} for col, count in issues.items()}
        }
        # Logowanie ostrzeżeń, jeśli znaleziono brakujące wartości
        if issues:
            logger.warning(f"Znaleziono brakujace wartosci: {missing.sum()} komorek")
            for col, count in sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.warning(f"    {col}: {count} ({missing_pct[col]:.1f}%)")
        else:
            logger.info("OK Brak brakujacych wartosci")
        return self.df
    
    # Metoda do sprawdzania duplikatów
    def check_duplicates(self):
        logger.info("Sprawdzanie duplikatow...")
        dup_count = self.df['AppID'].duplicated().sum() # Zliczenie duplikatów na podstawie 'AppID'
        total_dup_rows = len(self.df[self.df.duplicated(keep=False)]) # Całkowita liczba wierszy będących duplikatami
        # Zapisanie informacji o duplikatach do raportu walidacji
        self.validation_report['duplicates'] = {'appid_duplicates': int(dup_count), 'total_duplicate_rows': int(total_dup_rows)}
        # Logowanie ostrzeżeń, jeśli znaleziono duplikaty
        if dup_count > 0:
            logger.warning(f"Znaleziono {dup_count} duplikujacych AppID")
        else:
            logger.info("OK Brak duplikatow AppID")
        return self.df
    
    # Metoda do sprawdzania typów danych
    def check_data_types(self):
        logger.info("Sprawdzanie typow danych...")
        # Podsumowanie liczby kolumn dla różnych typów danych
        dtype_summary = {
            'numeric': len(self.df.select_dtypes(include=[np.number]).columns),
            'object': len(self.df.select_dtypes(include=['object']).columns),
            'datetime': len(self.df.select_dtypes(include=['datetime64']).columns),
            'boolean': len(self.df.select_dtypes(include=['bool']).columns)
        }
        # Zapisanie podsumowania typów danych do raportu walidacji
        self.validation_report['data_types'] = dtype_summary
        logger.info(f"  Kolumny numeryczne: {dtype_summary['numeric']}")
        logger.info(f"  Kolumny tekstowe: {dtype_summary['object']}")
        logger.info(f"  Kolumny datetime: {dtype_summary['datetime']}")
        logger.info(f"  Kolumny boolean: {dtype_summary['boolean']}")
        return self.df
    
    # Metoda do detekcji wartości odstających (outlierów) metodą IQR
    def detect_outliers(self):
        logger.info("Detekcja outliers (IQR method)...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns # Wybór kolumn numerycznych
        outliers_info = {} # Słownik do przechowywania informacji o outlierach
        # Iteracja przez kolumny numeryczne
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25) # Pierwszy kwartyl
            Q3 = self.df[col].quantile(0.75) # Trzeci kwartyl
            IQR = Q3 - Q1 # Rozstęp międzykwartylowy
            lower_bound = Q1 - 1.5 * IQR # Dolna granica dla outlierów
            upper_bound = Q3 + 1.5 * IQR # Górna granica dla outlierów
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum() # Zliczenie outlierów
            # Zapisanie informacji o outlierach, jeśli istnieją
            if outliers > 0:
                outliers_info[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(self.df) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        # Zapisanie informacji o outlierach do raportu walidacji
        self.validation_report['outliers'] = outliers_info
        # Logowanie ostrzeżeń, jeśli znaleziono outliery
        if outliers_info:
            logger.warning(f"Znaleziono outliers w {len(outliers_info)} kolumnach")
            for col, info in sorted(outliers_info.items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]:
                logger.warning(f"    {col}: {info['count']} ({info['percentage']:.1f}%)")
        else:
            logger.info("OK Brak znaczacych outliers")
        return self.df
    
    # Metoda do sprawdzania zakresów wartości numerycznych
    def check_numeric_ranges(self):
        logger.info("Sprawdzanie zakresow wartosci numerycznych...")
        range_checks = {} # Słownik do przechowywania wyników sprawdzenia zakresów
        # Sprawdzenie zakresu dla kolumny 'Price'
        if 'Price' in self.df.columns:
            invalid_prices = (self.df['Price'] < 0).sum()
            range_checks['Price'] = {
                'check': 'Price >= 0',
                'invalid_count': int(invalid_prices),
                'min': float(self.df['Price'].min()),
                'max': float(self.df['Price'].max())
            }
        # Sprawdzenie zakresu dla kolumny 'Metacritic score'
        if 'Metacritic score' in self.df.columns:
            invalid_scores = ((self.df['Metacritic score'] < 0) | (self.df['Metacritic score'] > 100)).sum()
            range_checks['Metacritic score'] = {
                'check': '0 <= score <= 100',
                'invalid_count': int(invalid_scores),
                'min': float(self.df['Metacritic score'].min()),
                'max': float(self.df['Metacritic score'].max())
            }
        # Zapisanie wyników sprawdzenia zakresów do raportu walidacji
        self.validation_report['range_checks'] = range_checks
        # Logowanie ostrzeżeń, jeśli znaleziono anomalie zakresu
        if range_checks:
            logger.warning(f"Znaleziono {len(range_checks)} anomalii zakresu")
        else:
            logger.info("OK Zakresy wartosci poprawne")
        return self.df
    
    # Metoda do analizy rozkładu danych dla kluczowych kolumn
    def check_distribution(self):
        logger.info("Analiza rozkladu danych...")
        key_columns = ['Price', 'Metacritic score', 'User score', 'Positive', 'Negative', 'Days_since_release'] # Kluczowe kolumny do analizy
        distribution_info = {} # Słownik do przechowywania informacji o rozkładzie
        # Iteracja przez kluczowe kolumny
        for col in key_columns:
            if col in self.df.columns:
                # Obliczenie statystyk opisowych dla kolumny
                stats = {
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'skewness': float(self.df[col].skew()), # Skośność rozkładu
                    'kurtosis': float(self.df[col].kurtosis()) # Kurtoza rozkładu
                }
                distribution_info[col] = stats
                # Logowanie ostrzeżeń, jeśli skośność jest wysoka
                if abs(stats['skewness']) > 2:
                    logger.warning(f"Kolumna {col} ma wysoki skos: {stats['skewness']:.2f}")
        # Zapisanie informacji o rozkładzie do raportu walidacji
        self.validation_report['distribution'] = distribution_info
        return self.df
    
    # Metoda do sprawdzania pokrycia kategorii w zmiennych kategorycznych
    def check_categorical_coverage(self):
        logger.info("Sprawdzanie kategorii...")
        categorical_cols = ['Metacritic_category', 'Price_category'] # Kolumny kategoryczne do sprawdzenia
        category_info = {} # Słownik do przechowywania informacji o kategoriach
        # Iteracja przez kolumny kategoryczne
        for col in categorical_cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts().to_dict() # Liczność wystąpień każdej kategorii
                total = sum(value_counts.values()) # Całkowita liczba wartości
                # Zapisanie informacji o pokryciu kategorii
                category_info[col] = {
                    'unique_values': len(value_counts),
                    'coverage': {k: {'count': int(v), 'percentage': float(v/total*100)} for k, v in value_counts.items()}
                }
        # Zapisanie informacji o kategoriach do raportu walidacji
        self.validation_report['categories'] = category_info
        logger.info("OK Kategorie sprawdzane")
        return self.df
    
    # Metoda do generowania ogólnego wyniku jakości danych
    def generate_quality_score(self):
        logger.info("\nObliczanie scoru jakosci...")
        score = 100.0 # Początkowy wynik jakości
        issues = [] # Lista wykrytych problemów
        # Obliczenie procentu brakujących wartości w całym zbiorze
        missing_pct = (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        # Odjęcie punktów za brakujące wartości
        if missing_pct > 0:
            score -= min(10, missing_pct)
            issues.append(f"Brakujace wartosci: {missing_pct:.1f}%")
        # Odjęcie punktów za duplikaty AppID
        if self.validation_report.get('duplicates', {}).get('appid_duplicates', 0) > 0:
            score -= 5
            issues.append("Znaleziono duplikaty AppID")
        # Odjęcie punktów za outliery
        if self.validation_report.get('outliers', {}):
            outlier_pct = sum(v['percentage'] for v in self.validation_report['outliers'].values())
            score -= min(10, outlier_pct / 10)
            issues.append(f"Outliers w danych: {outlier_pct:.1f}%")
        # Odjęcie punktów za anomalie zakresu
        if self.validation_report.get('range_checks', {}):
            invalid_ranges = len(self.validation_report['range_checks'])
            score -= min(5, invalid_ranges)
            issues.append(f"Anomalie zakresu w {invalid_ranges} kolumnach")
        # Zapisanie wyniku jakości do raportu walidacji
        self.validation_report['quality_score'] = {
            'score': float(max(0, score)), # Wynik nie może być ujemny
            'max_score': 100.0,
            'issues': issues
        }
        logger.info(f"Score jakosci: {score:.1f}/100")
        # Logowanie szczegółów problemów
        if issues:
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("  OK Brak znaczacych problemow z jakoscia")
        return score
    
    # Metoda do zapisywania raportu walidacji
    def save_validation_report(self):
        # Ustalenie ścieżki do katalogu 'reports' i utworzenie go, jeśli nie istnieje
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)
        # Definiowanie nazwy pliku raportu
        report_file = report_dir / "02_validation_report.json"
        # Zapisanie raportu walidacji do pliku JSON
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
        logger.info(f"\nOK Raport walidacji zapisany: reports/02_validation_report.json")
        logger.info("\nSTRESZCZENIE WALIDACJI:")
        logger.info("-" * 50)
        logger.info(f"Calkowite wiersze: {len(self.df)}")
        logger.info(f"Calkowite kolumny: {len(self.df.columns)}")
        logger.info(f"Brakujace wartosci: {self.df.isna().sum().sum()}")
        logger.info(f"Duplikaty AppID: {self.validation_report['duplicates']['appid_duplicates']}")
        logger.info(f"Quality Score: {self.validation_report['quality_score']['score']:.1f}/{self.validation_report['quality_score']['max_score']}")
    
    # Metoda uruchamiająca cały proces walidacji danych
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("WALIDACJA DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data() # Załadowanie danych
        self.check_missing_values() # Sprawdzenie brakujących wartości
        self.check_duplicates() # Sprawdzenie duplikatów
        self.check_data_types() # Sprawdzenie typów danych
        self.detect_outliers() # Detekcja outlierów
        self.check_numeric_ranges() # Sprawdzenie zakresów wartości numerycznych
        self.check_distribution() # Analiza rozkładu danych
        self.check_categorical_coverage() # Sprawdzenie pokrycia kategorii
        self.generate_quality_score() # Generowanie wyniku jakości danych
        self.save_validation_report() # Zapisanie raportu walidacji
        logger.info("\n" + "=" * 80)
        logger.info("OK WALIDACJA UKONCZNA")
        logger.info("=" * 80 + "\n")
        return self.df

# Główna funkcja programu
def main():
    validator = DataValidator() # Utworzenie instancji klasy DataValidator
    validator.run() # Uruchomienie procesu walidacji

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    main()
