import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys, io

if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.df = None
        self.validation_report = {}
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        input_file = data_dir / "games_engineered.csv"
        self.df = pd.read_csv(input_file, index_col=False)
        logger.info(f"Zaladowano dane do walidacji: {self.df.shape}")
        return self.df
    
    def check_missing_values(self):
        logger.info("Sprawdzanie brakujacych wartosci...")
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df) * 100)
        issues = missing[missing > 0].to_dict()
        self.validation_report['missing_values'] = {
            'count': int(missing.sum()),
            'columns_affected': len(issues),
            'details': {col: {'count': int(count), 'percentage': float(missing_pct[col])} for col, count in issues.items()}
        }
        if issues:
            logger.warning(f"[!] Znaleziono brakujace wartosci: {missing.sum()} komorek")
            for col, count in sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.warning(f"    {col}: {count} ({missing_pct[col]:.1f}%)")
        else:
            logger.info("[OK] Brak brakujacych wartosci")
        return self.df
    
    def check_duplicates(self):
        logger.info("Sprawdzanie duplikatow...")
        dup_count = self.df['AppID'].duplicated().sum()
        total_dup_rows = len(self.df[self.df.duplicated(keep=False)])
        self.validation_report['duplicates'] = {'appid_duplicates': int(dup_count), 'total_duplicate_rows': int(total_dup_rows)}
        if dup_count > 0:
            logger.warning(f"[!] Znaleziono {dup_count} duplikujacych AppID")
        else:
            logger.info("[OK] Brak duplikatow AppID")
        return self.df
    
    def check_data_types(self):
        logger.info("Sprawdzanie typow danych...")
        dtype_summary = {
            'numeric': len(self.df.select_dtypes(include=[np.number]).columns),
            'object': len(self.df.select_dtypes(include=['object']).columns),
            'datetime': len(self.df.select_dtypes(include=['datetime64']).columns),
            'boolean': len(self.df.select_dtypes(include=['bool']).columns)
        }
        self.validation_report['data_types'] = dtype_summary
        logger.info(f"  Kolumny numeryczne: {dtype_summary['numeric']}")
        logger.info(f"  Kolumny tekstowe: {dtype_summary['object']}")
        logger.info(f"  Kolumny datetime: {dtype_summary['datetime']}")
        logger.info(f"  Kolumny boolean: {dtype_summary['boolean']}")
        return self.df
    
    def detect_outliers(self):
        logger.info("Detekcja outliers (IQR method)...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_info[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(self.df) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        self.validation_report['outliers'] = outliers_info
        if outliers_info:
            logger.warning(f"[!] Znaleziono outliers w {len(outliers_info)} kolumnach")
            for col, info in sorted(outliers_info.items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]:
                logger.warning(f"    {col}: {info['count']} ({info['percentage']:.1f}%)")
        else:
            logger.info("[OK] Brak znaczacych outliers")
        return self.df
    
    def check_numeric_ranges(self):
        logger.info("Sprawdzanie zakresow wartosci numerycznych...")
        range_checks = {}
        if 'Price' in self.df.columns:
            invalid_prices = (self.df['Price'] < 0).sum()
            range_checks['Price'] = {
                'check': 'Price >= 0',
                'invalid_count': int(invalid_prices),
                'min': float(self.df['Price'].min()),
                'max': float(self.df['Price'].max())
            }
        if 'Metacritic score' in self.df.columns:
            invalid_scores = ((self.df['Metacritic score'] < 0) | (self.df['Metacritic score'] > 100)).sum()
            range_checks['Metacritic score'] = {
                'check': '0 <= score <= 100',
                'invalid_count': int(invalid_scores),
                'min': float(self.df['Metacritic score'].min()),
                'max': float(self.df['Metacritic score'].max())
            }
        self.validation_report['range_checks'] = range_checks
        if range_checks:
            logger.warning(f"[!] Znaleziono {len(range_checks)} anomalii zakresu")
        else:
            logger.info("[OK] Zakresy wartosci poprawne")
        return self.df
    
    def check_distribution(self):
        logger.info("Analiza rozkladu danych...")
        key_columns = ['Price', 'Metacritic score', 'User score', 'Positive', 'Negative', 'Days_since_release']
        distribution_info = {}
        for col in key_columns:
            if col in self.df.columns:
                stats = {
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'skewness': float(self.df[col].skew()),
                    'kurtosis': float(self.df[col].kurtosis())
                }
                distribution_info[col] = stats
                if abs(stats['skewness']) > 2:
                    logger.warning(f"[!] {col} ma wysoki skos: {stats['skewness']:.2f}")
        self.validation_report['distribution'] = distribution_info
        return self.df
    
    def check_categorical_coverage(self):
        logger.info("Sprawdzanie kategorii...")
        categorical_cols = ['Metacritic_category', 'Price_category']
        category_info = {}
        for col in categorical_cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts().to_dict()
                total = sum(value_counts.values())
                category_info[col] = {
                    'unique_values': len(value_counts),
                    'coverage': {k: {'count': int(v), 'percentage': float(v/total*100)} for k, v in value_counts.items()}
                }
        self.validation_report['categories'] = category_info
        logger.info("[OK] Kategorie sprawdzane")
        return self.df
    
    def generate_quality_score(self):
        logger.info("\nObliczanie scoru jakosci...")
        score = 100.0
        issues = []
        missing_pct = (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        if missing_pct > 0:
            score -= min(10, missing_pct)
            issues.append(f"Brakujace wartosci: {missing_pct:.1f}%")
        if self.validation_report.get('duplicates', {}).get('appid_duplicates', 0) > 0:
            score -= 5
            issues.append("Znaleziono duplikaty AppID")
        if self.validation_report.get('outliers', {}):
            outlier_pct = sum(v['percentage'] for v in self.validation_report['outliers'].values())
            score -= min(10, outlier_pct / 10)
            issues.append(f"Outliers w danych: {outlier_pct:.1f}%")
        if self.validation_report.get('range_checks', {}):
            invalid_ranges = len(self.validation_report['range_checks'])
            score -= min(5, invalid_ranges)
            issues.append(f"Anomalie zakresu w {invalid_ranges} kolumnach")
        self.validation_report['quality_score'] = {
            'score': float(max(0, score)),
            'max_score': 100.0,
            'issues': issues
        }
        logger.info(f"Score jakosci: {score:.1f}/100")
        if issues:
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("  [OK] Brak znaczacych problemow z jakoscia")
        return score
    
    def save_validation_report(self):
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / "02_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
        logger.info(f"\n[OK] Raport walidacji zapisany: reports/02_validation_report.json")
        logger.info("\nSTRESZCZENIE WALIDACJI:")
        logger.info("-" * 50)
        logger.info(f"Calkowite wiersze: {len(self.df)}")
        logger.info(f"Calkowite kolumny: {len(self.df.columns)}")
        logger.info(f"Brakujace wartosci: {self.df.isna().sum().sum()}")
        logger.info(f"Duplikaty AppID: {self.validation_report['duplicates']['appid_duplicates']}")
        logger.info(f"Quality Score: {self.validation_report['quality_score']['score']:.1f}/{self.validation_report['quality_score']['max_score']}")
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("WALIDACJA DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()
        self.detect_outliers()
        self.check_numeric_ranges()
        self.check_distribution()
        self.check_categorical_coverage()
        self.generate_quality_score()
        self.save_validation_report()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] WALIDACJA UKONCZNA")
        logger.info("=" * 80 + "\n")
        return self.df

def main():
    validator = DataValidator()
    validator.run()

if __name__ == "__main__":
    main()
