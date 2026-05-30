"""
05_data_validation.py
Statystyczna walidacja danych jakościowych po procesie czyszczenia.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
import io

# UTF-8 encoding dla konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.cleaned_file = (self.project_dir / "data" / "games_cleaned").with_suffix('.csv')
        self.reports_dir = self.project_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        """Ładuje oczyszczony zbiór danych"""
        if not self.cleaned_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {self.cleaned_file}. Uruchom najpierw czyszczenie.")
        self.df = pd.read_csv(self.cleaned_file)
        logger.info(f"[OK] Wczytano dane do walidacji statystycznej: {self.df.shape}")

    def run_statistical_checks(self):
        """Przeprowadza testy statystyczne i sprawdza spójność danych"""
        logger.info("Uruchamianie testów integralności i dystrybucji danych...")
        
        # 🔥 POPRAWKA: Dynamiczne dopasowanie wielkości liter w nazwach kolumn
        reviews_col = 'Total_Reviews' if 'Total_Reviews' in self.df.columns else 'Total_reviews'
        ratio_col = 'Review_ratio' if 'Review_ratio' in self.df.columns else 'Review_ratio'
        
        report = {
            "total_records": int(len(self.df)),
            "missing_values": self.df.isna().sum().to_dict(),
            "target_distribution": self.df['Market_Success'].value_counts().to_dict(),
            "basic_stats": {
                "price": {
                    "mean": float(self.df['Price'].mean()),
                    "max": float(self.df['Price'].max()),
                    "min": float(self.df['Price'].min())
                }
            }
        }

        # Dodanie statystyk recenzji, jeśli kolumny istnieją
        if reviews_col in self.df.columns:
            q1 = self.df[reviews_col].quantile(0.25)
            q3 = self.df[reviews_col].quantile(0.75)
            iqr = q3 - q1
            report["basic_stats"]["reviews"] = {
                "median": float(self.df[reviews_col].median()),
                "mean": float(self.df[reviews_col].mean()),
                "iqr_bounds": [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)]
            }
            
        if ratio_col in self.df.columns:
            report["basic_stats"]["review_ratio"] = {
                "mean": float(self.df[ratio_col].dropna().mean()),
                "skewness": float(self.df[ratio_col].dropna().skew())
            }

        # Zapis raportu walidacyjnego do JSON
        output_path = self.reports_dir / "02_validation_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"[OK] Statystyki zweryfikowane. Raport zapisany w: reports/02_validation_report.json")

    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("START ETAPU WALIDACJI STATYSTYCZNEJ DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.run_statistical_checks()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] WALIDACJA STATYSTYCZNA UKOŃCZONA")
        logger.info("=" * 80 + "\n")

if __name__ == "__main__":
    validator = DataValidator()
    validator.run()