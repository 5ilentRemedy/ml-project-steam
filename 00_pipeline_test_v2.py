# ============================================================================
# PEŁNY PIPELINE ML - WERSJA 2.0 Z ZAAWANSOWANĄ ANALIZĄ
# Orchestrator dla całej analizy i trenowania modeli
# ============================================================================

import subprocess
import sys
import time
from pathlib import Path
import logging
import io

# Ustawienie kodowania UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution_v2.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FullMLPipelineV2:
    """Pełny pipeline ML - Wersja 2.0 z zaawansowaną analizą"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        
        # Definicja kroków pipeline'u
        self.scripts = [
            ('02_data_exploration.py', 'Eksploracja danych'),
            ('03_data_cleaning.py', 'Czyszczenie danych'),
            ('04_feature_engineering.py', 'Inżynieria cech'),
            ('05_data_validation.py', 'Walidacja danych'),
            ('06_data_export.py', 'Export i przygotowanie do ML'),
            ('06a_data_validator.py', 'Dodatkowa walidacja'),
            ('02a_comprehensive_analysis.py', '[NOWE] Zaawansowana analiza danych'),
            ('07a_model_training_advanced.py', '[NOWE] Trenowanie zaawansowanych modeli'),
            ('08a_model_evaluation_advanced.py', '[NOWE] Zaawansowana ewaluacja modeli'),
        ]
        
        self.results = {}
    
    def run_script(self, script_name, description):
        """Uruchom pojedynczy skrypt"""
        logger.info("\n" + "=" * 80)
        logger.info(f"KROK: {description}")
        logger.info(f"Skrypt: {script_name}")
        logger.info("=" * 80)
        
        script_path = self.project_dir / script_name
        
        if not script_path.exists():
            logger.error(f"✗ Skrypt nie znaleziony: {script_path}")
            self.results[script_name] = 'ERROR: NOT FOUND'
            return False
        
        try:
            start_time = time.time()
            
            # Uruchom skrypt
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            elapsed_time = time.time() - start_time
            
            # Wyświetl output
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                logger.info(f"✓ SUKCES - Czas: {elapsed_time:.2f}s")
                self.results[script_name] = f'OK ({elapsed_time:.2f}s)'
                return True
            else:
                logger.error(f"✗ BŁĄD - Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output:\n{result.stderr}")
                self.results[script_name] = f'ERROR (code: {result.returncode})'
                return False
        
        except Exception as e:
            logger.error(f"✗ Wyjątek: {str(e)}")
            self.results[script_name] = f'ERROR: {str(e)}'
            return False
    
    def run(self):
        """Uruchom cały pipeline"""
        logger.info("\n" * 2)
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 78 + "║")
        logger.info("║" + "PEŁNY PIPELINE ML - WERSJA 2.0 Z ZAAWANSOWANĄ ANALIZĄ".center(78) + "║")
        logger.info("║" + " " * 78 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        
        logger.info("\nPipelineObejmuje:")
        logger.info("✓ Eksploracja i czyszczenie danych")
        logger.info("✓ Inżynieria cech")
        logger.info("✓ Zaawansowana analiza danych")
        logger.info("✓ Trenowanie modeli (Logistic Regression, Decision Tree, Random Forest,")
        logger.info("                     Neural Network, LightGBM, XGBoost)")
        logger.info("✓ Zaawansowana ewaluacja i wizualizacje")
        
        total_start_time = time.time()
        
        for script_name, description in self.scripts:
            success = self.run_script(script_name, description)
            if not success and script_name not in ['02a_comprehensive_analysis.py',
                                                    '07a_model_training_advanced.py',
                                                    '08a_model_evaluation_advanced.py']:
                # Jeśli krytyczny skrypt się nie powiódł, zatrzymaj
                logger.error(f"\n✗ PIPELINE ZATRZYMANY - Krytyczny skrypt się nie powiódł")
                break
        
        total_time = time.time() - total_start_time
        
        # Wyświetl podsumowanie
        self.print_summary(total_time)
    
    def print_summary(self, total_time):
        """Wyświetl podsumowanie"""
        logger.info("\n" + "=" * 80)
        logger.info("PODSUMOWANIE PIPELINE'U")
        logger.info("=" * 80)
        
        print("\nRESULTATY KROKÓW:")
        for i, (script_name, description) in enumerate(self.scripts, 1):
            status = self.results.get(script_name, 'UNKNOWN')
            symbol = "✓" if "OK" in status else "✗" if "ERROR" in status else "?"
            print(f"{symbol} [{i:2d}] {description:50s} - {status}")
        
        logger.info(f"\n✓ CAŁKOWITY CZAS WYKONANIA: {total_time:.2f} sekund ({total_time/60:.2f} minut)")
        logger.info("=" * 80)
        
        logger.info("\nWyniki zapisane w:")
        logger.info("  • data/processed/        - Dane do modelowania")
        logger.info("  • models/                - Zapisane modele")
        logger.info("  • reports/               - Raporty i metryki")
        logger.info("  • reports/figures/       - Wizualizacje i wykresy")
        logger.info("\n" + "=" * 80 + "\n")


def main():
    pipeline = FullMLPipelineV2()
    pipeline.run()


if __name__ == "__main__":
    main()
