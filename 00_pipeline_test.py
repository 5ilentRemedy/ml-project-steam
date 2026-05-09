"""
00_full_ml_pipeline.py
Główny orchestrator pełnego pipleline'u Machine Learning

Ten skrypt uruchamia wszystkie kroki w sekwencji:
1. Eksploracja danych
2. Czyszczenie danych
3. Inżynieria cech
4. Walidacja
5. Export
6. Trenowanie modeli ML
7. Ewaluacja modeli i wizualizacja
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
import io
import re
import importlib
from importlib import metadata

# UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FullMLPipeline:
    """Orchestrator calego pipleline'u ML"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        
        
        self.scripts = [
            ('02_data_exploration.py', 'Eksploracja danych'),
            ('03_data_cleaning.py', 'Czyszczenie danych'),
            ('04_feature_engineering.py', 'Inżynieria cech'),
            ('05_data_validation.py', 'Walidacja danych'),
            ('06_data_export.py', 'Export i przygotowanie'),
            ('07_model_training.py', 'Trenowanie modeli ML'),
            ('08_model_evaluation.py', 'Ewaluacja i wizualizacja')
        ]
        self.results = {}
    
    def run_script(self, script_name, description):
        """Uruchamia pojedynczy skrypt"""
        logger.info("\n" + "=" * 80)
        logger.info(f"KROK: {description}")
        logger.info(f"Skrypt: {script_name}")
        logger.info("=" * 80)
        
        script_path = self.project_dir / script_name
        
        if not script_path.exists():
            logger.error(f"[ERROR] Skrypt nie znaleziony: {script_path}")
            return False
        
        try:
            start_time = time.time()
            
            # Uruchom skrypt
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                cwd=str(self.project_dir)
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"[OK] {description} - SUKCES ({elapsed_time:.1f}s)")
                self.results[script_name] = {
                    'status': 'SUCCESS',
                    'time': elapsed_time
                }
                return True
            else:
                logger.error(f"[ERROR] {description} - BLĄD (kod: {result.returncode})")
                self.results[script_name] = {
                    'status': 'FAILED',
                    'time': elapsed_time
                }
                return False
        
        except Exception as e:
            logger.error(f"[ERROR] {description} - WYJĄTEK: {str(e)}")
            self.results[script_name] = {
                'status': 'ERROR',
                'time': 0,
                'error': str(e)
            }
            return False
    
    def verify_outputs(self):
        """Weryfikuje czy wszystkie pliki wyjsciowe zostały utworzone"""
        logger.info("\n" + "=" * 80)
        logger.info("WERYFIKACJA PLIKÓW WYJŚCIOWYCH")
        logger.info("=" * 80)
        
        
        expected_files = {
            'data/games_cleaned.csv': 'Oczyszczone dane',
            'data/games_engineered.csv': 'Dane z cechami',
            'data/processed/games_final.csv': 'Finalne dane (CSV)',
            'data/processed/games_train.csv': 'Zbiór treningowy',
            'data/processed/games_test.csv': 'Zbiór testowy',
            'reports/02_validation_report.json': 'Raport walidacji danych',
            'reports/03_export_summary.json': 'Raport exportu',
            'models/best_model.joblib': 'Zapisany model ML',
            'reports/04_model_training_report.json': 'Raport z treningu ML',
            'reports/05_evaluation_metrics.json': 'Metryki ewaluacji ML',
            'reports/figures/confusion_matrix.png': 'Wykres: Macierz pomyłek',
            'reports/figures/roc_pr_curves.png': 'Wykres: Krzywe ROC i PR'
        }
        
        verification = {}
        
        for file_path, description in expected_files.items():
            full_path = self.project_dir / file_path
            exists = full_path.exists()
            status = "[OK]" if exists else "[MISS]"
            
            logger.info(f"{status} {description:30} - {file_path}")
            verification[file_path] = exists
        
        return all(verification.values())
    
    def print_summary(self):
        """Drukuje podsumowanie wykonania"""
        logger.info("\n" + "=" * 80)
        logger.info("PODSUMOWANIE PIPLELINE'U")
        logger.info("=" * 80 + "\n")
        
        total_time = sum(r.get('time', 0) for r in self.results.values())
        
        # Tabela wyników
        logger.info(f"{'Skrypt':<30} {'Status':<10} {'Czas (s)':<10}")
        logger.info("-" * 50)
        
        for script, result in self.results.items():
            status = result['status']
            time_val = result.get('time', 0)
            logger.info(f"{script:<30} {status:<10} {time_val:<10.1f}")
        
        logger.info("-" * 50)
        logger.info(f"{'RAZEM':<30} {'':<10} {total_time:<10.1f}s")
        
        # Status ogólny
        all_success = all(r['status'] == 'SUCCESS' for r in self.results.values())
        
        if all_success:
            logger.info("\n[OK] PEŁNY PIPELINE ML ZAKOŃCZONY POMYŚLNIE")
        else:
            failed_count = sum(1 for r in self.results.values() if r['status'] != 'SUCCESS')
            logger.warning(f"\n[!] {failed_count} kroki nie powiodły się")
        
        logger.info("\nKatalogi wyjściowe:")
        logger.info("- Dane: data/processed/")
        logger.info("- Modele: models/")
        logger.info("- Raporty i wykresy: reports/figures/")
        logger.info("\n" + "=" * 80 + "\n")
    
    def run(self):
        """Uruchamia cały pipleline"""
        logger.info("\n" + "=" * 80)
        logger.info("STEAM GAMES - PEŁNY MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Utworzenie wszystkich wymaganych katalogów
        (self.project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models").mkdir(parents=True, exist_ok=True)
        
        logger.info("\nUstawienia:")
        logger.info(f"  Katalog projektu: {self.project_dir}")
        logger.info(f"  Liczba kroków: {len(self.scripts)}")
        
        # Uruchomienie wszystkich kroków
        all_success = True
        for script_name, description in self.scripts:
            success = self.run_script(script_name, description)
            if not success:
                all_success = False
                logger.warning(f"[!] Pipleline zatrzymany z powodu błędu w: {description}")
                break
        
        # Weryfikacja plików wyjściowych
        if all_success:
            all_files_exist = self.verify_outputs()
            if not all_files_exist:
                logger.warning("[!] Niektóre pliki wyjściowe nie zostały utworzone")
                all_success = False
        
        # Podsumowanie
        self.print_summary()
        
        return all_success

def main():
    """Główna funkcja"""
    try:
        # Moduł sprawdzania zależności z requirements.txt
        def check_and_install_requirements():
            req_file = Path(__file__).parent / 'requirements.txt'
            if not req_file.exists():
                logger.info('Brak requirements.txt — pomijam sprawdzanie zależności.')
                return

            logger.info('Sprawdzam wymagane zależności z requirements.txt...')
            lines = [l.strip() for l in req_file.read_text(encoding='utf-8').splitlines()]
            pkg_lines = [l for l in lines if l and not l.startswith('#')]

            missing = []
            mismatch = []

            for line in pkg_lines:
                m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
                if not m:
                    continue
                pkg_name = m.group(1)

                mod_candidates = [pkg_name.replace('-', '_')]
                if pkg_name.lower() == 'scikit-learn':
                    mod_candidates.insert(0, 'sklearn')

                found = False
                for mod in mod_candidates:
                    try:
                        importlib.import_module(mod)
                        found = True
                        break
                    except Exception:
                        continue

                if not found:
                    missing.append(pkg_name)
                    continue

                try:
                    installed_version = metadata.version(pkg_name)
                except metadata.PackageNotFoundError:
                    try:
                        installed_version = metadata.version(mod)
                    except Exception:
                        installed_version = None

                spec_match = re.search(r"([<>=!~].+)$", line)
                if installed_version and spec_match:
                    spec = spec_match.group(1)
                    try:
                        from packaging.specifiers import SpecifierSet
                        ss = SpecifierSet(spec)
                        if not ss.contains(installed_version):
                            mismatch.append((pkg_name, installed_version, spec))
                    except Exception:
                        pass

            if not missing and not mismatch:
                logger.info('Wszystkie zależności wydają się być spełnione.')
                return

            if missing:
                logger.warning(f'Brakuje pakietów: {missing}')
            if mismatch:
                logger.warning(f'Pakiety z niezgodną wersją: {mismatch}')

            resp = input('Zainstalować brakujące/niezgodne pakiety z requirements.txt? [y/N]: ').strip().lower()
            if resp != 'y':
                logger.info('Pominięto instalację pakietów.')
                return

            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)]
            logger.info(f'Uruchamiam: {cmd}')
            subprocess.check_call(cmd)
            logger.info('Instalacja zakończona. Kontynuuję.')

        check_and_install_requirements()

        pipeline = FullMLPipeline()
        success = pipeline.run()
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.warning("\n[!] Pipleline przerwany przez użytkownika")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"[ERROR] Niespodziewany błąd: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()