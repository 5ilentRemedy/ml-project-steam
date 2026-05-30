"""
00_pipeline_test.py
Główny orchestrator pełnego pipeline'u Machine Learning

Dostosowany do obecnej struktury plików na dysku użytkownika.
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
    """Orchestrator całego pipeline'u ML"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        
        # 🟢 Dopasowane nazwy skryptów dokładnie do struktury na Twoim dysku
        self.scripts = [
            ('02_data_exploration.py', 'Eksploracja danych i analiza szumu'),
            ('03_data_cleaning.py', 'Czyszczenie danych i etykietowanie sukcesu'),
            ('04_feature_engineering.py', 'Inżynieria cech (transformacje i standaryzacja)'),
            ('06_data_export.py', 'Eksport danych i podział na zbiory Train/Val/Test'),
            ('07_model_training.py', 'Trenowanie wieloklasowych modeli ML'),
            ('05_data_validation.py', 'Walidacja statystyczna danych jakościowych'),
            ('06a_data_validator.py', 'Generowanie zaawansowanych wykresów dystrybucji rynkowej'),
            ('08_model_evaluation.py', 'Ewaluacja końcowa klasyfikatorów i wykresy ROC/PR')
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
                logger.error(f"[ERROR] {description} - BŁĄD (kod wyjścia: {result.returncode})")
                self.results[script_name] = {
                    'status': 'FAILED',
                    'time': elapsed_time
                }
                return False
        
        except Exception as e:
            logger.error(f"[ERROR] {description} - WYJĄTEK PROCESU: {str(e)}")
            self.results[script_name] = {
                'status': 'ERROR',
                'time': 0,
                'error': str(e)
            }
            return False
    
    def verify_outputs(self):
        """Weryfikuje czy kluczowe pliki wyjściowe i raporty zostały poprawnie zapisane"""
        logger.info("\n" + "=" * 80)
        logger.info("WERYFIKACJA STRUKTURY PLIKÓW WYJŚCIOWYCH POTOKU")
        logger.info("=" * 80)
        
        expected_files = {
            'data/games_cleaned.csv': 'Baza oczyszczona z targetem',
            'data/games_engineered.csv': 'Zbiór po inżynierii cech',
            'data/processed/games_final.csv': 'Ostateczny plik cech modeli',
            'data/processed/games_train.csv': 'Zbiór treningowy (70%)',
            'data/processed/games_test.csv': 'Zbiór testowy (15%)',
            'reports/01_exploration_summary.json': 'Raport eksploracyjny',
            'reports/02_validation_report.json': 'Raport walidacji statystycznej',
            'reports/05_evaluation_metrics.json': 'Końcowe metryki ewaluacji ML',
            'models/best_model.joblib': 'Zapisany najlepszy model ML',
            'reports/figures/correlation_matrix.png': 'Wykres: Macierz korelacji',
            'reports/figures/market_success_distribution.png': 'Wykres: Rozkład klas targetu',
            'reports/figures/advanced_dependency_plots.png': 'Wykres: Panel geometrii cech',
            'reports/figures/roc_pr_curves.png': 'Wykres: Wieloklasowe krzywe ROC/PR'
        }
        
        verification = {}
        for file_path, description in expected_files.items():
            full_path = self.project_dir / file_path
            exists = full_path.exists()
            status = "[OK]" if exists else "[MISS]"
            logger.info(f"{status} {description:50} - {file_path}")
            verification[file_path] = exists
        
        return all(verification.values())
    
    def print_summary(self, all_success):
        """Prezentuje finalne podsumowanie wykonania potoku"""
        logger.info("\n" + "=" * 80)
        logger.info("ZBIORCZE PODSUMOWANIE PEŁNEGO PIPELINE'U ML")
        logger.info("=" * 80 + "\n")
        
        total_time = sum(r.get('time', 0) for r in self.results.values())
        
        logger.info(f"{'Nazwa skryptu potoku':<30} {'Status':<12} {'Czas wykonania':<10}")
        logger.info("-" * 60)
        
        for script, description in self.scripts:
            result = self.results.get(script, {'status': 'NOT RUN', 'time': 0.0})
            status = result['status']
            time_val = result['time']
            logger.info(f"{script:<30} {status:<12} {time_val:<10.1f}s")
        
        logger.info("-" * 60)
        logger.info(f"{'SUMARYCZNY CZAS PROCESU':<30} {'':<12} {total_time:<10.1f}s")
        
        if all_success:
            logger.info("\n🎉 [SUCCESS] PEŁNY POTOK MACHINE LEARNING ZAKOŃCZONY SUKCESEM!")
            logger.info("\nLokalizacja wyjściowych struktur projektowych:")
            logger.info("  -> Dane treningowe i Parquet: data/processed/")
            logger.info("  -> Pliki binarne modeli (.joblib): models/")
            logger.info("  -> Raporty JSON oraz wykresy PNG: reports/ oraz reports/figures/")
        else:
            logger.warning("\n❌ [FAILED] PIPELINE PRZERWANY Z POWODU BŁĘDU W JEDNYM Z KROKÓW.")
        logger.info("=" * 80 + "\n")
    
    def run(self):
        """Uruchamia sekwencyjnie cały potok"""
        logger.info("\n" + "=" * 80)
        logger.info("STEAM GAMES - AUTOMATYCZNY MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data startu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Inicjalizacja katalogów projektowych
        (self.project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models").mkdir(parents=True, exist_ok=True)
        
        logger.info("\n[INFO] Konfiguracja środowiska:")
        logger.info(f"  Ścieżka projektu: {self.project_dir}")
        logger.info(f"  Liczba kroków sekwencyjnych: {len(self.scripts)}")
        
        all_success = True
        for script_name, description in self.scripts:
            success = self.run_script(script_name, description)
            if not success:
                all_success = False
                logger.warning(f"\n[!] Pipeline przerwany automatycznie na kroku: {description}")
                break
        
        if all_success:
            self.verify_outputs()
            
        self.print_summary(all_success)
        return all_success

def main():
    try:
        def check_and_install_requirements():
            req_file = Path(__file__).parent / 'requirements.txt'
            if not req_file.exists():
                logger.info('Brak requirements.txt — pomijam automatyczne sprawdzanie bibliotek.')
                return

            logger.info('Weryfikacja wymaganych zależności pakietów z requirements.txt...')
            lines = [l.strip() for l in req_file.read_text(encoding='utf-8').splitlines()]
            pkg_lines = [l for l in lines if l and not l.startswith('#')]

            missing = []
            mismatch = []

            for line in pkg_lines:
                m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
                if not m: continue
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
                    try: installed_version = metadata.version(mod)
                    except Exception: installed_version = None

                spec_match = re.search(r"([<>=!~].+)$", line)
                if installed_version and spec_match:
                    spec = spec_match.group(1)
                    try:
                        from packaging.specifiers import SpecifierSet
                        ss = SpecifierSet(spec)
                        if not ss.contains(installed_version):
                            mismatch.append((pkg_name, installed_version, spec))
                    except Exception: pass

            if not missing and not mismatch:
                logger.info('[OK] Wszystkie zależności środowiska Python są spełnione.')
                return

            if missing: logger.warning(f'Wykryto brakujące pakiety systemowe: {missing}')
            if mismatch: logger.warning(f'Wykryto pakiety o niezgodnych wersjach specyfikacji: {mismatch}')

            resp = input('Zainstalować/zaktualizować pakiety automatycznie za pomocą menedżera pip? [y/N]: ').strip().lower()
            if resp != 'y':
                logger.info('Pominięto instalację. Uruchamianie potoku na obecnych pakietach.')
                return

            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)]
            logger.info(f'Instalacja w toku. Uruchamiam proces: {cmd}')
            subprocess.check_call(cmd)
            logger.info('[OK] Instalacja zakończona powodzeniem.')

        check_and_install_requirements()

        pipeline = FullMLPipeline()
        success = pipeline.run()
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.warning("\n[!] Pipeline przerwany ręcznie przez operatora.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Niespodziewany błąd krytyczny jądra potoku: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()