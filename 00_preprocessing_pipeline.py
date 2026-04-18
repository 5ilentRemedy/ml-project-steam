"""
00_preprocessing_pipeline.py
Glowny orchestrator pipleline'u preprocessingu

Ten skrypt uruchamia wszystkie kroki preprocessingu w sekwencji:
1. Eksploracja danych
2. Czyszczenie danych
3. Inżynieria cech
4. Walidacja
5. Export
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
        logging.FileHandler('preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Orchestrator calego pipleline'u preprocessingu"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.scripts = [
            ('02_data_exploration.py', 'Eksploracja danych'),
            ('03_data_cleaning.py', 'Czyszczenie danych'),
            ('04_feature_engineering.py', 'Inżynieria cech'),
            ('05_data_validation.py', 'Walidacja danych'),
            ('06_data_export.py', 'Export i przygotowanie')
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
                logger.error(f"[ERROR] {description} - BLAD (kod: {result.returncode})")
                self.results[script_name] = {
                    'status': 'FAILED',
                    'time': elapsed_time
                }
                return False
        
        except Exception as e:
            logger.error(f"[ERROR] {description} - WYJATEK: {str(e)}")
            self.results[script_name] = {
                'status': 'ERROR',
                'time': 0,
                'error': str(e)
            }
            return False
    
    def verify_outputs(self):
        """Weryfikuje czy wszystkie pliki wyjsciowe zostały utworzone"""
        logger.info("\n" + "=" * 80)
        logger.info("WERYFIKACJA PLIKOW WYJSCIOWYCH")
        logger.info("=" * 80)
        
        expected_files = {
            'data/games_cleaned.csv': 'Oczyszczone dane',
            'data/games_engineered.csv': 'Dane z cechami',
            'data/processed/games_final.csv': 'Finalne dane (CSV)',
            'data/processed/games_final.parquet': 'Finalne dane (Parquet)',
            'data/processed/games_train.csv': 'Zbior treningowy',
            'data/processed/games_test.csv': 'Zbior testowy',
            'data/processed/dataset_manifest.json': 'Manifest datasetu',
            'reports/01_exploration_summary.json': 'Raport eksploracji',
            'reports/02_validation_report.json': 'Raport walidacji',
            'reports/03_export_summary.json': 'Raport exportu'
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
        
        # Tabela wynikow
        logger.info(f"{'Skrypt':<30} {'Status':<10} {'Czas (s)':<10}")
        logger.info("-" * 50)
        
        for script, result in self.results.items():
            status = result['status']
            time_val = result.get('time', 0)
            logger.info(f"{script:<30} {status:<10} {time_val:<10.1f}")
        
        logger.info("-" * 50)
        logger.info(f"{'RAZEM':<30} {'':<10} {total_time:<10.1f}s")
        
        # Status ogolny
        all_success = all(r['status'] == 'SUCCESS' for r in self.results.values())
        
        if all_success:
            logger.info("\n[OK] PREPROCESSING ZAKONCZONY POMYSLNIE")
        else:
            failed_count = sum(1 for r in self.results.values() if r['status'] != 'SUCCESS')
            logger.warning(f"\n[!] {failed_count} kroki nie powiodly sie")
        
        logger.info("\nKatalog wyjsciowy: data/processed/")
        logger.info("Pliki logu: reports/")
        logger.info("\n" + "=" * 80 + "\n")
    
    def run(self):
        """Uruchamia caly pipleline"""
        logger.info("\n" + "=" * 80)
        logger.info("STEAM GAMES PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Utw katalogi
        (self.project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        logger.info("\nUstawienia:")
        logger.info(f"  Katalog projektu: {self.project_dir}")
        logger.info(f"  Liczba krokow: {len(self.scripts)}")
        
        # Uruchom wszystkie kroki
        all_success = True
        for script_name, description in self.scripts:
            success = self.run_script(script_name, description)
            if not success:
                all_success = False
                logger.warning(f"[!] Pipleline zatrzymany z powodu bledu w: {description}")
                break
        
        # Weryfikacja
        if all_success:
            all_files_exist = self.verify_outputs()
            if not all_files_exist:
                logger.warning("[!] Niektore pliki wyjsciowe nie zostaly utworzone")
                all_success = False
        
        # Podsumowanie
        self.print_summary()
        
        return all_success

def main():
    """Glowna funkcja"""
    try:
        # Sprawdź zależności przed uruchomieniem pipeline'u
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
                # Wyciągnij nazwę pakietu (przed operatorem wersji)
                m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
                if not m:
                    continue
                pkg_name = m.group(1)

                # Heurystyka nazwy modułu: zamień '-' na '_' (np. scikit-learn -> scikit_learn)
                mod_candidates = [pkg_name.replace('-', '_')]
                # popularne aliasy
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

                # Sprawdź wersję jeśli podano (jeśli metadata jest dostępne)
                try:
                    installed_version = metadata.version(pkg_name)
                except metadata.PackageNotFoundError:
                    try:
                        installed_version = metadata.version(mod)
                    except Exception:
                        installed_version = None

                # Jeśli linia zawiera specyfikator wersji, porównaj
                spec_match = re.search(r"([<>=!~].+)$", line)
                if installed_version and spec_match:
                    spec = spec_match.group(1)
                    try:
                        # Użyj packaging jeśli dostępny
                        from packaging.specifiers import SpecifierSet
                        ss = SpecifierSet(spec)
                        if not ss.contains(installed_version):
                            mismatch.append((pkg_name, installed_version, spec))
                    except Exception:
                        # Nie możemy sprawdzić dokładnie — pomiń wersję
                        pass

            if not missing and not mismatch:
                logger.info('Wszystkie zależności wydają się być spełnione.')
                return

            # Podsumowanie
            if missing:
                logger.warning(f'Brakuje pakietów: {missing}')
            if mismatch:
                logger.warning(f'Pakiety z niezgodną wersją: {mismatch}')

            # Zapytaj użytkownika o instalację
            resp = input('Zainstalować brakujące/niezgodne pakiety z requirements.txt? [y/N]: ').strip().lower()
            if resp != 'y':
                logger.info('Pominięto instalację pakietów.')
                return

            # Uruchom instalację
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)]
            logger.info(f'Uruchamiam: {cmd}')
            subprocess.check_call(cmd)
            logger.info('Instalacja zakończona. Kontynuuję.')

        check_and_install_requirements()

        pipeline = PreprocessingPipeline()
        success = pipeline.run()
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.warning("\n[!] Pipleline przerwany przez uzytkownika")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"[ERROR] Niespodziewany blad: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
