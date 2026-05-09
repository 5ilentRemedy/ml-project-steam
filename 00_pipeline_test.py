# Importowanie wymaganych bibliotek
import subprocess # Do uruchamiania zewnętrznych skryptów
import sys # Do interakcji z systemem operacyjnym
import time # Do mierzenia czasu wykonania
from pathlib import Path # Do operacji na ścieżkach plików
from datetime import datetime # Do pracy z datami i czasem
import logging # Do logowania informacji
import io # Do obsługi strumieni wejścia/wyjścia
import re # Do operacji na wyrażeniach regularnych
import importlib # Do dynamicznego importowania modułów
from importlib import metadata # Do pobierania metadanych pakietów

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('pipeline_execution.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Definicja klasy zarządzającej całym potokiem ML
class FullMLPipeline:
# Inicjalizacja obiektu potoku
    def __init__(self):
        self.project_dir = Path(__file__).parent # Ustawienie katalogu projektu
        
        # Definicja kroków potoku ML
        self.scripts = [
            ('02_data_exploration.py', 'Eksploracja danych'),
            ('03_data_cleaning.py', 'Czyszczenie danych'),
            ('04_feature_engineering.py', 'Inzynieria cech'),
            ('05_data_validation.py', 'Walidacja danych'),
            ('06a_data_validator.py', 'Walidacja i analiza danych'),
            ('06_data_export.py', 'Export i przygotowanie'),
            ('07_model_training.py', 'Trenowanie modeli ML'),
            ('08_model_evaluation.py', 'Ewaluacja i wizualizacja')
        ]
        self.results = {} # Słownik do przechowywania wyników wykonania każdego skryptu
    
    # Metoda do uruchamiania pojedynczego skryptu
    def run_script(self, script_name, description):
        logger.info("\n" + "=" * 80)
        logger.info(f"KROK: {description}")
        logger.info(f"Skrypt: {script_name}")
        logger.info("=" * 80)
        
        script_path = self.project_dir / script_name # Pełna ścieżka do skryptu
        
        # Sprawdzenie, czy skrypt istnieje
        if not script_path.exists():
            logger.error(f"ERROR Skrypt nie znaleziony: {script_path}")
            return False
        
        try:
            start_time = time.time() # Rozpoczęcie pomiaru czasu
            
            # Uruchomienie skryptu jako podprocesu
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False, # Wyjście podprocesu będzie widoczne w konsoli głównej
                text=True, # Użyj trybu tekstowego dla wyjścia
                cwd=str(self.project_dir) # Ustawienie bieżącego katalogu roboczego
            )
            
            elapsed_time = time.time() - start_time # Obliczenie czasu wykonania
            
            # Sprawdzenie kodu wyjścia podprocesu
            if result.returncode == 0:
                logger.info(f"OK {description} - SUKCES ({elapsed_time:.1f}s)")
                self.results[script_name] = {
                    'status': 'SUCCESS',
                    'time': elapsed_time
                }
                return True
            else:
                logger.error(f"ERROR {description} - BLAD (kod: {result.returncode})")
                self.results[script_name] = {
                    'status': 'FAILED',
                    'time': elapsed_time
                }
                return False
        
        # Obsługa wyjątków podczas uruchamiania skryptu
        except Exception as e:
            logger.error(f"ERROR {description} - WYJATEK: {str(e)}")
            self.results[script_name] = {
                'status': 'ERROR',
                'time': 0,
                'error': str(e)
            }
            return False
    
    # Metoda do weryfikacji istnienia plików wyjściowych
    def verify_outputs(self):
        logger.info("\n" + "=" * 80)
        logger.info("WERYFIKACJA PLIKOW WYJSCIOWYCH")
        logger.info("=" * 80)
        
        # Definicja oczekiwanych plików wyjściowych
        expected_files = {
            'data/games_cleaned.csv': 'Oczyszczone dane',
            'data/games_engineered.csv': 'Dane z cechami',
            'data/processed/games_final.csv': 'Finalne dane (CSV)',
            'data/processed/games_train.csv': 'Zbior treningowy',
            'data/processed/games_test.csv': 'Zbior testowy',
            'reports/outlier_report_iqr.csv': 'Raport outlierow',
            'reports/significant_correlations_report.csv': 'Raport istotnych korelacji',
            'reports/plots/correlation_distribution_histogram.png': 'Wykres: Histogram korelacji',
            'reports/plots/correlation_heatmap.png': 'Wykres: Heatmapa korelacji',
            'reports/02_validation_report.json': 'Raport walidacji danych',
            'reports/03_export_summary.json': 'Raport exportu',
            'models/best_model.joblib': 'Zapisany model ML',
            'reports/04_model_training_report.json': 'Raport z treningu ML',
            'reports/05_evaluation_metrics.json': 'Metryki ewaluacji ML',
            'reports/figures/confusion_matrix.png': 'Wykres: Macierz pomylek',
            'reports/figures/roc_pr_curves.png': 'Wykres: Krzywe ROC i PR'
        }
        
        verification = {} # Słownik do przechowywania wyników weryfikacji
        
        # Sprawdzenie istnienia każdego pliku
        for file_path, description in expected_files.items():
            full_path = self.project_dir / file_path
            exists = full_path.exists()
            status = "OK" if exists else "MISS"
            
            logger.info(f"{status} {description:30} - {file_path}")
            verification[file_path] = exists
        
        return all(verification.values()) # Zwraca True, jeśli wszystkie pliki istnieją
    
    # Metoda do drukowania podsumowania wykonania potoku
    def print_summary(self):
        logger.info("\n" + "=" * 80)
        logger.info("PODSUMOWANIE PIPLELINE'U")
        logger.info("=" * 80 + "\n")
        
        total_time = sum(r.get('time', 0) for r in self.results.values()) # Całkowity czas wykonania potoku
        
        # Nagłówek tabeli podsumowania
        logger.info(f"{'Skrypt':<30} {'Status':<10} {'Czas (s)':<10}")
        logger.info("-" * 50)
        
        # Wyświetlenie wyników dla każdego skryptu
        for script, result in self.results.items():
            status = result['status']
            time_val = result.get('time', 0)
            logger.info(f"{script:<30} {status:<10} {time_val:<10.1f}")
        
        logger.info("-" * 50)
        logger.info(f"{'RAZEM':<30} {'':<10} {total_time:<10.1f}s")
        
        # Określenie ogólnego statusu potoku
        all_success = all(r['status'] == 'SUCCESS' for r in self.results.values())
        
        if all_success:
            logger.info("\nOK PELNY PIPELINE ML ZAKONCZONY POMYSLNIE")
        else:
            failed_count = sum(1 for r in self.results.values() if r['status'] != 'SUCCESS')
            logger.warning(f"\n! {failed_count} kroki nie powiodly sie")
        
        # Informacje o katalogach wyjściowych
        logger.info("\nKatalogi wyjsciowe:")
        logger.info("- Dane: data/processed/")
        logger.info("- Modele: models/")
        logger.info("- Wykresy walidacji: reports/plots/")
        logger.info("- Raporty i wykresy: reports/figures/")
        logger.info("\n" + "=" * 80 + "\n")
    
    # Metoda uruchamiająca cały potok ML
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEAM GAMES - PELNY MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Tworzenie wszystkich wymaganych katalogów, jeśli nie istnieją
        (self.project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models").mkdir(parents=True, exist_ok=True)
        
        logger.info("\nUstawienia:")
        logger.info(f"  Katalog projektu: {self.project_dir}")
        logger.info(f"  Liczba krokow: {len(self.scripts)}")
        
        all_success = True # Flaga sukcesu całego potoku
        # Uruchamianie kolejnych skryptów potoku
        for script_name, description in self.scripts:
            success = self.run_script(script_name, description)
            if not success:
                all_success = False
                logger.warning(f"! Pipleline zatrzymany z powodu bledu w: {description}")
                break # Przerwanie potoku w przypadku błędu
        
        # Weryfikacja plików wyjściowych po zakończeniu wszystkich skryptów (jeśli nie było błędów)
        if all_success:
            all_files_exist = self.verify_outputs()
            if not all_files_exist:
                logger.warning("! Niektore pliki wyjsciowe nie zostaly utworzone")
                all_success = False
        
        # Wyświetlenie podsumowania wykonania potoku
        self.print_summary()
        
        return all_success # Zwrócenie ogólnego statusu potoku

# Główna funkcja programu
def main():
    try:
        # Funkcja do sprawdzania i instalowania zależności z requirements.txt
        def check_and_install_requirements():
            req_file = Path(__file__).parent / 'requirements.txt'
            if not req_file.exists():
                logger.info('Brak requirements.txt - pomijam sprawdzanie zaleznosci.')
                return

            logger.info('Sprawdzam wymagane zaleznosci z requirements.txt...')
            lines = [l.strip() for l in req_file.read_text(encoding='utf-8').splitlines()]
            pkg_lines = [l for l in lines if l and not l.startswith('#')]

            missing = [] # Lista brakujących pakietów
            mismatch = [] # Lista pakietów z niezgodnymi wersjami

            # Iteracja przez wymagane pakiety
            for line in pkg_lines:
                m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
                if not m:
                    continue
                pkg_name = m.group(1)

                # Kandydaci na nazwy modułów do importu
                mod_candidates = [pkg_name.replace('-', '_')]
                if pkg_name.lower() == 'scikit-learn':
                    mod_candidates.insert(0, 'sklearn')

                found = False
                for mod in mod_candidates:
                    try:
                        importlib.import_module(mod) # Próba importu modułu
                        found = True
                        break
                    except Exception:
                        continue

                if not found:
                    missing.append(pkg_name) # Dodanie do listy brakujących
                    continue

                # Sprawdzenie zainstalowanej wersji
                try:
                    installed_version = metadata.version(pkg_name)
                except metadata.PackageNotFoundError:
                    try:
                        installed_version = metadata.version(mod)
                    except Exception:
                        installed_version = None

                # Sprawdzenie zgodności wersji
                spec_match = re.search(r"([<>=!~].+)$", line)
                if installed_version and spec_match:
                    spec = spec_match.group(1)
                    try:
                        from packaging.specifiers import SpecifierSet
                        ss = SpecifierSet(spec)
                        if not ss.contains(installed_version):
                            mismatch.append((pkg_name, installed_version, spec)) # Dodanie do listy niezgodnych
                    except Exception:
                        pass

            # Raportowanie statusu zależności
            if not missing and not mismatch:
                logger.info('Wszystkie zaleznosci wydaja sie byc spelnione.')
                return

            if missing:
                logger.warning(f'Brakuje pakietow: {missing}')
            if mismatch:
                logger.warning(f'Pakiety z niezgodna wersja: {mismatch}')

            # Zapytanie użytkownika o instalację
            resp = input('Zainstalowac brakujace/niezgodne pakiety z requirements.txt? [y/N]: ').strip().lower()
            if resp != 'y':
                logger.info('Pomijam instalacje pakietow.')
                return

            # Wykonanie instalacji pakietów za pomocą pip
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)]
            logger.info(f'Uruchamiam: {cmd}')
            subprocess.check_call(cmd)
            logger.info('Instalacja zakonczona. Kontynuuje.')

        check_and_install_requirements() # Wywołanie funkcji sprawdzającej zależności

        pipeline = FullMLPipeline() # Utworzenie instancji potoku
        success = pipeline.run() # Uruchomienie potoku
        
        sys.exit(0 if success else 1) # Zakończenie programu z odpowiednim kodem wyjścia
    
    # Obsługa przerwania potoku przez użytkownika (Ctrl+C)
    except KeyboardInterrupt:
        logger.warning("\n! Pipleline przerwany przez uzytkownika")
        sys.exit(1)
    
    # Ogólna obsługa innych wyjątków
    except Exception as e:
        logger.error(f"ERROR Niespodziewany blad: {str(e)}", exc_info=True)
        sys.exit(1)

# Punkt wejścia programu
if __name__ == "__main__":
    main()