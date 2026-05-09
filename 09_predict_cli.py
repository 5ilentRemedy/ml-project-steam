# Importowanie wymaganych bibliotek
import pandas as pd # Do pracy z ramkami danych
import numpy as np # Do operacji numerycznych
from pathlib import Path # Do operacji na ścieżkach plików
import joblib # Do serializacji i deserializacji obiektów Pythona (np. modeli)
import logging # Do logowania informacji
import datetime # Do pracy z datami i czasem
import sys # Do interakcji z systemem operacyjnym
import io # Do obsługi strumieni wejścia/wyjścia

# Ustawienie kodowania wyjścia standardowego na UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Konfiguracja systemu logowania
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Definicja klasy do przewidywania sukcesu gry
class GameSuccessPredictor:
    # Inicjalizacja obiektu GameSuccessPredictor
    def __init__(self):
        self.project_dir = Path(__file__).parent # Ustalenie katalogu projektu
        self.models_dir = self.project_dir / "models" # Ścieżka do katalogu modeli
        self.reports_dir = self.project_dir / "reports" / "predictions" # Ścieżka do katalogu raportów predykcyjnych
        
        # Tworzenie katalogu na raporty predykcyjne, jeśli nie istnieje
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None # Załadowany model
        self.game_name = "" # Nazwa gry wprowadzona przez użytkownika
        self.features = {} # Słownik do przechowywania cech gry wprowadzonych przez użytkownika

    # Metoda do ładowania wytrenowanego modelu z dysku
    def load_model(self):
        model_path = self.models_dir / "best_model.joblib" # Ścieżka do pliku modelu
        # Sprawdzenie, czy plik modelu istnieje
        if not model_path.exists():
            logger.error(f"! Nie znaleziono modelu w {model_path}.")
            logger.error("Uruchom najpierw skrypt 07_model_training.py!")
            sys.exit(1) # Zakończenie działania skryptu z błędem
            
        self.model = joblib.load(model_path) # Załadowanie modelu za pomocą joblib
        logger.info(f"OK Zaladowano model: {type(self.model).__name__}")

    # Metoda do interaktywnego zbierania danych od użytkownika w terminalu
    def get_user_input(self):
        print("\n" + "="*60)
        print("STEAM SUCCESS PREDICTOR - KREATOR PREDYKCJI")
        print("="*60)
        print("Wprowadz parametry swojej planowanej gry, aby oszacowac")
        print("jej szanse na zdobycie wysokiej oceny (>=75 Metacritic/User Score).\n")

        self.game_name = input("1. Podaj robocza nazwe gry: ").strip() # Pobranie nazwy gry
        if not self.game_name:
            self.game_name = "Projekt_Nieznany" # Domyślna nazwa, jeśli użytkownik nic nie wpisze

        # Zbieranie podstawowych danych od użytkownika z walidacją
        price = self._ask_float("2. Planowana cena w USD (np. 14.99, wpisz 0 dla Free-To-Play): ", 0.0)
        platforms = self._ask_int("3. Na ilu platformach wyjdzie gra? (1-3 np. Win, Mac, Linux): ", 1, 1, 3)
        genres = self._ask_int("4. Ile gatunkow opisuje Twoja gre? (np. 2 dla Action, RPG): ", 1, 1, 10)
        achievements = self._ask_yes_no("5. Czy gra bedzie miala system osiagniec (Achievements)? (t/n): ")
        
        print("\n--- Pytania o estymacje po premierze ---")
        est_reviews = self._ask_int("6. Spodziewana calkowita liczba recenzji (np. 500): ", 100, 0)
        est_ratio = self._ask_float("7. Spodziewany % pozytywnych recenzji (np. 0.85 dla 85%): ", 0.8, 0.0, 1.0)
        est_owners = self._ask_int("8. Spodziewana liczba graczy/posiadaczy (np. 20000): ", 10000, 0)

        # Przetwarzanie zebranych danych na cechy dla modelu (identyczne jak w Feature Engineering)
        self.features = {
            'Release_year': datetime.datetime.now().year, # Bieżący rok jako rok wydania
            'Days_since_release': 0, # 0 dni od wydania dla nowej gry
            'Platform_count': platforms,
            'Price': price,
            'Is_free': 1 if price == 0 else 0, # Cechy binarne
            'Total_reviews': est_reviews,
            'Review_ratio': est_ratio,
            'Log_owners': np.log1p(est_owners), # Logarytmiczna transformacja
            'Has_achievements': 1 if achievements else 0, # Cecha binarna
            'Log_total_reviews': np.log1p(est_reviews), # Logarytmiczna transformacja
            'Genre_count': genres
        }

    # Metoda pomocnicza do pobierania wartości zmiennoprzecinkowych od użytkownika
    def _ask_float(self, prompt, default=0.0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default # Zwrócenie wartości domyślnej, jeśli użytkownik nic nie wpisze
            try:
                f_val = float(val)
                # Walidacja zakresu wartości
                if min_val is not None and f_val < min_val:
                    print(f"  ! Wartosc musi byc >= {min_val}")
                    continue
                if max_val is not None and f_val > max_val:
                    print(f"  ! Wartosc musi byc <= {max_val}")
                    continue
                return f_val
            except ValueError:
                print("  ! Wpisz poprawna liczbe (uzyj kropki dla dziesietnych).")

    # Metoda pomocnicza do pobierania wartości całkowitych od użytkownika
    def _ask_int(self, prompt, default=0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default # Zwrócenie wartości domyślnej, jeśli użytkownik nic nie wpisze
            try:
                i_val = int(val)
                # Walidacja zakresu wartości
                if min_val is not None and i_val < min_val:
                    print(f"  ! Wartosc musi byc >= {min_val}")
                    continue
                if max_val is not None and i_val > max_val:
                    print(f"  ! Wartosc musi byc <= {max_val}")
                    continue
                return i_val
            except ValueError:
                print("  ! Wpisz poprawna liczbe calkowita.")

    # Metoda pomocnicza do pobierania odpowiedzi tak/nie od użytkownika
    def _ask_yes_no(self, prompt):
        while True:
            val = input(prompt).lower()
            if val in ['t', 'tak', 'y', 'yes']: return True
            if val in ['n', 'nie', 'no']: return False
            print("  ! Wpisz 't' lub 'n'.")

    # Metoda do przeprowadzania predykcji na podstawie wprowadzonych danych
    def predict(self):
        # Konwersja słownika cech do pojedynczego wiersza DataFrame
        df_input = pd.DataFrame([self.features])
        
        # Upewnienie się o kolejności kolumn i uzupełnienie brakujących (jeśli model tego wymaga)
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = list(self.model.feature_names_in_)
            # Wypełnianie brakujących kolumn zerami (np. dla cech One-Hot Encoding, które nie zostały wprowadzone)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[expected_cols] # Ustawienie kolumn w oczekiwanej kolejności

        print("\nTrwa analizowanie danych przez model...")
        
        self.prediction = self.model.predict(df_input)[0] # Generowanie predykcji klasy
        
        # Obliczanie prawdopodobieństw sukcesu i porażki
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df_input)[0]
            self.prob_success = proba[1] * 100 # Prawdopodobieństwo klasy pozytywnej
            self.prob_fail = proba[0] * 100 # Prawdopodobieństwo klasy negatywnej
        else:
            # Fallback, jeśli model nie wspiera predict_proba
            self.prob_success = 100.0 if self.prediction == 1 else 0.0
            self.prob_fail = 100.0 if self.prediction == 0 else 0.0

        # Wyświetlenie wyniku predykcji
        if self.prediction == 1:
            print(f"SUKCES! Model przewiduje, ze gra osiagnie WYSOKA OCENE (Pewnosc: {self.prob_success:.1f}%)")
        else:
            print(f"RYZYKO! Model przewiduje, ze gra uzyska NISKA/SREDNIA OCENE (Pewnosc: {self.prob_fail:.1f}%)")

    # Metoda do generowania raportu w formacie Markdown
    def generate_markdown_report(self):
        print("Generowanie eleganckiego raportu Markdown...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Znacznik czasu
        # Tworzenie bezpiecznej nazwy pliku z nazwy gry
        safe_name = "".join([c for c in self.game_name if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_")
        report_filename = f"Raport_{safe_name}_{timestamp}.md" # Nazwa pliku raportu
        report_path = self.reports_dir / report_filename # Pełna ścieżka do raportu

        # Ustalanie tekstowego opisu wyniku predykcji
        result_text = "**WYSOKA OCENA (Sukces)**" if self.prediction == 1 else "**NISKA/SREDNIA OCENA (Ryzyko)**"
        
        # Treść raportu w formacie Markdown
        md_content = f"""# Raport Predykcyjny ML: {self.game_name}

**Data analizy:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Uzyty model:** `{type(self.model).__name__}`

---

## Wynik Analizy

* **Przewidywany status:** {result_text}
* **Prawdopodobienstwo sukcesu:** {self.prob_success:.1f}%
* **Prawdopodobienstwo porazki:** {self.prob_fail:.1f}%

---

## Wprowadzone Parametry

| Parametr | Wartosc | Znaczenie w Modelu |
|----------|---------|--------------------|
| Cena (Price) | ${self.features['Price']:.2f} | Finansowy prog wejscia |
| Free-To-Play | {'Tak' if self.features['Is_free'] == 1 else 'Nie'} | Model biznesowy |
| Platformy | {self.features['Platform_count']} | Dostepnosc (Windows/Mac/Linux) |
| Liczba Gatunkow | {self.features['Genre_count']} | Roznorodnosc mechanik |
| Osiagniecia | {'Tak' if self.features['Has_achievements'] == 1 else 'Nie'} | Zwiekszona retencja graczy |
| Spodziewane Recenzje | {self.features['Total_reviews']} | Poziom zaangazowania (Total_reviews) |
| Spodziewany % Pozytywow | {self.features['Review_ratio'] * 100:.0f}% | Sentyment bazy (Review_ratio) |
| Spodziewani Posiadacze | {int(np.expm1(self.features['Log_owners']))} | Szacowana wielkosc grupy odbiorcow |

---

## Automatyczne Rekomendacje

Na podstawie ogolnych wzorcow modelu (Mozesz to przetestowac zmieniajac parametry w CLI):
1. **Model Biznesowy:** Upewnij sie, ze Twoja cena odpowiada zawartosci. Zbyt wysoka cena drastycznie zwieksza oczekiwania graczy, co moze obnizyc recenzje.
2. **Zaangazowanie:** Gry z systemem osiagniec (Achievements) statystycznie dluzej utrzymuja graczy w grze.
3. **Platformy:** Portowanie gry na Mac/Linux (zwiekszenie liczby platform) czesto pozytywnie wplywa na odbior przez nisze, co zawyza srednia recenzji.

*Raport wygenerowany automatycznie przez narzedzie Steam Games Preprocessing Pipeline.*
"""
        # Zapisanie treści raportu do pliku Markdown
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"\nWygenerowano pelny raport w formacie Markdown!")
        print(f"Sciezka: {report_path.relative_to(self.project_dir)}")
        print("Wskazowka: Otworz ten plik w VS Code (prawy przycisk myszy -> 'Open Preview'), by zobaczyc sformatowany wynik.")

    # Metoda uruchamiająca cały proces predykcji
    def run(self):
        self.load_model() # Załadowanie modelu
        self.get_user_input() # Zebranie danych od użytkownika
        self.predict() # Przeprowadzenie predykcji
        self.generate_markdown_report() # Wygenerowanie raportu Markdown

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    try:
        app = GameSuccessPredictor() # Utworzenie instancji klasy GameSuccessPredictor
        app.run() # Uruchomienie procesu predykcji
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez uzytkownika.") # Obsługa przerwania przez użytkownika (Ctrl+C)
        sys.exit(0) # Zakończenie programu