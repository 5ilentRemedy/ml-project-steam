"""
09_predict_cli.py - Interaktywne narzędzie CLI do predykcji dla nowych gier
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import datetime
import sys
import io

# Kodowanie UTF-8 dla konsoli
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class GameSuccessPredictor:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports" / "predictions"
        
        # Tworzenie katalogu na raporty predykcyjne
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.game_name = ""
        self.features = {}

    def load_model(self):
        """Ładuje wytrenowany model z dysku"""
        model_path = self.models_dir / "best_model.joblib"
        if not model_path.exists():
            logger.error(f"[!] Nie znaleziono modelu w {model_path}.")
            logger.error("Uruchom najpierw skrypt 07_model_training.py!")
            sys.exit(1)
            
        self.model = joblib.load(model_path)
        logger.info(f"[OK] Załadowano model: {type(self.model).__name__}")

    def get_user_input(self):
        """Interaktywny kreator zbierający dane od użytkownika w terminalu"""
        print("\n" + "="*60)
        print("🎮 STEAM SUCCESS PREDICTOR - KREATOR PREDYKCJI 🎮")
        print("="*60)
        print("Wprowadź parametry swojej planowanej gry, aby oszacować")
        print("jej szanse na zdobycie wysokiej oceny (>=75 Metacritic/User Score).\n")

        self.game_name = input("1. Podaj roboczą nazwę gry: ").strip()
        if not self.game_name:
            self.game_name = "Projekt_Nieznany"

        # Zbieranie podstawowych danych (z walidacją)
        price = self._ask_float("2. Planowana cena w USD (np. 14.99, wpisz 0 dla Free-To-Play): ", 0.0)
        platforms = self._ask_int("3. Na ilu platformach wyjdzie gra? (1-3 np. Win, Mac, Linux): ", 1, 1, 3)
        genres = self._ask_int("4. Ile gatunków opisuje Twoją grę? (np. 2 dla Action, RPG): ", 1, 1, 10)
        achievements = self._ask_yes_no("5. Czy gra będzie miała system osiągnięć (Achievements)? (t/n): ")
        
        # Estymacje (rzeczy trudne do przewidzenia przed premierą)
        print("\n--- Pytania o estymacje po premierze ---")
        est_reviews = self._ask_int("6. Spodziewana całkowita liczba recenzji (np. 500): ", 100, 0)
        est_ratio = self._ask_float("7. Spodziewany % pozytywnych recenzji (np. 0.85 dla 85%): ", 0.8, 0.0, 1.0)
        est_owners = self._ask_int("8. Spodziewana liczba graczy/posiadaczy (np. 20000): ", 10000, 0)

        # Przetwarzanie zebranych danych na cechy dla modelu (identyczne jak w Feature Engineering)
        self.features = {
            'Release_year': datetime.datetime.now().year, # Rok bieżący
            'Days_since_release': 0, # Nowa gra
            'Platform_count': platforms,
            'Price': price,
            'Is_free': 1 if price == 0 else 0,
            'Total_reviews': est_reviews,
            'Review_ratio': est_ratio,
            'Log_owners': np.log1p(est_owners),           # Automatyczne przeliczenie!
            'Has_achievements': 1 if achievements else 0,
            'Log_total_reviews': np.log1p(est_reviews),   # Automatyczne przeliczenie!
            'Genre_count': genres
        }

    def _ask_float(self, prompt, default=0.0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default
            try:
                f_val = float(val)
                if min_val is not None and f_val < min_val:
                    print(f"  [!] Wartość musi być >= {min_val}")
                    continue
                if max_val is not None and f_val > max_val:
                    print(f"  [!] Wartość musi być <= {max_val}")
                    continue
                return f_val
            except ValueError:
                print("  [!] Wpisz poprawną liczbę (użyj kropki dla dziesiętnych).")

    def _ask_int(self, prompt, default=0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default
            try:
                i_val = int(val)
                if min_val is not None and i_val < min_val:
                    print(f"  [!] Wartość musi być >= {min_val}")
                    continue
                if max_val is not None and i_val > max_val:
                    print(f"  [!] Wartość musi być <= {max_val}")
                    continue
                return i_val
            except ValueError:
                print("  [!] Wpisz poprawną liczbę całkowitą.")

    def _ask_yes_no(self, prompt):
        while True:
            val = input(prompt).lower()
            if val in ['t', 'tak', 'y', 'yes']: return True
            if val in ['n', 'nie', 'no']: return False
            print("  [!] Wpisz 't' lub 'n'.")

    def predict(self):
        """Przeprowadza predykcję na podstawie wprowadzonych danych"""
        # Konwersja słownika do pojedynczego wiersza DataFrame
        df_input = pd.DataFrame([self.features])
        
        # Upewnienie się o kolejności kolumn (model tego wymaga)
        # Pobieramy oczekiwaną strukturę z modelu (jeśli posiada atrybut feature_names_in_)
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = list(self.model.feature_names_in_)
            # Wypełnianie brakujących kolumn zerami (jeśli jakieś pominęliśmy, choć tu mamy wszystkie 11)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[expected_cols]

        print("\n⏳ Trwa analizowanie danych przez model...")
        
        self.prediction = self.model.predict(df_input)[0]
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df_input)[0]
            self.prob_success = proba[1] * 100
            self.prob_fail = proba[0] * 100
        else:
            self.prob_success = 100.0 if self.prediction == 1 else 0.0
            self.prob_fail = 100.0 if self.prediction == 0 else 0.0

        if self.prediction == 1:
            print(f"🎉 SUKCES! Model przewiduje, że gra osiągnie WYSOKĄ OCENĘ (Pewność: {self.prob_success:.1f}%)")
        else:
            print(f"⚠️ RYZYKO! Model przewiduje, że gra uzyska NISKĄ/ŚREDNIĄ OCENĘ (Pewność: {self.prob_fail:.1f}%)")

    def generate_markdown_report(self):
        """Generuje elegancki raport Markdown"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c for c in self.game_name if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_")
        report_filename = f"Raport_{safe_name}_{timestamp}.md"
        report_path = self.reports_dir / report_filename

        # Ustalanie biznesowego wyniku
        result_text = "🟢 **WYSOKA OCENA (Sukces)**" if self.prediction == 1 else "🔴 **NISKA/ŚREDNIA OCENA (Ryzyko)**"
        
        md_content = f"""# Raport Predykcyjny ML: {self.game_name}

**Data analizy:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Użyty model:** `{type(self.model).__name__}`

---

## 🎯 Wynik Analizy

* **Przewidywany status:** {result_text}
* **Prawdopodobieństwo sukcesu:** {self.prob_success:.1f}%
* **Prawdopodobieństwo porażki:** {self.prob_fail:.1f}%

---

## 📊 Wprowadzone Parametry

| Parametr | Wartość | Znaczenie w Modelu |
|----------|---------|--------------------|
| Cena (Price) | ${self.features['Price']:.2f} | Finansowy próg wejścia |
| Free-To-Play | {'Tak' if self.features['Is_free'] == 1 else 'Nie'} | Model biznesowy |
| Platformy | {self.features['Platform_count']} | Dostępność (Windows/Mac/Linux) |
| Liczba Gatunków | {self.features['Genre_count']} | Różnorodność mechanik |
| Osiągnięcia | {'Tak' if self.features['Has_achievements'] == 1 else 'Nie'} | Zwiększona retencja graczy |
| Spodziewane Recenzje | {self.features['Total_reviews']} | Poziom zaangażowania (Total_reviews) |
| Spodziewany % Pozytywów | {self.features['Review_ratio'] * 100:.0f}% | Sentyment bazy (Review_ratio) |
| Spodziewani Posiadacze | {int(np.expm1(self.features['Log_owners']))} | Szacowana wielkość grupy odbiorców |

---

## 💡 Automatyczne Rekomendacje

Na podstawie ogólnych wzorców modelu (Możesz to przetestować zmieniając parametry w CLI):
1. **Model Biznesowy:** Upewnij się, że Twoja cena odpowiada zawartości. Zbyt wysoka cena drastycznie zwiększa oczekiwania graczy, co może obniżyć recenzje.
2. **Zaangażowanie:** Gry z systemem osiągnięć (Achievements) statystycznie dłużej utrzymują graczy w grze.
3. **Platformy:** Portowanie gry na Mac/Linux (zwiększenie liczby platform) często pozytywnie wpływa na odbiór przez nisze, co zawyża średnią recenzji.

*Raport wygenerowany automatycznie przez narzędzie Steam Games Preprocessing Pipeline.*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"\n📄 Wygenerowano pełny raport w formacie Markdown!")
        print(f"📁 Ścieżka: {report_path.relative_to(self.project_dir)}")
        print("💡 Wskazówka: Otwórz ten plik w VS Code (prawy przycisk myszy -> 'Open Preview'), by zobaczyć sformatowany wynik.")

    def run(self):
        self.load_model()
        self.get_user_input()
        self.predict()
        self.generate_markdown_report()

if __name__ == "__main__":
    try:
        app = GameSuccessPredictor()
        app.run()
    except KeyboardInterrupt:
        print("\n\n[!] Przerwano przez użytkownika.")
        sys.exit(0)