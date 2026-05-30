"""
08_predict_cli.py - Interaktywne narzędzie CLI do predykcji Sukcesu Rynkowego nowych gier.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import datetime
import sys
import io

# Kodowanie UTF-8 dla konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class GameSuccessPredictor:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.models_dir = self.project_dir / "models"
        self.reports_dir = self.project_dir / "reports" / "predictions"
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.label_encoder = None
        self.game_name = ""
        self.features = {}
        self.classes = ['Zrozumiały Hit', 'Ukryty Diament', 'Przeciętniak', 'Klapa rynkowa']

    def load_model(self):
        """Ładuje wytrenowany model oraz ewentualny LabelEncoder"""
        model_path = self.models_dir / "best_model.joblib"
        if not model_path.exists():
            logger.error(f"[!] Nie znaleziono zapisanego modelu w {model_path}.")
            logger.error("Uruchom najpierw pełny pipeline (00_pipeline_test.py)!")
            sys.exit(1)
            
        self.model = joblib.load(model_path)
        
        encoder_path = self.models_dir / "label_encoder.joblib"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            
        logger.info(f"[OK] Załadowano silnik predykcyjny klasyfikacji: {type(self.model).__name__}")

    def get_user_input(self):
        """Interaktywny kreator zbierający wyłącznie parametry przedpremierowe"""
        print("\n" + "="*60)
        print("🎮 STEAM MARKET SUCCESS PREDICTOR - CLI INTERFACES 🎮")
        print("="*60)
        print("Wprowadź parametry techniczno-biznesowe swojej projektowanej gry,\n"
              "aby sztuczna inteligencja oszacowała profil jej sukcesu rynkowego.\n")

        self.game_name = input("1. Podaj roboczą nazwę projektu: ").strip()
        if not self.game_name:
            self.game_name = "Projekt_Indie_X"

        price = self._ask_float("2. Planowana cena w USD (np. 9.99, wpisz 0 dla Free-To-Play): ", 0.0, min_val=0.0)
        platforms = self._ask_int("3. Na ilu platformach systemowych planujesz premierę? (1-3 np. Windows/Mac/Linux): ", 1, 1, 3)
        genres = self._ask_int("4. Ile głównych gatunków/tagów opisuje mechaniki gry?: ", 1, 1, 10)
        achievements = self._ask_yes_no("5. Czy gra będzie posiadała zaimplementowany system osiągnięć (Achievements)? (t/n): ")

        # Słownik bazowych cech projektowych
        self.features = {
            'Price': price,
            'Release_year': datetime.datetime.now().year,
            'Days_since_release': 0,  # Prognoza w dniu premiery
            'Platform_count': platforms,
            'Has_windows': 1,  # Zakładamy domyślne wsparcie dla Windows
            'Has_mac': 1 if platforms >= 2 else 0,
            'Has_linux': 1 if platforms == 3 else 0,
            'Is_multiplatform': 1 if platforms > 1 else 0,
            'Is_free': 1 if price == 0 else 0,
            'Has_achievements': 1 if achievements else 0,
            'Log_achievements': np.log1p(45 if achievements else 0),  # Średnia rynkowa dla gier z osiągnięciami
            'Genre_count': genres,
            'Category_count': genres + 2,  # Estymacja na podstawie powiązanych kategorii technicznych
            'Log_owners': np.log1p(0), # Start od zera posiadaczy
            'Price_Rating_ratio': price / 61.0,  # 60 to neutralna baza punktowa
            'Rating_Review_score': 0.0 # Brak recenzji na start
        }

        # 🔥 REPLIKACJA ONE-HOT ENCODINGU CENOWEGO (Zgodnie z potokiem Feature Engineering)
        price_categories = ['Free', 'Budget', 'Standard', 'Premium', 'AAA']
        current_cat = 'Free'
        if price > 0 and price <= 9.99: current_cat = 'Budget'
        elif price > 9.99 and price <= 19.99: current_cat = 'Standard'
        elif price > 19.99 and price <= 49.99: current_cat = 'Premium'
        elif price > 49.99: current_cat = 'AAA'

        for cat in price_categories:
            self.features[f'Price_category_{cat}'] = 1 if current_cat == cat else 0

        # Dodanie brakujących kategorii Metacritic z One-Hot
        meta_cats = ['None', 'Poor', 'Fair', 'Good', 'Excellent']
        for m_cat in meta_cats:
            self.features[f'Metacritic_category_{m_cat}'] = 1 if m_cat == 'None' else 0

    def _ask_float(self, prompt, default=0.0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default
            try:
                f_val = float(val)
                if min_val is not None and f_val < min_val:
                    print(f"  [!] Wartość musi być >= {min_val}"); continue
                if max_val is not None and f_val > max_val:
                    print(f"  [!] Wartość musi być <= {max_val}"); continue
                return f_val
            except ValueError:
                print("  [!] Wpisz poprawną wartość numeryczną.")

    def _ask_int(self, prompt, default=0, min_val=None, max_val=None):
        while True:
            val = input(prompt)
            if not val: return default
            try:
                i_val = int(val)
                if min_val is not None and i_val < min_val:
                    print(f"  [!] Wartość musi być >= {min_val}"); continue
                if max_val is not None and i_val > max_val:
                    print(f"  [!] Wartość musi być <= {max_val}"); continue
                return i_val
            except ValueError:
                print("  [!] Wpisz poprawną liczbę całkowitą.")

    def _ask_yes_no(self, prompt):
        while True:
            val = input(prompt).lower()
            if val in ['t', 'tak', 'y', 'yes']: return True
            if val in ['n', 'nie', 'no']: return False
            print("  [!] Wpisz 't' (tak) lub 'n' (nie).")

    def predict(self):
        """Dopasowuje cechy do oczekiwań modelu i wykonuje wieloklasową predykcję"""
        df_input = pd.DataFrame([self.features])
        
        # Automatyczne sortowanie kolumn pod strukturę wejściową modelu ML
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = list(self.model.feature_names_in_)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[expected_cols]

        print("\n⏳ Przetwarzanie i wnioskowanie probabilistyczne przez silnik AI...")
        
        # Wykonanie predykcji
        proba = self.model.predict_proba(df_input)[0]
        
        # Pobieramy poprawne nazwy klas w zależności od użytego algorytmu
        if type(self.model).__name__ == 'XGBClassifier' and self.label_encoder:
            pred_index = self.model.predict(df_input)[0]
            self.prediction = self.label_encoder.inverse_transform([pred_index])[0]
            model_classes = self.label_encoder.classes_
        else:
            self.prediction = self.model.predict(df_input)[0]
            model_classes = self.model.classes_

        # Mapujemy wyniki prawdopodobieństw do nazw klas
        self.prob_dict = {str(cls): float(p * 100) for cls, p in zip(model_classes, proba)}
        
        print("\n" + "="*60)
        print(f"📊 PROGNOZOWANY PROFIL SUKCESU DLA: {self.game_name}")
        print("="*60)
        print(f"Dominująca klasa wyniku: 🏆 **{self.prediction}**\n")
        print("Szczegółowy rozkład prawdopodobieństwa:")
        for cls in self.classes:
            p = self.prob_dict.get(cls, 0.0)
            bar = "█" * int(p / 5)
            print(f"  - {cls:22} | {p:5.1f}% | {bar}")
        print("="*60)

    def generate_markdown_report(self):
        """Eksportuje elegancki, strukturalny raport Markdown z predykcji"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c for c in self.game_name if c.isalnum() or c==' ']).rstrip().replace(" ", "_")
        report_path = self.reports_dir / f"Raport_Predykcji_{safe_name}_{timestamp}.md"
        
        table_rows = ""
        for cls in self.classes:
            p = self.prob_dict.get(cls, 0.0)
            table_rows += f"| {cls} | {p:.1f}% |\n"

        md_content = f"""# Raport Predykcji Biznesowej: {self.game_name}

**Data wykonania analizy:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sercem analizy jest model:** `{type(self.model).__name__}`

---

## 🎯 Główna Prognoza Rynkowa
Model klasyfikacyjny zaklasyfikował projekt do profilu: 🏆 **{self.prediction}**

### Pełny rozkład prawdopodobieństwa klas docelowych:
| Klasa Sukcesu Rynkowego | Szansa procentowa |
|-------------------------|-------------------|
{table_rows}

---

## 🛠️ Dane wejściowe projektu (Parametry Stałe)
- **Cena katalogowa:** ${self.features['Price']:.2f} USD
- **Model dystrybucji:** {'Darmowy (Free-to-Play)' if self.features['Is_free'] == 1 else 'Płatny komercyjny'}
- **Liczba wspieranych platform:** {self.features['Platform_count']} (Windows, Mac/Linux port)
- **Złożoność gatunkowa:** {self.features['Genre_count']} zdefiniowanych mechanik
- **Grywalność/Retencja:** {'Zaimplementowany system osiągnięć Steam' if self.features['Has_achievements'] == 1 else 'Brak osiągnięć'}

---

## 💡 Strategiczne Rekomendacje Potoku ML
1. **Dla Ukrytych Diamentów:** Jeśli gra ma wysoką szansę na *Ukryty Diament*, skup się na marketingu niszowym i budowaniu społeczności na Discordzie przed premierą – to pozwoli przesunąć grę do przedziału *Zrozumiały Hit*.
2. **Dla Klap Rynkowych:** Zweryfikuj strategię cenową. Zbyt wysoka cena przy małej liczbie platform i braku osiągnięć drastycznie spycha gry w obszary ignorowane przez algorytmy rekomendacji Steam.

*Raport wygenerowany automatycznie przez moduł wnioskowania potoku ML Steam Project.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n[OK] Raport Markdown zapisany pomyślnie.")
        print(f"📁 Lokalizacja pliku: {report_path.relative_to(self.project_dir)}")

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
        print("\n\n[!] Interfejs CLI został zamknięty przez użytkownika.")
        sys.exit(0)