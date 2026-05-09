# Steam Games ML Pipeline

## Przegląd

Ten projekt implementuje kompleksowy potok (pipeline) uczenia maszynowego (ML) dla danych gier Steam. Jego celem jest transformacja surowych danych z Kaggle w czysty, dobrze ustrukturyzowany dataset, a następnie wytrenowanie i ewaluacja modelu ML przewidującego sukces gry, oraz udostępnienie interaktywnego narzędzia do predykcji.

## Cel

- **Przetwarzanie Danych**: Od surowych danych do gotowego do ML datasetu.
- **Inżynieria Cech**: Tworzenie nowych, wartościowych cech z istniejących danych.
- **Walidacja Danych**: Zapewnienie wysokiej jakości danych na każdym etapie.
- **Modelowanie ML**: Trenowanie i porównywanie różnych modeli klasyfikacji.
- **Ewaluacja Modelu**: Głęboka analiza wydajności modelu, w tym interpretowalność (SHAP).
- **Narzędzie Predykcyjne**: Interaktywna aplikacja CLI do przewidywania sukcesu nowych gier.

## Struktura Projektu

```
ml-project-steam/
├── 00_pipeline_test.py             # Główny orchestrator całego pipeline'u ML
├── 01_data_collection.py           # Pobieranie danych z Kaggle
├── 02_data_exploration.py          # Krok 1: Eksploracja danych
├── 03_data_cleaning.py             # Krok 2: Czyszczenie danych
├── 04_feature_engineering.py       # Krok 3: Inżynieria cech
├── 05_data_validation.py           # Krok 4: Walidacja danych
├── 06_data_export.py               # Krok 5: Export i przygotowanie danych do ML
├── 06a_data_validator.py           # Krok 5: Dodatkowa walidacja i analiza danych
├── 07_model_training.py            # Krok 6: Trenowanie i porównywanie modeli ML
├── 08_model_evaluation.py          # Krok 7: Ewaluacja modelu i wizualizacje
├── 09_predict_cli.py               # Krok 8: Interaktywne narzędzie CLI do predykcji
├── data/
│   ├── games_YYYYMMDD_HHMMSS.csv   # Surowe dane z Kaggle
│   ├── games_cleaned.csv           # Dane po czyszczeniu
│   ├── games_engineered.csv        # Dane z nowymi cechami
│   └── processed/                  # Finalne dane dla ML i dokumentacja
│       ├── games_final.csv
│       ├── games_final.parquet
│       ├── games_train.csv
│       ├── games_test.csv
│       ├── games_group_*.csv       # Dane pogrupowane wg cech
│       ├── games_*_with_filters.xlsx # Dane w Excelu z filtrami
│       ├── games_final_grouped.xlsx  # Dane w Excelu z arkuszami dla grup cech
│       ├── columns_documentation.csv
│       ├── dataset_manifest.json
│       └── dataset_documentation.md
├── models/
│   └── best_model.joblib           # Zapisany najlepszy model ML
├── reports/
│   ├── pipeline_execution.log      # Logi z wykonania całego pipeline'u
│   ├── 01_exploration_summary.json # Raport z eksploracji
│   ├── 02_validation_report.json   # Raport z walidacji danych
│   ├── 03_export_summary.json      # Raport z eksportu danych
│   ├── outlier_report_iqr.csv      # Raport z wykrytymi outlierami
│   ├── significant_correlations_report.csv # Raport z istotnymi korelacjami
│   ├── 04_model_training_report.json # Raport z treningu modeli
│   ├── 05_evaluation_metrics.json  # Raport z metryk ewaluacji
│   └── figures/                    # Wykresy i wizualizacje
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       ├── feature_importance.png
│       └── shap_summary.png
│   └── plots/                      # Wykresy z dodatkowej walidacji
│       ├── correlation_distribution_histogram.png
│       ├── correlation_heatmap.png
│       ├── histogram_*.png         # Histogramy dla poszczególnych cech
│       └── boxplot_outliers_*.png  # Boxploty dla kolumn z outlierami
├── reports/predictions/            # Raporty z predykcji CLI
│   └── Raport_*.md
└── requirements.txt                # Wymagane pakiety Pythona
```

## Opis Skryptów

### `01_data_collection.py` - Pobieranie Danych

- **Funkcja**: Pobiera dataset gier Steam z Kaggle (`fronkongames/steam-games-dataset`).
- **Wyjście**: Zapisuje surowe dane do `data/games_YYYYMMDD_HHMMSS.csv`.
- **Uwaga**: Ten skrypt należy uruchomić raz, przed rozpoczęciem głównego pipeline'u, aby pobrać dane.

### `02_data_exploration.py` - Eksploracja Danych

- **Funkcja**: Ładuje surowe dane, analizuje ich strukturę, typy danych, braki, unikalne wartości i generuje podstawowe statystyki opisowe.
- **Wyjście**: Raport `reports/01_exploration_summary.json` oraz szczegółowe logi na konsoli.

### `03_data_cleaning.py` - Czyszczenie Danych

- **Funkcja**: Przetwarza surowe dane: usuwa duplikaty, konwertuje typy danych (np. daty, ceny), czyści błędne wartości, tworzy kolumnę `Review_ratio`.
- **Wyjście**: Oczyszczone dane w `data/games_cleaned.csv`.

### `04_feature_engineering.py` - Inżynieria Cech

- **Funkcja**: Tworzy nowe, wartościowe cechy z oczyszczonych danych. Obejmuje cechy czasowe, platformowe, recenzji, ocen (w tym nowa definicja `Is_highly_rated`), zawartości, ceny, kodowanie zmiennych kategorycznych, normalizację numerycznych i cechy interakcji.
- **Definicja sukcesu (`Is_highly_rated`)**: Gra jest uznana za sukces (1), jeśli ma minimum 30 recenzji i przynajmniej 80% z nich jest pozytywnych.
- **Wyjście**: Dane z nowymi cechami w `data/games_engineered.csv`.

### `05_data_validation.py` - Walidacja Danych

- **Funkcja**: Przeprowadza kompleksową walidację danych po inżynierii cech. Sprawdza braki, duplikaty, typy danych, wykrywa wartości odstające (outliers), weryfikuje zakresy wartości i analizuje rozkłady. Oblicza ogólny **Quality Score** dla datasetu.
- **Wyjście**: Raport `reports/02_validation_report.json`.

### `06_data_export.py` - Export i Przygotowanie Danych do ML

### `06a_data_validator.py` - Dodatkowa Walidacja i Analiza Danych

- **Funkcja**: Przeprowadza pogłębioną analizę danych po inżynierii cech. Generuje histogramy dla istotnych cech numerycznych, analizuje i wizualizuje korelacje między cechami (histogram rozkładu korelacji, heatmapa), identyfikuje i raportuje istotne korelacje oraz wykrywa i wizualizuje wartości odstające (outliers) za pomocą metody IQR.
- **Wyjście**: Raporty `reports/outlier_report_iqr.csv`, `reports/significant_correlations_report.csv` oraz wykresy w `reports/plots/` (np. `correlation_distribution_histogram.png`, `correlation_heatmap.png`, `histogram_*.png`, `boxplot_outliers_*.png`).

### `06_data_export.py` - Export i Przygotowanie Danych do ML

- **Funkcja**: Selekcjonuje 15 kluczowych cech do modelowania ML. Dzieli dane na zbiory treningowy i testowy (80/20). Eksportuje dane do formatów CSV i Parquet. Generuje również dodatkowe pliki:
    - CSV dla każdej grupy cech (np. `games_group_identifiers.csv`).
    - Pliki XLSX z filtrami na nagłówkach i zamrożonymi wierszami (dla całego datasetu, treningowego i testowego).
    - Plik XLSX z oddzielnymi arkuszami dla każdej grupy cech.
    - Dokumentację kolumn (`columns_documentation.csv`) i manifest datasetu (`dataset_manifest.json`).
- **Wyjście**: Wszystkie finalne pliki danych i dokumentacji w `data/processed/`. Raport `reports/03_export_summary.json`.

### `07_model_training.py` - Trenowanie Modeli ML

- **Funkcja**: Ładuje zbiory treningowy i testowy. Przygotowuje dane (usuwa zbędne kolumny, obsługuje braki). Definiuje i trenuje różne modele klasyfikacji (Regresja Logistyczna, Random Forest, LightGBM/XGBoost). Ewaluuje modele na zbiorze testowym za pomocą metryk takich jak ROC-AUC, F1-Score. Zapisuje najlepszy model na dysku.
- **Wyjście**: Zapisany model (`models/best_model.joblib`) oraz raport z treningu (`reports/04_model_training_report.json`).

### `08_model_evaluation.py` - Ewaluacja Modelu i Wizualizacje

- **Funkcja**: Ładuje najlepszy wytrenowany model i zbiór testowy. Generuje predykcje i prawdopodobieństwa. Tworzy szczegółowy raport klasyfikacji, macierz pomyłek, krzywe ROC i Precision-Recall. Rysuje ważność cech i, jeśli dostępne, przeprowadza analizę SHAP dla interpretowalności modelu.
- **Wyjście**: Raport z metryk (`reports/05_evaluation_metrics.json`) oraz wykresy w `reports/figures/`.

### `09_predict_cli.py` - Interaktywne Narzędzie CLI do Predykcji

- **Funkcja**: Interaktywne narzędzie wiersza poleceń, które pozwala użytkownikowi wprowadzić parametry nowej gry. Ładuje wytrenowany model i przewiduje, czy gra osiągnie wysoką ocenę. Generuje szczegółowy raport w formacie Markdown z wynikiem predykcji i rekomendacjami.
- **Wyjście**: Raporty predykcyjne w `reports/predictions/Raport_*.md`.

### `00_pipeline_test.py` - Orchestrator Pełnego Pipeline'u ML

- **Funkcja**: Jest to główny skrypt, który uruchamia wszystkie powyższe kroki (od eksploracji danych do ewaluacji modelu) w odpowiedniej kolejności. Monitoruje postęp, obsługuje błędy i generuje podsumowanie wykonania całego pipeline'u. Zawiera również funkcję sprawdzającą i instalującą zależności z `requirements.txt`.
- **Wyjście**: Kompleksowy log wykonania (`pipeline_execution.log`) oraz podsumowanie na konsoli.

## Jak Korzystać

### 1. Wymagania Wstępne

- **Python**: Wersja 3.8 lub nowsza.
- **pip**: Menedżer pakietów Pythona (zazwyczaj dołączony do Pythona).
- **Kaggle API Token**: Aby pobrać dane z Kaggle, musisz mieć skonfigurowany token API.
    1.  Przejdź do Kaggle.
    2.  Zaloguj się, a następnie przejdź do "Your Profile" -> "Account".
    3.  W sekcji "API" kliknij "Create New API Token". Spowoduje to pobranie pliku `kaggle.json`.
    4.  Przenieś `kaggle.json` do katalogu `C:\Users\<YourUsername>\.kaggle\` (Windows) lub `~/.kaggle/` (Linux/macOS).

### 2. Instalacja Zależności

Zainstaluj wszystkie wymagane biblioteki Pythona za pomocą pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Pobieranie Danych (Pierwsze Uruchomienie)

Przed uruchomieniem głównego pipeline'u, musisz pobrać surowe dane z Kaggle:

```bash
python 01_data_collection.py
```

### 4. Uruchomienie Pełnego Pipeline'u ML

Aby uruchomić wszystkie etapy przetwarzania danych, trenowania i ewaluacji modelu, użyj głównego orchestratora:

```bash
python 00_pipeline_test.py
```

Ten skrypt automatycznie sprawdzi i zainstaluje brakujące zależności, a następnie wykona wszystkie kroki od `02_data_exploration.py` do `08_model_evaluation.py` w sekwencji.

### 5. Uruchomienie Narzędzia Predykcyjnego CLI

Po pomyślnym zakończeniu pipeline'u i wytrenowaniu modelu, możesz użyć interaktywnego narzędzia do predykcji sukcesu nowych gier:

```bash
python 09_predict_cli.py
```

Narzędzie poprosi Cię o wprowadzenie parametrów gry i wygeneruje raport predykcyjny w formacie Markdown.

### 6. Uruchamianie Pojedynczych Skryptów (dla debugowania/specyficznych zadań)

Możesz również uruchamiać poszczególne skrypty niezależnie, na przykład w celu debugowania konkretnego etapu:

```bash
# Eksploracja
python 02_data_exploration.py

# Czyszczenie
python 03_data_cleaning.py

# Inżynieria Cech
python 04_feature_engineering.py

# Walidacja
python 05_data_validation.py

# Export
python 06_data_export.py

# Trenowanie Modeli
python 07_model_training.py

# Ewaluacja Modeli
python 08_model_evaluation.py
```

## Generowane Pliki i Wyniki

Po pomyślnym wykonaniu pełnego pipeline'u, w katalogach `data/`, `models/` i `reports/` znajdziesz następujące pliki:

### `data/`
- `games_YYYYMMDD_HHMMSS.csv`: Surowe dane pobrane z Kaggle.
- `games_cleaned.csv`: Dane po etapie czyszczenia.
- `games_engineered.csv`: Dane po etapie inżynierii cech.

### `data/processed/` (Finalne dane i dokumentacja)
- `games_final.csv`: Ostateczny dataset z 15 wybranymi cechami w formacie CSV.
- `games_final.parquet`: Ostateczny dataset w zoptymalizowanym formacie Parquet (znacznie mniejszy i szybszy do odczytu).
- `games_train.csv`: Zbiór treningowy (80% danych) w formacie CSV.
- `games_test.csv`: Zbiór testowy (20% danych) w formacie CSV.
- `columns_documentation.csv`: Dokumentacja wszystkich kolumn w finalnym datasetcie.
- `dataset_manifest.json`: Manifest zawierający metadane datasetu, statystyki i informacje o plikach.
- `dataset_documentation.md`: Plik README opisujący finalny dataset.
- `games_group_identifiers.csv`, `games_group_temporal.csv`, etc.: Pliki CSV zawierające podgrupy cech, ułatwiające analizę.
- `games_final_with_filters.xlsx`, `games_train_with_filters.xlsx`, `games_test_with_filters.xlsx`: Pliki Excel z danymi, filtrami na nagłówkach i zamrożonymi wierszami.
- `games_final_grouped.xlsx`: Plik Excel zawierający oddzielne arkusze dla każdej grupy cech.

### `models/`
- `best_model.joblib`: Zserializowany najlepszy model ML, gotowy do użycia w predykcjach.

### `reports/`
- `pipeline_execution.log`: Szczegółowy log z wykonania całego pipeline'u.
- `01_exploration_summary.json`: Podsumowanie eksploracji danych.
- `02_validation_report.json`: Raport z walidacji danych, w tym Quality Score.
- `03_export_summary.json`: Podsumowanie eksportu danych.
- `04_model_training_report.json`: Raport z treningu modeli, zawierający metryki i ważność cech.
- `05_evaluation_metrics.json`: Szczegółowe metryki ewaluacji najlepszego modelu (np. classification report).
- `outlier_report_iqr.csv`: Raport z wykrytymi outlierami w danych.
- `significant_correlations_report.csv`: Raport z parami cech o istotnych korelacjach.

### `reports/plots/`

### `reports/figures/`
- `confusion_matrix.png`: Wizualizacja macierzy pomyłek modelu.
- `roc_pr_curves.png`: Wykresy krzywych ROC i Precision-Recall.
- `feature_importance.png`: Wykres ważności cech dla najlepszego modelu.
- `shap_summary.png`: Wykres podsumowujący analizę SHAP (jeśli biblioteka SHAP jest zainstalowana).

### `reports/plots/`
- `correlation_distribution_histogram.png`: Histogram rozkładu wartości korelacji.
- `correlation_heatmap.png`: Heatmapa macierzy korelacji.
- `histogram_*.png`: Histogramy rozkładów dla poszczególnych cech numerycznych.
- `boxplot_outliers_*.png`: Boxploty wizualizujące outliery dla poszczególnych cech.

### `reports/predictions/`
- `Raport_*.md`: Raporty generowane przez narzędzie `09_predict_cli.py`.

## Finalne 15 Kolumn do Modelowania ML

Po inżynierii cech i selekcji, pipeline wybiera 15 najwartościowszych kolumn do modelowania:

| # | Kolumna | Typ | Opis |
|---|---------|-----|------|
| 1 | AppID | int64 | Unikalny identyfikator gry w Steam |
| 2 | Name | object | Nazwa gry |
| 3 | Genres | object | Gatunki gry (separator: przecinek) |
| 4 | Release_year | int64 | Rok wydania gry |
| 5 | Days_since_release | int64 | Liczba dni od wydania gry |
| 6 | Platform_count | int64 | Liczba platform (0-3: brak, tylko jedna, dwie, wszystkie) |
| 7 | Price | float64 | Cena gry w USD |
| 8 | Is_free | int64 | Czy gra jest darmowa (1=tak, 0=nie) |
| 9 | Total_reviews | int64 | Całkowita liczba recenzji (pozytywne + negatywne) |
| 10 | Review_ratio | float64 | Udział recenzji pozytywnych do całkowitych |
| 11 | Is_highly_rated | int64 | Czy gra ma wysoką ocenę (1=min. 30 recenzji i >=80% pozytywnych) |
| 12 | Log_owners | float64 | Log transformacja liczby oszacowanych właścicieli |
| 13 | Has_achievements | int64 | Czy gra posiada osiągnięcia (1=tak, 0=nie) |
| 14 | Log_total_reviews | float64 | Log transformacja całkowitej liczby recenzji |
| 15 | Genre_count | int64 | Liczba gatunków, do których należy gra |

## Quality Score - Szczegółowa Analiza

Quality Score jest obliczany w `05_data_validation.py` na podstawie algorytmu punktacji, który ocenia jakość danych.

**Punkt startowy:** 100.0

**Kryteria i kary (przykładowe):**

| Kryteria | Formuła | Przykładowa Kara |
|----------|---------|-------------------|
| **Brakujące wartości** | `min(10, missing_pct)` | -3.6 (dla 3.6% braków) |
| **Duplikaty AppID** | `-5 jeśli > 0` | 0.0 (jeśli brak duplikatów) |
| **Outliers** | `min(10, outlier_pct/10)` | -10.0 (dla >100% outlierów) |
| **Anomalie zakresu** | `min(5, invalid_count)` | -2.0 (dla 2 anomalii) |

**Interpretacja wyniku:**
- Wynik powyżej 80 jest zazwyczaj uważany za dobry.
- Kary za "outliers" mogą wynikać z natury danych (np. zmienne binarne, które są technicznie "outlierami" w rozkładzie ciągłym).

## Przykład Użycia w Pythonie

Po uruchomieniu pipeline'u, możesz łatwo załadować finalne dane do dalszej analizy lub modelowania:

```python
import pandas as pd

# Opcja 1: Załaduj CSV
df_csv = pd.read_csv('data/processed/games_final.csv')
print(f"Shape z CSV: {df_csv.shape}")
print(df_csv.head())

# Opcja 2: Szybciej - Załaduj Parquet
df_parquet = pd.read_parquet('data/processed/games_final.parquet')
print(f"Shape z Parquet: {df_parquet.shape}")
print(df_parquet.head())

# Załaduj już podzielone dane do treningu i testowania
train_df = pd.read_csv('data/processed/games_train.csv')
test_df = pd.read_csv('data/processed/games_test.csv')

print(f"Liczba wierszy w zbiorze treningowym: {len(train_df)}")
print(f"Liczba wierszy w zbiorze testowym: {len(test_df)}")

# Pracuj z dokumentacją kolumn
columns_doc = pd.read_csv('data/processed/columns_documentation.csv')
print("\nDokumentacja kolumn:")
print(columns_doc[['column_name', 'data_type', 'description']].head())

# Manifest z metadanymi
import json
with open('data/processed/dataset_manifest.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)
    print(f"\nCałkowita liczba rekordów: {manifest['total_records']}")
    print(f"Całkowita liczba cech: {manifest['total_features']}")
    print(f"Quality Score: {manifest['quality_metrics']['quality_score']:.1f}/100")
```

## Troubleshooting

### Problem: `ModuleNotFoundError` lub `ImportError`

**Opis**: Brakujące pakiety Pythona.
**Rozwiązanie**: Upewnij się, że wszystkie zależności są zainstalowane z `requirements.txt`. Główny orchestrator (`00_pipeline_test.py`) powinien to obsłużyć automatycznie, ale możesz to zrobić ręcznie:

```bash
pip install -r requirements.txt
```

### Problem: `FileNotFoundError` dla danych Kaggle

**Opis**: Skrypty nie mogą znaleźć pliku `games_*.csv` w katalogu `data/`.
**Rozwiązanie**: Upewnij się, że uruchomiłeś `01_data_collection.py` i że plik `kaggle.json` jest poprawnie skonfigurowany.

```bash
python 01_data_collection.py
```

### Problem: `PermissionError` lub "plik używany przez inny proces"

**Opis**: Plik CSV/Excel jest otwarty w innej aplikacji (np. Excel, edytor tekstu) i skrypt nie może go zapisać.
**Rozwiązanie**: Zamknij wszystkie aplikacje korzystające z plików w katalogach `data/` i `reports/`, a następnie spróbuj ponownie.

### Problem: Błędy kodowania (`UnicodeDecodeError`)

**Opis**: Problemy z odczytem/zapisem plików zawierających znaki specjalne (np. polskie diakrytyki).
**Rozwiązanie**: Wszystkie skrypty w pipeline'ie są skonfigurowane do używania kodowania UTF-8, co powinno rozwiązać większość problemów. Upewnij się, że Twoje środowisko systemowe również poprawnie obsługuje UTF-8.

### Monitorowanie Postępu

Podczas wykonania pipeline'u, logi są zapisywane do `pipeline_execution.log`. Możesz monitorować postęp i szukać błędów, używając narzędzi takich jak `tail` (Linux/macOS) lub otwierając plik w edytorze.

```bash
# Wyświetl ostatnie 50 linii logu
tail -f pipeline_execution.log

# Wyszukaj błędy
grep ERROR pipeline_execution.log
```

## Metadata

| Parametr | Wartość |
|----------|---------|
| **Dataset** | Steam Games (Kaggle) |
| **Oryginalne kolumny** | 39 |
| **Wybrane kolumny** | 18 |
| **Engineered features** | 61 |
| **Finalne kolumny (ML)** | 15 |
| **Wierszy** | ~122,611 |
| **Train/Test** | ~98,089 / ~24,522 (80/20) |
| **Quality Score** | ~84.4/100 |
| **Czas pipeline'u** | ~1-2 minuty (zależnie od sprzętu) |
| **Rozmiar finalnego CSV** | ~17 MB |
| **Rozmiar finalnego Parquet** | ~3.2 MB |
| **Brakujące wartości** | < 4% (głównie w kolumnach tekstowych) |
| **Duplikaty AppID** | 0 |
| **Python** | 3.8+ |
| **Ostatnia aktualizacja zbioru danych** | 2023-11-20 |
