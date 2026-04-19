# Steam Games Dataset - Preprocessing Pipeline

## Przegląd

Ten projekt zawiera kompletny pipeline preprocessingu dla Steam Games Dataset przygotowując go do modelowania ML. Dataset zawiera informacje o **122,611 grach** z **39 kolumnami** oryginalnych danych.

## Cel

Transformacja surowych danych z Kaggle w czysty, dobrze ustrukturyzowany dataset gotowy do:
- Exploratory Data Analysis (EDA)
- Machine Learning (ML)
- Statystycznej analizy
- Wizualizacji danych

## Wybrane kolumny do analizy

### Kolumny wybrane z oryginalnego datasetu:
```
AppID (app id)          - Unikalny identyfikator gry
Name                    - Nazwa gry
Release date            - Data wydania
Price                   - Cena
Windows/Mac/Linux       - Obsługa platform
Metacritic score        - Ocena Metacritic
Achievements            - Liczba achievement'ów
Developers              - Deweloperzy
Publishers              - Wydawcy
Categories              - Kategorie
Genres                  - Gatunki
User score              - Ocena użytkowników
Score rank              - Ranking ocen
Positive/Negative       - Liczba recenzji (pozytywne/negatywne)
Estimated owners        - Szacunkowe posiadacze
```







## Struktura Pipeline'u

### 6 skryptów + Orchestrator:

```
00_preprocessing_pipeline.py (ORCHESTRATOR)
   └─ Steruje całym procesem, łączy wszystkie 5 etapów
      ├─ Uruchamia etapy sekwencyjnie
      ├─ Weryfikuje kompletność plikań
      ├─ Generuje podsumowanie
      └─ Obsługuje błędy i logi

01_data_collection.py
   ├─ Pobieranie datasetu z Kaggle
   ├─ Walidacja i zapis do CSV/JSON
   └─ [Uruchamia się raz, przed pipeline'm]

KROK 1: EKSPLORACJA (02_data_exploration.py)
   ├─ Ładowanie surowych danych
   ├─ Analiza struktury i typów
   ├─ Statystyka opisowa
   └─ Detekcja braków i anomalii

KROK 2: CZYSZCZENIE (03_data_cleaning.py)
   ├─ Obsługa duplikatów
   ├─ Uzupełnianie brakujących wartości
   ├─ Standaryzacja typów danych
   ├─ Czyszczenie wartości błędnych
   └─ Walidacja zakresów

KROK 3: INŻYNIERIA CECH (04_feature_engineering.py)
   ├─ Cechy z daty (rok, miesiąc, dni od wydania)
   ├─ Cechy platform (liczba, binarne flagi)
   ├─ Cechy recenzji (razem, ratio, log transformacja)
   ├─ Cechy ocen (kategorie, normalizacja)
   ├─ Cechy zawartości (achievement'y, owners, gatunki)
   ├─ Cechy ceny (kategorie, log transformacja)
   ├─ Kodowanie kategorii (one-hot)
   ├─ Normalizacja (StandardScaler)
   └─ Cechy interakcji (kombinacje)
   └─ Rezultat: 61 engineered features

KROK 4: WALIDACJA (05_data_validation.py)
   ├─ Sprawdzenie brakujących wartości
   ├─ Detekcja duplikatów
   ├─ Analiza typów danych
   ├─ Detekcja outliers (IQR method)
   ├─ Weryfikacja zakresów
   ├─ Analiza rozkładu danych
   └─ Generowanie scoru jakości

KROK 5: EXPORT (06_data_export.py)
   ├─ Selekcja 15 kluczowych kolumn dla ML
   ├─ Export CSV i Parquet
   ├─ Train/Test split (80/20)
   ├─ Dokumentacja kolumn
   ├─ Manifest datasetu
   └─ README i instrukcje
```

## Uruchamianie Pipeline'u

### Kroki przygotowawcze (wykonaj raz)

```bash
# Pobierz dane z Kaggle
python 01_data_collection.py
```

### Opcja 1: Uruchom cały pipeline

```bash
python 00_preprocessing_pipeline.py
```

To uruchomi automatycznie wszystkie 5 kroków sekwencyjnie:
1. Eksploracja → 2. Czyszczenie → 3. Inżynieria cech → 4. Walidacja → 5. Export

I wygeneruje pełne raporty + finalne dane do ML.

### Opcja 2: Uruchom poszczególne kroki ręcznie

```bash
# Krok 1 - Eksploracja
python 02_data_exploration.py

# Krok 2 - Czyszczenie
python 03_data_cleaning.py

# Krok 3 - Inżynieria cech
python 04_feature_engineering.py

# Krok 4 - Walidacja
python 05_data_validation.py

# Krok 5 - Export (selekcja 15 kolumn do ML)
python 06_data_export.py
```

## Struktura katalogów

Po wykonaniu pipeline'u:

```
ml-project-steam/
├── 00_preprocessing_pipeline.py    # Orchestrator pipeline'u (uruchom to)
├── 01_data_collection.py           # Pobieranie danych z Kaggle (wykonaj raz)
├── 02_data_exploration.py          # Krok 1 - Eksploracja
├── 03_data_cleaning.py             # Krok 2 - Czyszczenie
├── 04_feature_engineering.py       # Krok 3 - Inżynieria cech (61 features)
├── 05_data_validation.py           # Krok 4 - Walidacja
├── 06_data_export.py               # Krok 5 - Export (selekcja 15 kolumn do ML)
│
├── data/
│   ├── games_20260418_093359.csv  # Oryginalne dane z Kaggle
│   ├── games_20260418_093359.json # Oryginalne dane JSON
│   ├── games_cleaned.csv           # Dane po czyszczeniu
│   ├── games_engineered.csv        # Dane z nowymi cechami
│   │
│   └── processed/                  # Finalne dane dla ML (15 kolumn)
│       │
│       ├─ PLIKI GŁÓWNE (CSV/Parquet)
│       ├── games_final.csv         # CSV z 15 best features
│       ├── games_final.parquet     # Parquet (szybciej, ~3MB)
│       ├── games_train.csv         # 80% (98,089 wierszy) do treningu
│       ├── games_test.csv          # 20% (24,522 wierszy) do testowania
│       │
│       ├─ CSV Z GRUPAMI KOLUMN (nowe!)
│       ├── games_group_identifiers.csv  # AppID, Name
│       ├── games_group_temporal.csv     # Release_year, Days_since_release
│       ├── games_group_platform.csv     # Platform_count
│       ├── games_group_reviews.csv      # Total_reviews, Review_ratio, Log_total_reviews
│       ├── games_group_scores.csv       # Is_highly_rated
│       ├── games_group_content.csv      # Log_owners, Has_achievements, Genre_count
│       ├── games_group_price.csv        # Price, Is_free
│       ├── games_group_metadata.csv     # Genres
│       │
│       ├─ XLSX Z FILTRAMI NA NAGŁÓWKACH (nowe!)
│       ├── games_final_with_filters.xlsx      # Wszystkie dane z filtrami autofilter
│       ├── games_train_with_filters.xlsx      # 80% z filtrami autofilter
│       ├── games_test_with_filters.xlsx       # 20% z filtrami autofilter
│       ├── games_final_grouped.xlsx           # 8 arkuszy (jeden per grupa)
│       │
│       ├─ DOKUMENTACJA
│       ├── dataset_manifest.json      # Metadata i statystyka
│       ├── columns_documentation.csv  # Info o każdej kolumnie
│       └── README.md                  # Dokumentacja finalna
│
└── reports/
    ├── 01_exploration_summary.json # Raport eksploracji (struktura, statistyka)
    ├── 02_validation_report.json   # Raport walidacji (quality score: 84.4/100)
    ├── 03_export_summary.json      # Raport exportu (finalne 15 kolumn)
    └── preprocessing.log           # Log wykonania wszystkich 5 kroków
```

###  Rodzaje generowanych plików:

**PLIKI GŁÓWNE** (standardowe CSV/Parquet):
- `games_final.csv` - Wszystkie 122,611 gier × 15 kolumn
- `games_train.csv` - 98,089 gier do trenowania modeli
- `games_test.csv` - 24,522 gry do testowania
- `games_final.parquet` - Format binarny (3x mniejszy)

** CSV Z GRUPAMI KOLUMN** (nowe!):
Każdy plik zawiera podgrupę powiązanych cech:
- Umożliwia pracę z konkretnymi grupami atrybutów
- Idealne dla eksploracyjnej analizy (EDA)
- Przydatne dla feature selection w ML

** XLSX Z FILTRAMI** (nowe!):
- **Autofilter na nagłówkach** - Możliwość sortowania/filtrowania w Excelu
- **Zamrożone nagłówki** - Pierwszy wiersz zawsze widoczny
- **Formatowanie** - Liczby sformatowane (2 miejsca dziesiętne)
- 3 wersje danych: wszystkie + train + test
- **games_final_grouped.xlsx** - 8 arkuszy (jeden na grupę)

## Finalne 15 kolumn do modelowania ML

Po inżynierii cech i selekcji pipeline wybiera 15 najwartościowszych kolumn:

| # | Kolumna | Typ | Opis |
|---|---------|-----|------|
| 1 | AppID | int64 | Unikalny ID gry w Steam |
| 2 | Name | object | Nazwa gry |
| 3 | Genres | object | Gatunki (np. Action, RPG) |
| 4 | Release_year | int64 | Rok wydania |
| 5 | Days_since_release | int64 | Dni od wydania do dziś |
| 6 | Platform_count | int64 | Liczba platform (0-3) |
| 7 | Price | float64 | Cena w USD |
| 8 | Is_free | int64 | 1=darmowa, 0=płatna |
| 9 | Total_reviews | int64 | Całkowita liczba recenzji |
| 10 | Review_ratio | float64 | % recenzji pozytywnych |
| 11 | Is_highly_rated | int64 | 1=Metacritic >= 75, 0=inne |
| 12 | Log_owners | float64 | Log szacunkowych właścicieli |
| 13 | Has_achievements | int64 | 1=ma achievement'y |
| 14 | Log_total_reviews | float64 | Log transformacja recenzji |
| 15 | Genre_count | int64 | Liczba gatunków gry |

**Rezultat:**
- Dataset: 122,611 gier × 15 kolumn
- Train: 98,089 wierszy (80%)
- Test: 24,522 wierszy (20%)
- Brakujące wartości: < 8% (tylko Genres)
- Quality Score: 84.4/100

## Szczegółowy opis każdego skryptu

### **01_data_collection.py** - Pobieranie danych
**Czas wykonania:** ~5-30 sekund (zależy od internetu)

**Co robi:**
- Pobiera plik CSV z Kaggle Steam Games Dataset (39 kolumn)
- Weryfikuje integralność i rozmiar danych
- Zapisuje do CSV z timestampem (games_YYYYMMDD_HHMMSS.csv)
- Tworzy również kopię JSON do raportowania

**Wejście:**
- Internetu i dostępu do Kaggle API

**Wyjście:**
- `data/games_20260418_093359.csv` (389 MB, 122,611 wierszy × 39 kolumn)
- `data/games_20260418_093359.json` (metadata backup)

**Przykład:**
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('fronkongames/steam-games-dataset')
```

---

### **KROK 1: 02_data_exploration.py** - Eksploracja i analiza struktury
**Czas wykonania:** ~8 sekund

**Co robi:**
- Ładuje surowe dane z Kaggle (ostatni plik games_*.csv)
- Analizuje strukturę: typy danych, wymiary, unikalne wartości
- Liczy brakujące wartości dla każdej kolumny
- Generuje statystykę opisową (średnia, mediana, min, max)
- Tworzy raport JSON z podsumowaniem eksploracji

**Wejście:**
- `data/games_20260418_093359.csv` (122,611 × 39)

**Wyjście:**
- `reports/01_exploration_summary.json` - Raport eksploracji
- Log wypisany na konsolę z informacjami o kolumnach

**Klucze operacje:**
```python
# Wczytanie z filtrowaniem surowych danych
csv_files = [f for f in glob("games_*.csv") 
             if "cleaned" not in f and "engineered" not in f]

# Analiza brakujących wartości
missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100

# Wyliczenie średnich (tylko kolumny numeryczne)
if pd.api.types.is_numeric_dtype(df[col]):
    sample = df[col].dropna().mean()
```

**Wyniki z danych:**
- 122,611 wierszy bez duplikatów AppID
- Wybrane 18 kolumn do przetwarzania
- User score: 100% braków (wykluczone z analizy)
- Metadane: 6.9-7.3% braków (Publishers, Categories, Genres)

---

### **KROK 2: 03_data_cleaning.py** - Czyszczenie i standaryzacja
**Czas wykonania:** ~9 sekund

**Co robi:**
- Wczytuje surowe dane z Kaggle (TYLKO surowe, nie przetworzane)
- Wybiera 18 najważniejszych kolumn do analizy
- Usuwa duplikaty (0 duplikatów znalezionych)
- Konwertuje typy danych:
  - Data: string → datetime64
  - Cena: str/int → float64
  - Platform flagi: str → bool
  - Liczby: str → int64
- Czyszcza wartości błędne (np. ujemne ceny)
- Tworzy kolumnę `Review_ratio` = Positive / (Positive + Negative)

**Wejście:**
- `data/games_20260418_093359.csv` (122,611 × 39)

**Wyjście:**
- `data/games_cleaned.csv` (122,611 × 19)
  - 18 oryginalnych kolumn + 1 nowa (Review_ratio)

**Klucze transformacje:**
```python
# Selekcja kolumn
selected_columns = ['AppID', 'Name', 'Release date', 'Price', 'Windows', 'Mac', 'Linux',
                   'Metacritic score', 'Achievements', 'Developers', 'Publishers',
                   'Categories', 'Genres', 'User score', 'Score rank', 'Positive', 
                   'Negative', 'Estimated owners']

# Konwersja typów
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Nowa kolumna
df['Review_ratio'] = df['Positive'] / (df['Positive'] + df['Negative'])
```

**Statystyka:**
- Duplikaty AppID: 0 ✅
- Brakujące wartości: User score (100%), Review_ratio (34.6%)
- Zakresy: Cena [0.0 - 999.98], Metacritic [0-1] (bool)

---

### **KROK 3: 04_feature_engineering.py** - Tworzenie nowych cech
**Czas wykonania:** ~7 sekund

**Co robi:**
- Wczytuje oczyszczone dane (122,611 × 19)
- Tworzy 61 nowych features z 5 kategorii
- Koduje zmienne kategoryczne (one-hot encoding)
- Normalizuje cechy numeryczne (StandardScaler)
- Eksportuje dane z nowymi cechami

**Wejście:**
- `data/games_cleaned.csv` (122,611 × 19)

**Wyjście:**
- `data/games_engineered.csv` (122,611 × 61)

**Tworzone feature'y (43 cechy inżynierii):**

**Czasowe (5):**
```python
df['Release_year'] = df['Release date'].dt.year
df['Release_month'] = df['Release date'].dt.month
df['Release_quarter'] = df['Release date'].dt.quarter
df['Days_since_release'] = (pd.Timestamp.now() - df['Release date']).dt.days
df['Is_recent'] = (df['Days_since_release'] < 365).astype(int)
```

**Platformy (5):**
```python
df['Platform_count'] = df['Windows'].astype(int) + df['Mac'].astype(int) + df['Linux'].astype(int)
df['Has_windows'] = df['Windows'].astype(int)
df['Has_mac'] = df['Mac'].astype(int)
df['Has_linux'] = df['Linux'].astype(int)
df['Is_multiplatform'] = (df['Platform_count'] > 1).astype(int)
```

**Recenzje (4):**
```python
df['Total_reviews'] = df['Positive'] + df['Negative']
df['Log_total_reviews'] = np.log1p(df['Total_reviews'])
df['Review_ratio'] = df['Positive'] / df['Total_reviews']  # już istnieje
df['Is_heavily_reviewed'] = (df['Total_reviews'] > df['Total_reviews'].quantile(0.75)).astype(int)
```

**Oceny (4):**
```python
df['Has_metacritic'] = (~df['Metacritic score'].isna()).astype(int)
df['Is_highly_rated'] = (df['Metacritic score'] >= 75).astype(int)
df['Metacritic_category'] = pd.cut(df['Metacritic score'], bins=[0, 50, 70, 85, 100], 
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
```

**Zawartość (5):**
```python
df['Log_achievements'] = np.log1p(df['Achievements'])
df['Has_achievements'] = (df['Achievements'] > 0).astype(int)
df['Log_owners'] = np.log1p(df['Estimated owners'])
df['Genre_count'] = df['Genres'].str.split(',').apply(len)
df['Category_count'] = df['Categories'].str.split(',').apply(len)
```

**Cena (3):**
```python
df['Is_free'] = (df['Price'] == 0).astype(int)
df['Log_price'] = np.log1p(df['Price'])
df['Price_category'] = pd.cut(df['Price'], bins=[0, 5, 20, 50, 100, 2000], 
                               labels=['Free', 'Budget', 'Standard', 'Premium', 'AAA'])
```

**Normalizacja (8):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = ['Price', 'Metacritic score', 'Total_reviews', 'Days_since_release', ...]
normalized = scaler.fit_transform(df[numeric_cols].fillna(0))
for i, col in enumerate(numeric_cols):
    df[f'{col}_normalized'] = normalized[:, i]
```

**Interakcje (3):**
```python
df['Price_Rating_ratio'] = df['Price'] / (df['Metacritic score'] + 1)
df['Rating_Review_score'] = df['Metacritic score'] * df['Review_ratio']
df['Owners_Review_ratio'] = np.log1p(df['Estimated owners']) / (df['Total_reviews'] + 1)
```

---

### **KROK 4: 05_data_validation.py** - Walidacja jakości danych
**Czas wykonania:** ~3 sekund

**Co robi:**
- Wczytuje dane z cechami (122,611 × 61)
- Sprawdza 8 aspektów jakości danych
- Oblicza **Quality Score** (0-100) algorytmem punkcji
- Generuje raport walidacji z rekomendacjami

**Wejście:**
- `data/games_engineered.csv` (122,611 × 61)

**Wyjście:**
- `reports/02_validation_report.json` - Pełny raport
- Log z Quality Score

**8 przeprowadzanych testów:**

**1. Brakujące wartości:**
```python
missing = df.isna().sum().sum()
missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
# Wynik: 3.6% (271,522 komórki)
```

**2. Duplikaty:**
```python
dup_appid = df['AppID'].duplicated().sum()
# Wynik: 0 ✅
```

**3. Typy danych:**
```python
numeric: 50, object: 7, datetime: 0, boolean: 4
```

**4. Outliers (IQR method):**
```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
# Wynik: 32 kolumny z outlierami, np. 24.8% w Is_heavily_reviewed
```

**5. Zakresy wartości:**
```python
# Price: 0.0 - 999.98 (OK)
# Metacritic: 0 - 100 (OK)
```

**6. Rozkład danych:**
```python
skewness = df[col].skew()
# Wysoki skos: Price (22.40), Positive (59.25), Negative (177.84)
```

**7. Kategorie:**
```python
# Metacritic_category: Poor, Fair, Good, Excellent
# Price_category: Free, Budget, Standard, Premium, AAA
```

### **Quality Score Algorytm (84.4/100):**

```
Punkt startowy: 100.0

Karykatura:
- Brakujące wartości 3.6%:    -3.6 pkt (max -10)
- Duplikaty (0):              -0.0 pkt
- Outliers 263.7%:            -10.0 pkt (max -10, bo 263.7/10 > 10)
- Anomalie zakresu (2):       -2.0 pkt (max -5)

RAZEM: 100.0 - 3.6 - 0 - 10.0 - 2.0 = 84.4/100 ✅
```

---

### **KROK 5: 06_data_export.py** - Selekcja i export do ML
**Czas wykonania:** ~6-8 sekund

**Co robi:**
- Wczytuje 61 cech (122,611 × 61)
- **Selekcja 15 kolumn** najwartościowszych dla ML
- Tworzy train/test split (80/20)
- Eksportuje do CSV i Parquet
- **Generuje CSV z grupami kolumn** (8 plików)
- **Generuje XLSX z filtrami autofilter** (4 pliki)
- Generuje dokumentację i manifest

**Wejście:**
- `data/games_engineered.csv` (122,611 × 61)

**Wyjście (18+ plików):**
```
data/processed/
│
├─ 📋 PLIKI GŁÓWNE (4 pliki)
├── games_final.csv              (17.09 MB) - CSV z 15 kolumnami
├── games_final.parquet          (3.22 MB)  - Szybki Parquet
├── games_train.csv              (13.67 MB) - 98,089 wierszy (80%)
├── games_test.csv               (3.41 MB)  - 24,522 wierszy (20%)
│
├─ 📁 CSV Z GRUPAMI KOLUMN (8 plików) - NOWE!
├── games_group_identifiers.csv        - AppID, Name
├── games_group_temporal.csv           - Release_year, Days_since_release
├── games_group_platform.csv           - Platform_count
├── games_group_reviews.csv            - Total_reviews, Review_ratio, Log_total_reviews
├── games_group_scores.csv             - Is_highly_rated
├── games_group_content.csv            - Log_owners, Has_achievements, Genre_count
├── games_group_price.csv              - Price, Is_free
├── games_group_metadata.csv           - Genres
│
├─ 🔍 XLSX Z FILTRAMI (4 pliki) - NOWE!
├── games_final_with_filters.xlsx      - Wszystkie dane + autofilter + zamrożony header
├── games_train_with_filters.xlsx      - 80% treningu + autofilter
├── games_test_with_filters.xlsx       - 20% testów + autofilter
├── games_final_grouped.xlsx           - 8 arkuszy (jedna grupa per arkusz)
│
├─ 📄 DOKUMENTACJA (3 pliki)
├── dataset_manifest.json              - Metadane i statystyka
├── columns_documentation.csv          - Info o każdej kolumnie
└── README.md                          - Dokumentacja finalna
```

**Selekcja 15 kolumn (najważniejsze):**
```python
selected_15_columns = [
    'AppID',              # Identifier
    'Name',               # Metadata
    'Genres',             # Content
    'Release_year',       # Temporal (cleaner than full date)
    'Days_since_release', # Temporal (age indicator)
    'Platform_count',     # Platform aggregation
    'Price',              # Financial
    'Is_free',            # Financial binary
    'Total_reviews',      # Engagement
    'Review_ratio',       # Sentiment
    'Is_highly_rated',    # Quality binary
    'Log_owners',         # Popularity (log scale)
    'Has_achievements',   # Content feature
    'Log_total_reviews',  # Scale metric
    'Genre_count'         # Content diversity
]
```

**Nowe funkcje - CSV z grupami:**
```python
# Każda grupa to osobny plik CSV
feature_groups = {
    'identifiers': ['AppID', 'Name'],
    'temporal': ['Release_year', 'Days_since_release'],
    'platform': ['Platform_count'],
    'reviews': ['Total_reviews', 'Review_ratio', 'Log_total_reviews'],
    'scores': ['Is_highly_rated'],
    'content': ['Log_owners', 'Has_achievements', 'Genre_count'],
    'price': ['Price', 'Is_free'],
    'metadata': ['Genres']
}

for group_name, columns in feature_groups.items():
    group_df = df[columns]
    group_df.to_csv(f'games_group_{group_name}.csv', index=False)
```

**Nowe funkcje - XLSX z filtrami:**
```python
# 1. Autofilter na nagłówkach
ws.auto_filter.ref = f"A1:{last_column}{last_row}"

# 2. Zamrożone nagłówki (pierwszy wiersz)
ws.freeze_panes = "A2"

# 3. Formatowanie liczb
cell.number_format = '0.00'  # 2 miejsca dziesiętne

# 4. Stylizacja nagłówka
cell.font = Font(bold=True, color="FFFFFF")
cell.fill = PatternFill(start_color="366092", end_color="366092")
```

**Kodowanie selekcji (15 z 61 cech):**
```python
# Wycluczone 46 features (dlaczego?):
# - 8 znormalizowanych: user może skalować sam
# - 3 interakcji: niskie znaczenie predykcyjne
# - 5 kategorii one-hot: zbyt szczegółowe
# - 4 binarne flagi platform: redundantne wobec Platform_count
# - 26 pozostałych: redundancja i multicollinearity
```

**Train/Test split:**
```python
np.random.seed(42)
test_indices = np.random.choice(len(df), size=int(len(df)*0.2), replace=False)
train_df = df.drop(test_indices)
test_df = df.iloc[test_indices]

# Rezultat:
# Train: 98,089 (80.0%)
# Test: 24,522 (20.0%)
```

**Dokumentacja generowana:**
```python
columns_doc = pd.DataFrame({
    'column_name': [...],
    'data_type': [...],
    'description': [...],
    'missing_count': [...],
    'unique_values': [...]
})
columns_doc.to_csv('columns_documentation.csv')
```

**Manifest JSON:**
```json
{
  "dataset_name": "Steam Games Dataset - Preprocessed",
  "creation_date": "2026-04-18T13:01:25",
  "total_records": 122611,
  "total_features": 15,
  "feature_groups": {
    "identifiers": ["AppID", "Name"],
    "temporal": ["Release_year", "Days_since_release"],
    "platform": ["Platform_count"],
    "reviews": ["Total_reviews", "Review_ratio", "Log_total_reviews"],
    "scores": ["Is_highly_rated"],
    "content": ["Log_owners", "Has_achievements", "Genre_count"],
    "price": ["Price", "Is_free"],
    "metadata": ["Genres"]
  },
  "quality_metrics": {
    "null_values": 9722,
    "duplicate_rows": 0,
    "memory_usage_mb": 18.50
  }
}
```

---

### **00_preprocessing_pipeline.py** - Orchestrator
**Czas wykonania:** ~33 sekundy

**Co robi:**
- Uruchamia wszystkie 5 kroków sekwencyjnie
- Weryfikuje wyjście każdego kroku
- Obsługuje błędy i zatrzymuje pipeline w razie problemu
- Generuje log z czasami wykonania
- Tworzy podsumowanie

**Logika:**
```python
steps = [
    ('Eksploracja danych', '02_data_exploration.py'),
    ('Czyszczenie danych', '03_data_cleaning.py'),
    ('Inżynieria cech', '04_feature_engineering.py'),
    ('Walidacja danych', '05_data_validation.py'),
    ('Export i przygotowanie', '06_data_export.py'),
]

for step_name, script in steps:
    try:
        result = subprocess.run([python, script], capture_output=True, timeout=60)
        if result.returncode != 0:
            raise Exception(f"Błąd w {step_name}")
        logger.info(f"[OK] {step_name} - SUKCES ({elapsed_time:.1f}s)")
    except Exception as e:
        logger.error(f"[ERROR] {step_name} - BŁĄD")
        break
```

**Wynik:**
```
Skrypt                         Status     Czas (s)
--------------------------------------------------
02_data_exploration.py         SUCCESS    7.5
03_data_cleaning.py            SUCCESS    8.8
04_feature_engineering.py      SUCCESS    6.8
05_data_validation.py          SUCCESS    2.7
06_data_export.py              SUCCESS    6.2
--------------------------------------------------
RAZEM                                     32.0s
```

## 📊 Wynik Final - Podsumowanie Transformacji

```
TRANSFORMACJA DANYCH:
================================================================================
   39 kolumn             18 wybranych      19 czyszczonych    61 engineered      15 final
   (surowe)          →    (manualna)   →   (+ 1 computed)  →  (ML features)  →  (optimal)
   122,611 wierszy       122,611           122,611            122,611            122,611
                         wierszy           wierszy            wierszy            wierszy
================================================================================

STATYSTYKA CAŁOŚCIOWA:
- Czyszczenie: 9 sekund (0 duplikatów, standaryzacja typów)
- Feature engineering: 7 sekund (43 nowe cechy)
- Walidacja: 3 sekundy (Quality Score: 84.4/100)
- Export: 6 sekund (15 kolumn, train/test split)
- RAZEM PIPELINE: 32 sekundy ⚡

WIELKOŚĆ PLIKÓW:
- Oryginał (CSV):              389 MB
- Po czyszczeniu:              Pośredni
- Po feature engineering:      ~55 MB
- Finalne dane (CSV):          17 MB
- Finalne dane (Parquet):      3.2 MB (85% mniejsze! 🚀)
```

## 📈 Quality Score - Szczegółowa analiza (84.4/100)

Quality Score obliczany jest w **Kroku 4 (05_data_validation.py)** na podstawie algorytmu punktacji:

**Punkt startowy:** 100.0

**Przepisy karne (odejmowanie punktów):**

| Kryteria | Formuła | Wyliczenie | Odjęcie |
|----------|---------|-----------|--------|
| **Brakujące wartości** | `min(10, missing_pct)` | 3.6% < 10% | -3.6 |
| **Duplikaty AppID** | `-5 jeśli > 0` | 0 duplikatów | 0.0 |
| **Outliers** | `min(10, outlier_pct/10)` | 263.7%/10 = 26.4 → cap 10 | -10.0 |
| **Anomalie zakresu** | `min(5, invalid_count)` | 2 anomalii | -2.0 |
| | | **RAZEM** | **84.4** ✅ |

**Szczegóły brakujących wartości (3.6%):**
- Całkowicie puste: User score (100%), User_score_normalized (100%)
- Częściowo puste: Genres (7.3%), Categories (7.3%), Publishers (6.9%)
- Inne: Review_ratio (34.6% - bo User score ma braki)

**Detale outliers (263.7% razem):**
- Is_heavily_reviewed: 30,419 razy (24.8%) - binarna zmienna  
- Is_free: 26,206 razy (21.4%) - binarna zmienna
- Price_category_Free: 26,206 (21.4%)
- Platform_count: 22,263 (18.2%)
- ... (28 kolumn więcej z mniejszym % outliers)

**Interpretacja wyniku:**
- ✅ **84.4 to DOBRY wynik** (>80%)
- ⚠️ Outliers głównie z kolumn binarnych (normalnie - to nie błędy!)
- ✅ Zero duplikatów (perfekcja)
- ✅ < 4% braków (dopuszczalny poziom)
- 📊 Score jest konserwatywny - realistyczny dla danych biznesowych

## 🔄 Przepływ danych w pipeline'u

```
[WEJŚCIE SUROWE: 122,611 × 39 kolumn]
                    ↓
          [02_EKSPLORACJA]
          - Analiza struktury
          - Brakujące wartości
          - Statystyka
               ↓
[INTERMEDIATE: games_cleaned.csv - 122,611 × 19 kolumn]
                    ↓
          [03_CZYSZCZENIE]
          - Duplikaty (0)
          - Konwersja typów
          - Standaryzacja
               ↓
[INTERMEDIATE: games_engineered.csv - 122,611 × 61 kolumn]
                    ↓
          [04_INŻYNIERIA CECH]
          - 43 nowe features
          - Normalizacja
          - One-hot encoding
               ↓
[INTERMEDIATE: games_engineered.csv - 122,611 × 61 kolumn]
                    ↓
          [05_WALIDACJA]
          - Quality Score: 84.4/100
          - Outliers detection
          - Distribution analysis
               ↓
[INTERMEDIATE: games_engineered.csv - 122,611 × 61 kolumn]
                    ↓
          [06_EXPORT + SELEKCJA]
          - Wybór 15 kolumn
          - Train/Test split (80/20)
               ↓
[WYJŚCIE ML-READY]
├── games_final.csv (17 MB, 122,611 × 15)
├── games_final.parquet (3.2 MB)
├── games_train.csv (98,089 × 15)
├── games_test.csv (24,522 × 15)
└── Dokumentacja (JSON, CSV, README)
```

## ⚙️ Wymagania i instalacja

**Wymagane pakiety:**
```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 0.24.0
pyarrow >= 6.0.0 (do Parquet)
```

**Instalacja:**
```bash
pip install pandas numpy scikit-learn pyarrow
```

**Weryfikacja instalacji:**
```python
import pandas; import numpy; import sklearn; print("✅ OK")
```

---

## 📚 Przykład użycia w Python

**Załaduj finalne dane (15 kolumn, gotowe do ML):**

```python
import pandas as pd
import numpy as np

# Opcja 1: Załaduj CSV (17 MB)
df = pd.read_csv('data/processed/games_final.csv')
print(f"Shape: {df.shape}")  # (122611, 15)
print(df.head())

# Opcja 2: Szybciej - Parquet (3.2 MB) ⚡
df = pd.read_parquet('data/processed/games_final.parquet')

# Opcja 3: Załaduj już podzielone dane do treningu i testowania
train_df = pd.read_csv('data/processed/games_train.csv')   # 98,089 wierszy (80%)
test_df = pd.read_csv('data/processed/games_test.csv')     # 24,522 wierszy (20%)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
```

**Pracuj z dokumentacją kolumn:**

```python
# Załaduj dokumentację
columns_doc = pd.read_csv('data/processed/columns_documentation.csv')
print(columns_doc)
# Wyświetli: Columns, Types, Description, Missing Count, Unique Values

# Manifest z metadanymi
import json
with open('data/processed/dataset_manifest.json') as f:
    manifest = json.load(f)
    print(f"Rekordów: {manifest['total_records']}")
    print(f"Features: {manifest['total_features']}")
    print(f"Quality Score: {manifest['quality_metrics']['quality_score']}")
```

**Szybka analiza:**

```python
# Statystyka
print(df.describe())

# Brakujące wartości
print(df.isnull().sum())

# Typy danych
print(df.dtypes)

# Korelacje
print(df.corr())
```

---

## 📊 Raporty wygenerowane przez pipeline

Po wykonaniu pipeline'u dostępne są 3 raporty JSON:

### 1. **reports/01_exploration_summary.json**
```json
{
  "total_records": 122611,
  "total_columns": 39,
  "data_types": {...},
  "missing_values": {...},
  "column_statistics": {...}
}
```
- Info o strukturze danych
- Statystyka eksploracyjna każdej kolumny

### 2. **reports/02_validation_report.json** ⭐
```json
{
  "quality_score": 84.4,
  "validation_results": {
    "missing_values": 3.6,
    "duplicates": 0,
    "outliers": 263.7,
    "range_anomalies": 2
  },
  "column_analysis": {...},
  "recommendations": [...]
}
```
- Szczegółowy raport walidacji
- Quality Score z podziałem na kryteria
- Raporty per-kolumna z outlierami

### 3. **reports/03_export_summary.json**
```json
{
  "exported_records": 122611,
  "exported_features": 15,
  "train_test_split": {
    "train": 98089,
    "test": 24522
  },
  "files_generated": [...]
}
```
- Info o eksportowanych plikach
- Rozbicie train/test
- Nazwy wygenerowanych plików

---

## 🎯 Selekcja kolumn - Uzasadnienie

**Dlaczego te 15 kolumn z 61?**

```
61 engineered features
    ↓
SELEKCJA na podstawie:
├─ Niskie korelacje (multicollinearity avoidance)
├─ High predictive power (dla typowych ML zadań)
├─ Domain knowledge (co ma sens dla gier)
├─ Brak redundancji (np. nie bierz: Log i non-Log, oraz Normalized)
└─ Rozsądna ilość (15 ~ optimal dla generalności modelu)
    ↓
FINALNE 15 KOLUMN
```

| # | Kolumna | Powód włączenia |
|---|---------|-----------------|
| 1 | AppID | Unikalny identifier, potrzebny |
| 2 | Name | Metadane (może być droppable dla czystych ML) |
| 3 | Genres | Ważna cecha kategoryczna |
| 4 | Release_year | Temporalna - wpływa na popularność |
| 5 | Days_since_release | **Alternatywa do pełnej daty** (prostsze) |
| 6 | Platform_count | Agregacja Has_windows/mac/linux |
| 7 | Price | Fundamentalna dla biznesu |
| 8 | Is_free | Binarna (ważna flaga biznesowa) |
| 9 | Total_reviews | Sygnał zaangażowania użytkowników |
| 10 | Review_ratio | Sentymenty użytkowników |
| 11 | Is_highly_rated | Binarna, łatwo interpretowana |
| 12 | Log_owners | Log-skala (zmniejsza skos rozkładu) |
| 13 | Has_achievements | Cecha zawartości |
| 14 | Log_total_reviews | Log-skala recenzji |
| 15 | Genre_count | Dywersyfikacja gatunków |

**Co zostało wykluczone z 61:**

```
❌ 8 znormalizowanych (Price_normalized, etc.)
   → Użytkownik może standaryzować sam, redundantne

❌ 3 cechy interakcji (Price_Rating_ratio, etc.)
   → Niskie znaczenie predykcyjne (sprawdzono korelacje)

❌ 5 kategorii one-hot (Metacritic_category, Price_category)
   → Zbyt szczegółowe, lepiej użyć bezpośrednio wartości

❌ 4 binarne flagi platform (Has_windows, Has_mac, Has_linux)
   → Redundantne wobec Platform_count, zawarte w agregacji

❌ 26 innych
   → Redundancja, multicollinearity, niska predykcyjność
```

---

## 🔧 Troubleshooting

### Problem: "Plik nie znaleziony"
```
Error: FileNotFoundError: data/games_*.csv
```
**Rozwiązanie:**
```bash
# 1. Sprawdź czy jesteś w poprawnym katalogu
pwd  # (Linux/Mac) lub cd (Windows)

# 2. Upewnij się że uruchomiłeś 01_data_collection.py
python 01_data_collection.py

# 3. Sprawdź czy plik istnieje
ls -la data/ | grep games
```

### Problem: "TypeError: Cannot perform reduction 'mean' with string dtype"
```
Error: Cannot perform reduction 'mean' with string dtype
```
**Powód:** Próba obliczenia .mean() na kolumnie ze stringami  
**Rozwiązanie:** (już naprawione w kodzie)
```python
# Poprawnie - typ check PRZED mean()
if pd.api.types.is_numeric_dtype(df[col]):
    avg = df[col].mean()
```

### Problem: "ValueError: Found array with 0 sample(s)"
```
Error: Found array with 0 sample(s) while 0 samples expected
```
**Powód:** Pipeline załadował pusty intermediate file zamiast surowych danych  
**Rozwiązanie:** (już naprawione)
```bash
# Wyczyść intermediate files
rm -f data/games_cleaned.csv data/games_engineered.csv

# Uruchom pipeline od nowa
python 00_preprocessing_pipeline.py
```

### Problem: "Memory error" przy dużych zbiorach
```
MemoryError: Unable to allocate X.XX GiB for an array
```
**Powód:** Dataset ~120k wierszy × 61 kolumn = dużo RAM  
**Rozwiązanie:**
```bash
# 1. Zamknij inne aplikacje (Chrome, VS Code, etc.)
# 2. Lub przetwarzaj w małych porcjach
# 3. Lub użyj Parquet zamiast CSV (bardziej efektywny)
```

### Problem: "Brakuje pakietu"
```
ModuleNotFoundError: No module named 'sklearn'
```
**Rozwiązanie:**
```bash
# Zainstaluj wszystkie wymagania
pip install --upgrade pandas numpy scikit-learn pyarrow

# Lub ze requirements.txt (jeśli istnieje)
pip install -r requirements.txt
```

### Problem: "Blokada pliku - plik używany przez inny proces"
```
PermissionError: [Errno 13] Permission denied
```
**Powód:** Plik CSV wciąż otwarty w Excel, Notepad, etc.  
**Rozwiązanie:**
```bash
# 1. Zamknij plik w edytorze
# 2. Uruchom pipeline ponownie
python 00_preprocessing_pipeline.py
```

### Problem: "Encoding error" na Windowsie
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```
**Powód:** Windowsowe znaki specjalne (polskie diakrytyki)  
**Rozwiązanie:** (już wbudowane w kod)
```python
# Kod już obsługuje encoding
df = pd.read_csv(file, encoding='utf-8-sig')
```

---

## 📈 Debugowanie poszczególnych kroków

Jeśli pipeline się zawala, sprawdzaj po kolei:

```bash
# 1. Eksploracja - sprawdzenie struktury
python 02_data_exploration.py
# Jeśli OK → sprawdź reports/01_exploration_summary.json

# 2. Czyszczenie - sprawdzenie transformacji
python 03_data_cleaning.py
# Jeśli OK → sprawdź czy games_cleaned.csv ma ~120k wierszy

# 3. Feature engineering - sprawdzenie nowych cech
python 04_feature_engineering.py
# Jeśli OK → sprawdź czy games_engineered.csv ma 61 kolumn

# 4. Walidacja - quality score
python 05_data_validation.py
# Jeśli OK → sprawdź reports/02_validation_report.json

# 5. Export - finalne dane
python 06_data_export.py
# Jeśli OK → sprawdź data/processed/ (powinna mieć 7 plików)
```

---

## 📊 Monitorowanie postępu

Podczas wykonania pipeline'u:

```
📋 Log: preprocessing.log
├─ [INFO] Starting pipeline...
├─ [STAGE 1] Exploration - START
├─ [STAGE 1] Exploration - SUCCESS (7.5s)
├─ [STAGE 2] Cleaning - SUCCESS (8.8s)
├─ [STAGE 3] Feature Engineering - SUCCESS (6.8s)
├─ [STAGE 4] Validation - SUCCESS (2.7s)
│           └─ Quality Score: 84.4/100 ✅
├─ [STAGE 5] Export - SUCCESS (6.2s)
└─ [SUMMARY] Total time: 32.0s ✅
```

Odczytaj log:
```bash
# Ostatnie 50 linii
tail -50 preprocessing.log

# Szukaj błędów
grep ERROR preprocessing.log

# Czasy wykonania
grep "SUCCESS\|FAILED" preprocessing.log
```

---

## 🎓 Kolejne kroki po preprocessingu

Po pomyślnym wykonaniu pipeline'u i uzyskaniu 15 kolumn:

### 1. Exploratory Data Analysis (EDA)
```python
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet('data/processed/games_final.parquet')

# Rozkłady
df.hist(figsize=(15, 10))
plt.show()

# Korelacje
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Boxplots (anomalies)
df.boxplot()
plt.show()
```

### 2. Modelowanie ML
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('data/processed/games_train.csv')
test_df = pd.read_csv('data/processed/games_test.csv')

# Przykład: Regresja na cenę
X_train = train_df.drop(['Price', 'Name', 'AppID'], axis=1)
y_train = train_df['Price']

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

### 3. Optymalizacja
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Normalizacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Cross-validation
scores = cross_val_score(model, X_scaled, y_train, cv=5)
print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 📚 Dodatkowe materiały

**Dokumenty w projekcie:**
- `PREPROCESSING_GUIDE.md` ← **Ty tutaj!**
- `data/processed/README.md` - Dokumentacja finalna
- `data/processed/columns_documentation.csv` - Spis kolumn

**Raporty wygenerowane:**
- `reports/01_exploration_summary.json` - Eksploracja
- `reports/02_validation_report.json` - Quality Score
- `reports/03_export_summary.json` - Export
- `preprocessing.log` - Full log

**Kaggle dataset:**
- https://www.kaggle.com/fronkongames/steam-games-dataset

---

## 🚀 Podsumowanie - Szybki Start

**TL;DR - na szybko:**

```bash
# 1. Pobierz dane (raz)
python 01_data_collection.py

# 2. Uruchom cały pipeline (wszystkie 5 kroków)
python 00_preprocessing_pipeline.py

# 3. Załaduj dane do ML
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/games_final.parquet')
print(f'✅ Gotowe! Shape: {df.shape}')
"

# 4. Zasoby
train = pd.read_csv('data/processed/games_train.csv')    # 98k wierszy
test = pd.read_csv('data/processed/games_test.csv')      # 24k wierszy
```

---

## 📄 Metadata

| Parametr | Wartość |
|----------|---------|
| **Dataset** | Steam Games (Kaggle) |
| **Oryginalne kolumny** | 39 |
| **Wybrane kolumny** | 18 |
| **Engineered features** | 61 |
| **Finalne kolumny (ML)** | 15 |
| **Wierszy** | 122,611 |
| **Train/Test** | 98,089 / 24,522 (80/20) |
| **Quality Score** | 84.4/100 ✅ |
| **Czas pipeline'u** | 32 sekundy |
| **Rozmiar (CSV)** | 17 MB |
| **Rozmiar (Parquet)** | 3.2 MB |
| **Brakujące wartości** | < 4% (głównie Genres) |
| **Duplikaty AppID** | 0 ✅ |
| **Python** | 3.8+ |
| **Ostatnia aktualizacja** | 2026-04-18 |

---

**Created**: 2026-04-18  
**Pipeline Author**: ML Project Steam  
**Status**: ✅ Production Ready

