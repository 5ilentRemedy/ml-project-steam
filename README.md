# ML Project Steam

Pipeline Machine Learning do przewidywania sukcesu gier Steam na podstawie danych z Kaggle.

## Start

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pierwszy etap pobiera dataset przez `kagglehub` i zapisuje go jako
`data/games_YYYYMMDD_HHMMSS.csv`. Domyslny dataset to
`fronkongames/steam-games-dataset`.

Mozesz wskazac inny dataset:

```powershell
$env:KAGGLE_DATASET="autor/nazwa-datasetu"
python 01_data_collection.py
```

## Uruchomienie

Pelny pipeline z trenowaniem i ewaluacja:

```powershell
python 00_pipeline_test_v2.py
```

Jesli nie ma lokalnego surowego CSV, pelny pipeline najpierw pobierze dane.

Pipeline przygotowania danych bez trenowania modeli:

```powershell
python 00_pipeline_test.py
```

Etapy mozna uruchamiac osobno:

```powershell
python 02_data_exploration.py
python 03_data_cleaning.py
python 04_feature_engineering.py
python 05_data_validation.py
python 06_data_export.py
python 02a_comprehensive_analysis.py
python 07a_model_training_advanced.py
python 08a_model_evaluation_advanced.py
python 09_predict_cli.py
```

## Co robi czyszczenie

- usuwa duplikaty po `AppID`,
- usuwa rekordy bez wymaganych pol: `AppID`, `Name`, `Release date`,
- odrzuca niepoprawne i przyszle daty wydania,
- konwertuje pola liczbowe oraz platformy do stabilnych typow,
- przycina niemozliwe wartosci cen, ocen i licznikow,
- usuwa rekordy bez sensownego sygnalu dla ML: brak gatunku, brak recenzji i brak Metacritic.

## Definicja sukcesu

`Is_highly_rated = 1`, gdy:

- `Review_ratio >= 0.70` i `Total_reviews >= 20`, lub
- `Metacritic_score >= 75`.

To daje sensowniejszy balans klas niz bardzo restrykcyjna definicja 80%/30 recenzji.

## Wyniki

Pipeline zapisuje:

- `data/games_cleaned.csv`
- `data/games_engineered.csv`
- `data/processed/games_final.csv`
- `data/processed/games_train.csv`
- `data/processed/games_val.csv`
- `data/processed/games_test.csv`
- `models/best_model.joblib`
- `reports/*.json`
- `reports/figures/*.png`

Archiwum poprzedniej wersji projektu jest w `_backup_before_rebuild/`.
