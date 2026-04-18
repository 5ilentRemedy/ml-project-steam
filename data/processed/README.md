# Steam Games Dataset - Przetworzenie

## Opis
Przetworzony dataset gier ze Steam przygotowany do modelowania ML.

## Zawartosc
- games_final.csv - Ostateczne dane w formacie CSV
- games_final.parquet - Ostateczne dane w formacie Parquet (binarny)
- games_train.csv - Zbior treningowy (80%)
- games_test.csv - Zbior testowy (20%)
- columns_documentation.csv - Dokumentacja wszystkich kolumn
- dataset_manifest.json - Manifest i metadata datasetu

## Statystyka
- Calkowite rekordy: 122,611
- Calkowite cechy: 15
- Wartosci brakujace: 8,954
- Duplikaty: 0

## Ladownie danych w Pythonie
```python
import pandas as pd

# CSV
df = pd.read_csv('games_final.csv')

# Parquet (szybsze)
df = pd.read_parquet('games_final.parquet')

# Train/Test split
train_df = pd.read_csv('games_train.csv')
test_df = pd.read_csv('games_test.csv')
```

---
Wygenerowano: 2026-04-18 14:19:57
