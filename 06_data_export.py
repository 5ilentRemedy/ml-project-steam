import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import sys, io

if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    logger.warning("openpyxl nie zainstalowany - XLSX z filtrami nie będą generowane << dorzuć openpyxl do requirements.txt i zainstaluj, aby mieć te pliki >>")

class DataExporter:
    def __init__(self):
        self.df = None
        self.export_info = {}
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        input_file = data_dir / "games_engineered.csv"
        self.df = pd.read_csv(input_file, index_col=False)
        logger.info(f"Zaladowano dane: {self.df.shape}")
        return self.df
    
    def select_final_features(self):
        logger.info("Wybieranie 15 kluczowych cech do modelowania...")
        kept_columns = [
            'AppID', 'Name', 'Genres',
            'Release_year', 'Days_since_release',
            'Platform_count', 'Price', 'Is_free',
            'Total_reviews', 'Review_ratio', 'Is_highly_rated',
            'Log_owners', 'Has_achievements', 'Log_total_reviews', 'Genre_count'
        ]
        final_columns = [col for col in kept_columns if col in self.df.columns]
        self.df = self.df[final_columns].copy()
        logger.info(f"[OK] Wybrano {len(final_columns)} cech")
        logger.info(f"Ostateczne dane: {self.df.shape}")
        return self.df
    
    def export_csv(self):
        logger.info("Eksportowanie do CSV...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "games_final.csv"
        self.df.to_csv(output_file, index=False)
        logger.info(f"[OK] Eksportowano: {output_file.name}")
        self.export_info['csv'] = str(output_file)
        return self.df
    
    def export_parquet(self):
        logger.info("Eksportowanie do Parquet...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "games_final.parquet"
        self.df.to_parquet(output_file, index=False, compression='gzip')
        logger.info(f"[OK] Eksportowano: {output_file.name}")
        self.export_info['parquet'] = str(output_file)
        return self.df
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        logger.info(f"Tworzenie train/test splitu (test_size={test_size})...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(random_state)
        test_indices = np.random.choice(len(self.df), size=int(len(self.df) * test_size), replace=False)
        train_df = self.df.drop(test_indices)
        test_df = self.df.iloc[test_indices]
        train_file = output_dir / "games_train.csv"
        test_file = output_dir / "games_test.csv"
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        logger.info(f"[OK] Train set: {len(train_df)} wierszy ({len(train_df)/len(self.df)*100:.1f}%)")
        logger.info(f"[OK] Test set: {len(test_df)} wierszy ({len(test_df)/len(self.df)*100:.1f}%)")
        self.export_info['train_test_split'] = {
            'train_file': str(train_file),
            'test_file': str(test_file),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'split_ratio': float(test_size)
        }
        return train_df, test_df
    
    def create_feature_groups(self):
        logger.info("Tworzenie grup cech dla 15 kolumn...")
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
        self.export_info['feature_groups'] = feature_groups
        logger.info(f"[OK] Zdefiniowano {len(feature_groups)} grup cech")
        return feature_groups
    
    def create_data_documentation(self):
        logger.info("Tworzenie dokumentacji...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        columns_doc = {'column_name': [], 'data_type': [], 'description': [], 'missing_count': [], 'unique_values': []}
        descriptions = {
            'AppID': 'Unikalny identyfikator gry w Steam',
            'Name': 'Nazwa gry',
            'Genres': 'Gatunki gry (separator: przecinek)',
            'Release_year': 'Rok wydania gry',
            'Days_since_release': 'Liczba dni od wydania gry',
            'Platform_count': 'Liczba platform (0-3: brak, tylko jedna, dwie, wszystkie)',
            'Price': 'Cena gry w USD',
            'Is_free': 'Czy gra jest darmowa (1=tak, 0=nie)',
            'Total_reviews': 'Calkowita liczba recenzji (pozytywne + negatywne)',
            'Review_ratio': 'Udzial recenzji pozytywnych do calkowitych',
            'Is_highly_rated': 'Czy gra ma wysoka ocene (1=Metacritic >= 75)',
            'Log_owners': 'Log transformacja liczby oszacowanych wlascicieli',
            'Has_achievements': 'Czy gra posiada osiagniecia (1=tak, 0=nie)',
            'Log_total_reviews': 'Log transformacja calkowitej liczby recenzji',
            'Genre_count': 'Liczba gatunkow, ktorych przydzie gra'
        }
        for col in self.df.columns:
            columns_doc['column_name'].append(col)
            columns_doc['data_type'].append(str(self.df[col].dtype))
            columns_doc['description'].append(descriptions.get(col, ''))
            columns_doc['missing_count'].append(int(self.df[col].isna().sum()))
            columns_doc['unique_values'].append(int(self.df[col].nunique()))
        doc_df = pd.DataFrame(columns_doc)
        doc_file = output_dir / "columns_documentation.csv"
        doc_df.to_csv(doc_file, index=False)
        logger.info(f"[OK] Dokumentacja kolumn: {doc_file.name}")
        return doc_df
    
    def create_manifest(self):
        logger.info("Tworzenie manifestu...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            'dataset_name': 'Steam Games Dataset - Preprocessed',
            'creation_date': datetime.now().isoformat(),
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'data_shape': list(self.df.shape),
            'feature_groups': self.export_info.get('feature_groups', {}),
            'train_test_split': self.export_info.get('train_test_split', {}),
            'files': {
                'main_data': 'games_final.csv',
                'parquet_data': 'games_final.parquet',
                'train_data': 'games_train.csv',
                'test_data': 'games_test.csv',
                'columns_doc': 'columns_documentation.csv',
                'manifest': 'dataset_manifest.json'
            },
            'quality_metrics': {
                'null_values': int(self.df.isna().sum().sum()),
                'duplicate_rows': int(self.df.duplicated().sum()),
                'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            'column_summary': {
                'numeric': len(self.df.select_dtypes(include=[np.number]).columns),
                'categorical': len(self.df.select_dtypes(include=['object']).columns),
                'datetime': len(self.df.select_dtypes(include=['datetime64']).columns)
            }
        }
        manifest_file = output_dir / "dataset_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Manifest: {manifest_file.name}")
        return manifest
    
    def create_readme(self):
        logger.info("Tworzenie README...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        readme_content = f"""# Steam Games Dataset - Przetworzenie

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
- Calkowite rekordy: {len(self.df):,}
- Calkowite cechy: {len(self.df.columns)}
- Wartosci brakujace: {self.df.isna().sum().sum():,}
- Duplikaty: {self.df.duplicated().sum():,}

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
Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        logger.info(f"[OK] README: {readme_file.name}")
    
    def save_summary(self):
        summary_file = Path(__file__).parent / "reports" / "03_export_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        self.export_info['dataset_shape'] = list(self.df.shape)
        self.export_info['columns'] = list(self.df.columns)
        self.export_info['timestamp'] = datetime.now().isoformat()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.export_info, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Streszczenie exportu: {summary_file.name}")
    
    def export_feature_groups_csv(self):
        """Generuje osobne CSV dla każdej grupy cech"""
        logger.info("Generowanie CSV dla grup cech...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            cols_present = [col for col in columns if col in self.df.columns]
            if cols_present:
                group_df = self.df[cols_present].copy()
                group_file = output_dir / f"games_group_{group_name}.csv"
                group_df.to_csv(group_file, index=False)
                logger.info(f"  [OK] {group_name}: {group_file.name} ({len(cols_present)} kolumn)")
    
    def export_with_filters_xlsx(self):
        """Generuje XLSX z filtrami na nagłówkach i zamrożonymi wierszami"""
        if not HAS_OPENPYXL:
            logger.warning("  [SKIP] openpyxl nie dostępny - pomijanie XLSX z filtrami")
            return
        
        logger.info("Generowanie XLSX z filtrami na nagłówkach...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Całe dane z filtrami
        output_file = output_dir / "games_final_with_filters.xlsx"
        self._create_xlsx_with_filters(self.df, output_file, "Games Data")
        logger.info(f"  [OK] games_final_with_filters.xlsx")
        
        # Train set z filtrami
        np.random.seed(42)
        test_indices = np.random.choice(len(self.df), size=int(len(self.df) * 0.2), replace=False)
        train_df = self.df.drop(test_indices)
        
        train_file = output_dir / "games_train_with_filters.xlsx"
        self._create_xlsx_with_filters(train_df, train_file, "Training Data")
        logger.info(f"  [OK] games_train_with_filters.xlsx ({len(train_df)} wierszy)")
        
        # Test set z filtrami
        test_df = self.df.iloc[test_indices]
        test_file = output_dir / "games_test_with_filters.xlsx"
        self._create_xlsx_with_filters(test_df, test_file, "Test Data")
        logger.info(f"  [OK] games_test_with_filters.xlsx ({len(test_df)} wierszy)")
    
    def _create_xlsx_with_filters(self, df, output_file, sheet_name="Data"):
        """Helper: Tworzy XLSX z filtrami i zamrożonymi nagłówkami"""
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        
        # Wpisz nagłówki
        for col_idx, col_name in enumerate(df.columns, 1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = cell.font.copy()
            cell.font = cell.font.copy()
            from openpyxl.styles import Font, PatternFill
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        # Wpisz dane
        for row_idx, row in enumerate(df.values, 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                # Formatowanie liczb
                if isinstance(value, float):
                    cell.number_format = '0.00'
                elif isinstance(value, int) and col_idx != 1:  # Nie formatuj AppID
                    cell.number_format = '0'
        
        # Ustaw szerokość kolumn
        for col_idx, col_name in enumerate(df.columns, 1):
            width = max(len(str(col_name)), 12)
            ws.column_dimensions[get_column_letter(col_idx)].width = width
        
        # Zamróż nagłówek (pierwszy wiersz)
        ws.freeze_panes = "A2"
        
        # Dodaj filtry autofilter
        max_row = len(df) + 1
        max_col = len(df.columns)
        ws.auto_filter.ref = f"A1:{get_column_letter(max_col)}{max_row}"
        
        wb.save(output_file)
    
    def export_grouped_xlsx(self):
        """Generuje XLSX z oddzielnymi arkuszami dla każdej grupy cech"""
        if not HAS_OPENPYXL:
            logger.warning("  [SKIP] openpyxl nie dostępny - pomijanie XLSX z grupami")
            return
        
        logger.info("Generowanie XLSX z grupami cech jako oddzielne arkusze...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        feature_groups = {
            'Identifiers': ['AppID', 'Name'],
            'Temporal': ['Release_year', 'Days_since_release'],
            'Platform': ['Platform_count'],
            'Reviews': ['Total_reviews', 'Review_ratio', 'Log_total_reviews'],
            'Scores': ['Is_highly_rated'],
            'Content': ['Log_owners', 'Has_achievements', 'Genre_count'],
            'Price': ['Price', 'Is_free'],
            'Metadata': ['Genres']
        }
        
        output_file = output_dir / "games_final_grouped.xlsx"
        wb = Workbook()
        wb.remove(wb.active)  # Usuń domyślny arkusz
        
        for group_name, columns in feature_groups.items():
            cols_present = [col for col in columns if col in self.df.columns]
            if cols_present:
                ws = wb.create_sheet(group_name)
                group_df = self.df[cols_present].copy()
                
                # Nagłówki
                for col_idx, col_name in enumerate(cols_present, 1):
                    cell = ws.cell(row=1, column=col_idx, value=col_name)
                    from openpyxl.styles import Font, PatternFill
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                
                # Dane
                for row_idx, row in enumerate(group_df.values, 2):
                    for col_idx, value in enumerate(row, 1):
                        cell = ws.cell(row=row_idx, column=col_idx, value=value)
                        if isinstance(value, float):
                            cell.number_format = '0.00'
                
                # Zamróż nagłówek i szerokość
                ws.freeze_panes = "A2"
                for col_idx, col_name in enumerate(cols_present, 1):
                    width = max(len(str(col_name)), 12)
                    from openpyxl.utils import get_column_letter
                    ws.column_dimensions[get_column_letter(col_idx)].width = width
        
        wb.save(output_file)
        logger.info(f"  [OK] games_final_grouped.xlsx ({len(feature_groups)} arkuszy)")
    
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("EXPORT I PRZYGOTOWANIE DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.select_final_features()
        
        # ISTNIEJĄCE PLIKI
        logger.info("\n>>> PLIKI GŁÓWNE (CSV/Parquet)")
        self.export_csv()
        self.export_parquet()
        self.create_train_test_split()
        
        # NOWE PLIKI - GRUPY I FILTRY
        logger.info("\n>>> PLIKI Z GRUPAMI I FILTRAMI")
        self.export_feature_groups_csv()
        self.export_with_filters_xlsx()
        self.export_grouped_xlsx()
        
        # DOKUMENTACJA
        logger.info("\n>>> DOKUMENTACJA")
        self.create_feature_groups()
        self.create_data_documentation()
        self.create_manifest()
        self.create_readme()
        self.save_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] EXPORT UKONCZNY")
        logger.info("=" * 80 + "\n")
        logger.info("Pliki wyjsciowe w: data/processed/\n")
        logger.info("📋 PLIKI GŁÓWNE (niezmienione):")
        logger.info("  ✓ games_final.csv")
        logger.info("  ✓ games_final.parquet")
        logger.info("  ✓ games_train.csv")
        logger.info("  ✓ games_test.csv")
        logger.info("\n📁 GRUPY KOLUMN (nowe):")
        logger.info("  ✓ games_group_identifiers.csv")
        logger.info("  ✓ games_group_temporal.csv")
        logger.info("  ✓ games_group_platform.csv")
        logger.info("  ✓ games_group_reviews.csv")
        logger.info("  ✓ games_group_scores.csv")
        logger.info("  ✓ games_group_content.csv")
        logger.info("  ✓ games_group_price.csv")
        logger.info("  ✓ games_group_metadata.csv")
        logger.info("\n🔍 XLSX Z FILTRAMI (nowe):")
        if HAS_OPENPYXL:
            logger.info("  ✓ games_final_with_filters.xlsx (wszystkie dane)")
            logger.info("  ✓ games_train_with_filters.xlsx (80% treningowe)")
            logger.info("  ✓ games_test_with_filters.xlsx (20% testowe)")
            logger.info("  ✓ games_final_grouped.xlsx (8 arkuszy z grupami)")
        else:
            logger.info("openpyxl nie dostępny - XLSX nie wygenerowane << dorzuć openpyxl do requirements.txt i zainstaluj, aby mieć te pliki >>")
        logger.info("\n DOKUMENTACJA:")
        logger.info("  ✓ columns_documentation.csv")
        logger.info("  ✓ dataset_manifest.json")
        logger.info("  ✓ README.md")
        logger.info("  - games_train.csv")
        logger.info("  - games_test.csv")
        logger.info("  - dataset_manifest.json")
        logger.info("  - columns_documentation.csv")
        logger.info("  - README.md")

def main():
    exporter = DataExporter()
    exporter.run()

if __name__ == "__main__":
    main()
