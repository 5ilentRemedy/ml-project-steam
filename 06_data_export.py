"""
04_data_export.py
Selekcja cech końcowych, podział na zbiory (Train/Val/Test) i eksport wieloformatowy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import sys
import io
from sklearn.model_selection import train_test_split

# UTF-8 encoding dla poprawnego wyświetlania znaków w konsoli Windows
if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    logger.warning("openpyxl nie jest zainstalowany - pliki XLSX nie zostaną wygenerowane. Dodaj openpyxl do requirements.txt.")

class DataExporter:
    def __init__(self):
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.export_info = {}
        self.feature_groups = {
            'identifiers': ['AppID', 'Name'],
            'temporal': ['Release_year', 'Days_since_release'],
            'platform': ['Platform_count'],
            'reviews': ['Total_reviews', 'Review_ratio', 'Log_total_reviews'],
            'scores': ['Market_Success'],
            'content': ['Log_owners', 'Has_achievements', 'Genre_count'],
            'price': ['Price', 'Is_free'],
            'metadata': ['Genres']
        }
    
    def load_data(self):
        data_dir = Path(__file__).parent / "data"
        input_file = data_dir / "games_engineered.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku cech inżynieryjnych: {input_file}")
        self.df = pd.read_csv(input_file, index_col=False)
        logger.info(f"Załadowano dane do eksportu: {self.df.shape}")
        return self.df
    
    def select_final_features(self):
        logger.info("Wybieranie kluczowych cech do ostatecznego modelowania...")
        # Spłaszczamy listę kolumn ze zdefiniowanych grup cech
        kept_columns = []
        for cols in self.feature_groups.values():
            kept_columns.extend(cols)
            
        final_columns = [col for col in kept_columns if col in self.df.columns]
        self.df = self.df[final_columns].copy()
        logger.info(f"[OK] Selekcja zakończona. Zachowano {len(final_columns)} cech.")
        return self.df
    
    def export_csv(self):
        logger.info("Eksportowanie głównego zbioru do formatu CSV...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "games_final.csv"
        self.df.to_csv(output_file, index=False)
        self.export_info['csv'] = str(output_file)
        return self.df
    
    def export_parquet(self):
        logger.info("Eksportowanie głównego zbioru do wydajnego formatu Parquet...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "games_final.parquet"
        self.df.to_parquet(output_file, index=False, compression='gzip')
        self.export_info['parquet'] = str(output_file)
        return self.df
    
    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        logger.info(f"Tworzenie podziału Train/Val/Test ze stratyfikacją klas sukcesu...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        stratify_col = 'Market_Success' if 'Market_Success' in self.df.columns else None

        if stratify_col:
            logger.info(f"[INFO] Stratyfikacja matematyczna wg kolumny docelowej: {stratify_col}")
            self.train_df, temp_df = train_test_split(
                self.df,
                train_size=train_ratio,
                random_state=random_state,
                stratify=self.df[stratify_col]
            )
            # Podział pozostałego zbioru (30%) po połowie na walidację i test (0.15 / 0.30 = 0.5)
            self.val_df, self.test_df = train_test_split(
                temp_df,
                train_size=0.5,
                random_state=random_state,
                stratify=temp_df[stratify_col]
            )
        else:
            logger.warning("[!] Kolumna docelowa nieznaleziona. Stosowanie podziału losowego bez stratyfikacji.")
            self.train_df, temp_df = train_test_split(self.df, train_size=train_ratio, random_state=random_state)
            self.val_df, self.test_df = train_test_split(temp_df, train_size=0.5, random_state=random_state)

        # Zapis zbiorów do plików CSV
        self.train_df.to_csv(output_dir / "games_train.csv", index=False)
        self.val_df.to_csv(output_dir / "games_val.csv", index=False)
        self.test_df.to_csv(output_dir / "games_test.csv", index=False)

        total_len = len(self.train_df) + len(self.val_df) + len(self.test_df)
        logger.info(f"[OK] Zbiór treningowy: {len(self.train_df)} wierszy ({len(self.train_df)/total_len*100:.1f}%)")
        logger.info(f"[OK] Zbiór walidacyjny: {len(self.val_df)} wierszy ({len(self.val_df)/total_len*100:.1f}%)")
        logger.info(f"[OK] Zbiór testowy: {len(self.test_df)} wierszy ({len(self.test_df)/total_len*100:.1f}%)")

        self.export_info['train_val_test_split'] = {
            'train_size': len(self.train_df),
            'val_size': len(self.val_df),
            'test_size': len(self.test_df),
            'stratified': True if stratify_col else False
        }
        return self.train_df, self.val_df, self.test_df
    
    def export_feature_groups_csv(self):
        logger.info("Generowanie osobnych plików CSV dla wyznaczonych grup cech...")
        output_dir = Path(__file__).parent / "data" / "processed"
        
        for group_name, columns in self.feature_groups.items():
            cols_present = [col for col in columns if col in self.df.columns]
            if cols_present:
                group_df = self.df[cols_present].copy()
                group_file = output_dir / f"games_group_{group_name}.csv"
                group_df.to_csv(group_file, index=False)
    
    def export_with_filters_xlsx(self):
        """Generuje raporty XLSX z filtrami na podstawie zsynchronizowanych podziałów"""
        if not HAS_OPENPYXL:
            return
        
        logger.info("Generowanie zaawansowanych skoroszytów XLSX z filtrami...")
        output_dir = Path(__file__).parent / "data" / "processed"
        
        self._create_xlsx_with_filters(self.df, output_dir / "games_final_with_filters.xlsx", "Wszystkie Dane")
        self._create_xlsx_with_filters(self.train_df, output_dir / "games_train_with_filters.xlsx", "Zbiór Treningowy")
        self._create_xlsx_with_filters(self.test_df, output_dir / "games_test_with_filters.xlsx", "Zbiór Testowy")
    
    def _create_xlsx_with_filters(self, df, output_file, sheet_name="Data"):
        """Przekształca DataFrame w sformatowany arkusz Excel z zamrożonym nagłówkiem"""
        if df is None:
            return
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        
        # Formatowanie nagłówków
        from openpyxl.styles import Font, PatternFill
        for col_idx, col_name in enumerate(df.columns, 1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        # Wprowadzanie wierszy i formatowanie wartości numerycznych
        for row_idx, row in enumerate(df.values, 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                if isinstance(value, float):
                    cell.number_format = '0.00'
                elif isinstance(value, int) and col_idx != 1:
                    cell.number_format = '0'
        
        # Automatyczne dopasowanie szerokości
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = max(len(str(col_name)), 12)
        
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
        wb.save(output_file)
    
    def export_grouped_xlsx(self):
        if not HAS_OPENPYXL:
            return
        logger.info("Generowanie zbiorczego skoroszytu XLSX z podziałem grup na arkusze...")
        output_dir = Path(__file__).parent / "data" / "processed"
        output_file = output_dir / "games_final_grouped.xlsx"
        
        wb = Workbook()
        wb.remove(wb.active) # Usuwamy arkusz domyślny
        
        for group_name, columns in self.feature_groups.items():
            cols_present = [col for col in columns if col in self.df.columns]
            if cols_present:
                ws = wb.create_sheet(title=group_name.capitalize())
                group_df = self.df[cols_present].copy()
                
                from openpyxl.styles import Font, PatternFill
                for col_idx, col_name in enumerate(cols_present, 1):
                    cell = ws.cell(row=1, column=col_idx, value=col_name)
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                
                for row_idx, row in enumerate(group_df.values, 2):
                    for col_idx, value in enumerate(row, 1):
                        cell = ws.cell(row=row_idx, column=col_idx, value=value)
                        if isinstance(value, float):
                            cell.number_format = '0.00'
                
                ws.freeze_panes = "A2"
                for col_idx in range(1, len(cols_present) + 1):
                    ws.column_dimensions[get_column_letter(col_idx)].width = 15
        wb.save(output_file)

    def create_data_documentation(self):
        logger.info("Generowanie technicznej dokumentacji kolumn (Słownik Danych)...")
        output_dir = Path(__file__).parent / "data" / "processed"
        
        descriptions = {
            'AppID': 'Unikalny identyfikator gry w bazie danych platformy Steam',
            'Name': 'Pełna nazwa rynkowa gry',
            'Genres': 'Gatunki przypisane do gry (wartości rozdzielone przecinkami)',
            'Release_year': 'Kalendarzowy rok oficjalnego wydania gry',
            'Days_since_release': 'Liczba dni, jakie upłynęły od premiery gry do punktu kontrolnego potoku',
            'Platform_count': 'Liczba wspieranych systemów operacyjnych spośród: Windows, Mac, Linux',
            'Price': 'Cena zakupu gry podana w walucie USD',
            'Is_free': 'Flaga binarna określająca model darmowy (1: Tak, 0: Gra płatna)',
            'Total_reviews': 'Łączny wolumen opinii użytkowników (Suma ocen pozytywnych i negatywnych)',
            'Review_ratio': 'Stosunek liczby recenzji pozytywnych do całkowitej liczby ocen',
            'Market_Success': 'Wieloklasowa zmienna docelowa określająca profil sukcesu rynkowego gry',
            'Log_owners': 'Zlogarytmowana wartość szacowanej liczby posiadaczy gry',
            'Has_achievements': 'Flaga binarna określająca implementację systemu osiągnięć Steam',
            'Log_total_reviews': 'Zlogarytmowany łączny wolumen wszystkich recenzji gry',
            'Genre_count': 'Całkowita liczba gatunków przypisanych do danej gry'
        }
        
        columns_doc = {'column_name': [], 'data_type': [], 'description': [], 'missing_count': [], 'unique_values': []}
        for col in self.df.columns:
            columns_doc['column_name'].append(col)
            columns_doc['data_type'].append(str(self.df[col].dtype))
            columns_doc['description'].append(descriptions.get(col, 'Cecha wygenerowana automatycznie w potoku cech'))
            columns_doc['missing_count'].append(int(self.df[col].isna().sum()))
            columns_doc['unique_values'].append(int(self.df[col].nunique()))
            
        doc_df = pd.DataFrame(columns_doc)
        doc_df.to_csv(output_dir / "columns_documentation.csv", index=False)
    
    def create_manifest(self):
        logger.info("Zapisywanie strukturalnego manifestu zestawu danych...")
        output_dir = Path(__file__).parent / "data" / "processed"
        manifest = {
            'dataset_name': 'Steam Games Dataset - Preprocessed Potok ML',
            'creation_date': datetime.now().isoformat(),
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'feature_groups': self.feature_groups,
            'quality_metrics': {
                'null_values': int(self.df.isna().sum().sum()),
                'duplicate_rows': int(self.df.duplicated().sum())
            }
        }
        with open(output_dir / "dataset_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    def create_readme(self):
        logger.info("Generowanie dokumentacji Markdown dla struktur wyjściowych...")
        output_dir = Path(__file__).parent / "data" / "processed"
        readme_content = f"""# Dokumentacja struktur danych wejściowych do Modelu ML

## Zawartość katalogu processed/
- `games_final.csv` - Kompletny, odszumiony zbiór danych gotowy do modelowania.
- `games_final.parquet` - Zbiór danych w zoptymalizowanym formacie kolumnowym.
- `games_train.csv` - Reprezentatywny zbiór treningowy (70% danych).
- `games_val.csv` - Zbiór walidacyjny do optymalizacji hiperparametrów (15% danych).
- `games_test.csv` - Zbiór testowy do ostatecznej ewaluacji uogólnienia modeli (15% danych).

## Podstawowe statystyki zbioru:
- Liczba rekordów aktywnych rynkowo: {len(self.df):,}
- Wyselekcjonowane cechy uczące: {len(self.df.columns)}

Wygenerowano automatycznie w potoku dnia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        with open(output_dir / "dataset_documentation.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def save_summary(self):
        summary_file = Path(__file__).parent / "reports" / "03_export_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        self.export_info['dataset_shape'] = list(self.df.shape)
        self.export_info['columns'] = list(self.df.columns)
        self.export_info['timestamp'] = datetime.now().isoformat()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.export_info, f, indent=2, ensure_ascii=False)
    
    def run(self):
        logger.info("\n" + "=" * 80)
        logger.info("START POTOKU EKSPORTU I DOKUMENTACJI DANYCH")
        logger.info("=" * 80 + "\n")
        self.load_data()
        self.select_final_features()
        self.export_csv()
        self.export_parquet()
        self.create_train_val_test_split()
        self.export_feature_groups_csv()
        self.export_with_filters_xlsx()
        self.export_grouped_xlsx()
        self.create_data_documentation()
        self.create_manifest()
        self.create_readme()
        self.save_summary()
        logger.info("\n" + "=" * 80)
        logger.info("[OK] PROCES EKSPORTU ZAKOŃCZONY POMYŚLNIE")
        logger.info("=" * 80 + "\n")

def main():
    exporter = DataExporter()
    exporter.run()

if __name__ == "__main__":
    main()