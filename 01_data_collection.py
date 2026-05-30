"""
01_data_collection.py
Automatyczne pobieranie najnowszej wersji zbioru danych ze Steam API/Kaggle.
"""

import os
import shutil
import kagglehub
import datetime

def download_and_move_dataset():
    """Pobiera dane z Kaggle i przenosi je z cache do katalogu projektu z timestampem"""
    try:
        # Ustalanie katalogu docelowego w projekcie
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")

        # ID zbioru danych na Kaggle
        dataset_id = "fronkongames/steam-games-dataset"
        print(f"Pobieranie zbioru danych: {dataset_id}...")
        
        # Pobieranie przez kagglehub (zapisuje w cache lokalnym użytkownika)
        cache_path = kagglehub.dataset_download(dataset_id)
        print(f"Pliki zapisane w cache: {cache_path}")

        # Tworzenie katalogu data, jeśli nie istnieje
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Utworzono katalog docelowy: {data_dir}")

        # Generowanie znacznika czasu (timestamp)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Przenoszenie (kopiowanie) plików z cache do katalogu data
        copied_any = False
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                # Interesują nas tylko pliki CSV z danymi, pomijamy ewentualne metadane
                if not file.endswith('.csv'):
                    continue
                    
                source_path = os.path.join(root, file)
                
                # Dodanie znacznika czasu do nazwy pliku dla pełnego wersjonowania
                filename, ext = os.path.splitext(file)
                new_filename = f"{filename}_{timestamp}{ext}"
                destination_path = os.path.join(data_dir, new_filename)
                
                shutil.copy2(source_path, destination_path)
                print(f"Skopiowano: {new_filename} do katalogu data/")
                copied_any = True

        if copied_any:
            print(f"\n[OK] Zakończono sukcesem pobieranie i przenoszenie danych do: {data_dir}")
            return True
        else:
            print("\n[!] Nie znaleziono żadnych plików CSV do skopiowania w cache.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Wystąpił błąd podczas pobierania danych: {e}")
        return False

if __name__ == "__main__":
    success = download_and_move_dataset()
    # Zwracamy odpowiedni kod wyjścia dla systemów operacyjnych i procesów subprocess
    import sys
    sys.exit(0 if success else 1)