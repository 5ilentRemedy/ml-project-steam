# Importowanie wymaganych bibliotek
import os # Do operacji na systemie plików
import shutil # Do operacji na plikach i katalogach (np. kopiowanie)
import kagglehub # Do pobierania danych z Kaggle
import datetime # Do pracy z datami i czasem

# Definicja funkcji do pobierania i przenoszenia zbioru danych
def download_and_move_dataset():
    # Ustalanie ścieżki do bieżącego katalogu skryptu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Definiowanie ścieżki do katalogu 'data' w projekcie
    data_dir = os.path.join(current_dir, "data")

    # Identyfikator zbioru danych na Kaggle
    dataset_id = "fronkongames/steam-games-dataset"
    print(f"Pobieranie zbioru danych: {dataset_id}...")
    
    # Pobieranie zbioru danych z Kaggle do lokalnego cache'u
    cache_path = kagglehub.dataset_download(dataset_id)
    print(f"Pliki zapisane w cache: {cache_path}")

    # Sprawdzenie, czy katalog 'data' istnieje, jeśli nie - utworzenie go
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Utworzono katalog docelowy: {data_dir}")

    # Generowanie znacznika czasu w formacie RRRRMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Przechodzenie przez wszystkie pliki w pobranym katalogu cache
    for root, dirs, files in os.walk(cache_path):
        for file in files:
            source_path = os.path.join(root, file) # Pełna ścieżka do pliku źródłowego
            
            # Rozdzielenie nazwy pliku od rozszerzenia
            filename, ext = os.path.splitext(file)
            # Utworzenie nowej nazwy pliku z dodanym znacznikiem czasu
            new_filename = f"{filename}_{timestamp}{ext}"
            # Definiowanie pełnej ścieżki docelowej dla pliku
            destination_path = os.path.join(data_dir, new_filename)
            
            # Kopiowanie pliku z cache do katalogu 'data'
            shutil.copy2(source_path, destination_path)
            print(f"Skopiowano: {new_filename} do katalogu data/")

    print(f"\nZakonczono pobieranie i przenoszenie danych do: {data_dir}")

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
if __name__ == "__main__":
    download_and_move_dataset() # Wywołanie funkcji pobierającej i przenoszącej dane
