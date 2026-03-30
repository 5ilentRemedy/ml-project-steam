import os
import shutil
import kagglehub

path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Files cached: {path}")


for root, dirs, files in os.walk(path):
    for file in files:
        source_path = os.path.join(root, file)
        
        destination_path = os.path.join(current_dir, file)
        
        shutil.copy2(source_path, destination_path)
        print(f"Copied: {file}")

print(f"\nDatasets moved to workdir: {current_dir}")