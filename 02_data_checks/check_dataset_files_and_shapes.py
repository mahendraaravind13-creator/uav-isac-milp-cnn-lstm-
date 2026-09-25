from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

print("=== FILES FOUND IN Dataset FOLDER ===")
for f in DATASET_DIR.iterdir():
    print(f.name)

print("\nLoading BIG (training) file...")
df_big_file = pd.read_excel(DATASET_DIR / "Copy of UAV90BIG1.xlsx", nrows=1000)

print("Loading SMALL (testing) file...")
df_small_file = pd.read_excel(DATASET_DIR / "master_table_FULLdec.xlsx", nrows=1000)

print("BIG shape:", df_big_file.shape)
print("SMALL shape:", df_small_file.shape)
