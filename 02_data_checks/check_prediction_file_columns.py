import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

df = pd.read_excel(DATASET_DIR / "master_table_FULLdec_with_predictions.xlsx")
print(df.columns.tolist())
