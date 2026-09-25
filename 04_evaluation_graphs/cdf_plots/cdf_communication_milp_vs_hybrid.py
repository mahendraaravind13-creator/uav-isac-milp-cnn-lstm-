import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

FILE = DATASET_DIR / "master_table_FULLdec_with_predictions.xlsx"

# ================= LOAD =================
df = pd.read_excel(FILE)
df.columns = df.columns.str.strip()

# ================= CHECK =================
required = ["PsiComm", "Label", "Predicted_Label"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

TIME_STEPS = 15
nrows = (len(df) // TIME_STEPS) * TIME_STEPS
df = df.iloc[:nrows].reset_index(drop=True)

num_envs = nrows // TIME_STEPS
print("Total trajectories:", num_envs)

Agg_Comm_MILP = np.zeros(num_envs)
Agg_Comm_CNN  = np.zeros(num_envs)

# ================= AGGREGATION =================
for env in range(num_envs):
    blk = df.iloc[env*TIME_STEPS:(env+1)*TIME_STEPS]

    # Actual (MILP): stages 2 & 3
    mask_milp = (blk["Label"] == 2) | (blk["Label"] == 3)
    Agg_Comm_MILP[env] = blk.loc[mask_milp, "PsiComm"].sum()

    # Predicted (CNN): stages 2 & 3
    mask_cnn = (blk["Predicted_Label"] == 2) | (blk["Predicted_Label"] == 3)
    Agg_Comm_CNN[env] = blk.loc[mask_cnn, "PsiComm"].sum()

# ================= CDF =================
x_m = np.sort(np.log10(Agg_Comm_MILP + 1e-12))
x_c = np.sort(np.log10(Agg_Comm_CNN  + 1e-12))

y = np.arange(1, num_envs + 1) / num_envs

plt.figure(figsize=(8,6))
plt.plot(x_m, y, 'b-', lw=2, label="Actual (MILP)")
plt.plot(x_c, y, 'r--', lw=2, label="Predicted (CNN)")
plt.xlabel("log10(Aggregate Communication Performance)")
plt.ylabel("CDF")
plt.title("CDF of Aggregate Communication Performance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print("DONE")

