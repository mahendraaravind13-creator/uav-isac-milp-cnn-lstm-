import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit
import sys

# ================= PATHS =================
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
DATASET_DIR = BASE_DIR / "Dataset"

MASTER_FILE = DATASET_DIR / "master_table_FULLdec.xlsx"
PRED_FILE   = DATASET_DIR / "master_table_FULLdec_with_predictions.xlsx"

# ================= LOAD =================
df = pd.read_excel(MASTER_FILE)
df_pred = pd.read_excel(PRED_FILE)

df.columns = df.columns.str.strip()
df_pred.columns = df_pred.columns.str.strip()

df["Predicted_Label"] = df_pred["Predicted_Label"].astype(np.int32)

# ================= CONSTANTS =================
TIME_STEPS = 15
K = 10
H = 70.0
Pt = 0.1
Gp = 10.0
Gs = 5.0
Gt = 10.0
sigma2 = 1e-9
sigma_rcs = 1.0
lambda_c = 0.1
a = 1.0
epsilon = 1e-12

beta0 = (Gt * Gs * sigma_rcs * lambda_c**2) / (4*np.pi)**3
coeff = (a * sigma2) / (Pt * Gp * beta0)

# ================= NUMBA CORE =================
@njit(fastmath=True)
def compute_agg_psisense(s, ST, labels):
    J = s.shape[0]
    agg = 0.0

    for j in range(J):
        J11 = np.zeros(K)
        J22 = np.zeros(K)
        J12 = np.zeros(K)

        for jp in range(j + 1):
            if labels[jp] != 1 and labels[jp] != 3:
                continue

            ux, uy = s[jp]

            for k in range(K):
                dx0 = ST[k,0] - ux
                dy0 = ST[k,1] - uy
                d = np.sqrt(H*H + dx0*dx0 + dy0*dy0)

                inv_var = 1.0 / (coeff * d**4 + epsilon)
                dx = dx0 / d
                dy = dy0 / d

                J11[k] += inv_var * dx * dx
                J22[k] += inv_var * dy * dy
                J12[k] += inv_var * dx * dy

        psi_j = 0.0
        for k in range(K):
            denom = J11[k]*J22[k] - J12[k]*J12[k]
            if denom > epsilon:
                psi_j += (J11[k] + J22[k]) / denom
            else:
                psi_j += 1e18

        agg += psi_j / K

    return agg

# ================= MAIN =================
nrows = (len(df) // TIME_STEPS) * TIME_STEPS
df = df.iloc[:nrows].reset_index(drop=True)

num_envs = nrows // TIME_STEPS
Agg_MILP = np.zeros(num_envs)
Agg_CNN  = np.zeros(num_envs)

print("Total trajectories:", num_envs)

for env in range(num_envs):
    blk = df.iloc[env*TIME_STEPS:(env+1)*TIME_STEPS]

    s = blk[["X","Y"]].values.astype(np.float64)

    ST = np.array([
        [blk.iloc[0][f"ST{k}_x"], blk.iloc[0][f"ST{k}_y"]]
        for k in range(1, K+1)
    ], dtype=np.float64)

    Agg_MILP[env] = compute_agg_psisense(
        s, ST, blk["Label"].values.astype(np.int32)
    )

    Agg_CNN[env] = compute_agg_psisense(
        s, ST, blk["Predicted_Label"].values.astype(np.int32)
    )

    if env % 5000 == 0:
        print(f"Processed {env}/{num_envs}")

# ================= CDF =================
x_m = np.sort(np.log10(Agg_MILP + 1e-12))
x_c = np.sort(np.log10(Agg_CNN  + 1e-12))

y_m = np.arange(1, len(x_m)+1) / len(x_m)
y_c = np.arange(1, len(x_c)+1) / len(x_c)


plt.figure(figsize=(8,6))
plt.plot(x_m, y_m, 'g-', lw=2, label="Actual (MILP)")
plt.plot(x_c, y_c, 'm--', lw=2, label="Predicted (CNN)")
plt.xlabel("log10(Aggregate PsiSense)")
plt.ylabel("CDF")
plt.title("CDF: Recomputed Sensing Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("DONE")
