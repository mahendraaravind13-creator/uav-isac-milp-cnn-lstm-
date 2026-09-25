import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PATH =================
HERE = Path(__file__).resolve().parent
MASTER = HERE / "master_table_FULLdec.xlsx"
PRED   = HERE / "full_dec_cnn.xlsx"

# ================= LOAD =================
print("Loading datasets...")
df = pd.read_excel(MASTER)
dfp = pd.read_excel(PRED)

df.columns  = df.columns.str.strip()
dfp.columns = dfp.columns.str.strip()

df["Predicted_Label"] = dfp["Predicted_Label"].astype(int)

# ================= CONSTANTS =================
TIME_STEPS = 15

# Trim to full trajectories
nrows = (len(df) // TIME_STEPS) * TIME_STEPS
df = df.iloc[:nrows].reset_index(drop=True)
num_envs = nrows // TIME_STEPS
print("Total trajectories:", num_envs)

# ================= PARAM SWEEPS =================
ST_list = [2, 4, 6, 8, 10]   # effective sensing activity
CU_list = [2, 4, 6, 8, 10]   # effective communication activity

sense_milp, sense_cnn = [], []
comm_milp,  comm_cnn  = [], []

# ================= (a) SENSING vs STs =================
for K_eff in ST_list:
    s_m, s_c = [], []

    for env in range(num_envs):
        blk = df.iloc[env*TIME_STEPS:(env+1)*TIME_STEPS]

        # pick only first K_eff sensing-capable stages
        milp_idx = blk["Label"].isin([1,3])
        cnn_idx  = blk["Predicted_Label"].isin([1,3])

        s_m.append( blk.loc[milp_idx, "PsiSense"].head(K_eff).sum() )
        s_c.append( blk.loc[cnn_idx,  "PsiSense"].head(K_eff).sum() )
        

    sense_milp.append(np.mean(s_m))
    sense_cnn.append(np.mean(s_c))

# ================= (b) COMM vs CUs =================
for M_eff in CU_list:
    c_m, c_c = [], []

    for env in range(num_envs):
        blk = df.iloc[env*TIME_STEPS:(env+1)*TIME_STEPS]

        milp_idx = blk["Label"].isin([2,3])
        cnn_idx  = blk["Predicted_Label"].isin([2,3])

        c_m.append( blk.loc[milp_idx, "PsiComm"].head(M_eff).sum() )
        c_c.append( blk.loc[cnn_idx,  "PsiComm"].head(M_eff).sum() )

    comm_milp.append(np.mean(c_m))
    comm_cnn.append(np.mean(c_c))

# ================= PLOT =================
plt.figure(figsize=(12,5))

# ---- (a) Sensing ----
plt.subplot(1,2,1)
plt.plot(ST_list, sense_milp, 'o-', lw=2, label="MILP")
plt.plot(ST_list, sense_cnn,  's--', lw=2, label="Hybrid")
plt.xlabel("Number of STs (CUs fixed)")
plt.ylabel("Aggregate Sensing Performance")
plt.title("(a) Sensing vs Number of STs")
plt.grid(True)
plt.legend()

# ---- (b) Communication ----
plt.subplot(1,2,2)
plt.plot(CU_list, comm_milp, 'o-', lw=2, label="MILP")
plt.plot(CU_list, comm_cnn,  's--', lw=2, label="Hybrid")
plt.xlabel("Number of CUs (STs fixed)")
plt.ylabel("Aggregate Communication Performance")
plt.title("(b) Communication vs Number of CUs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("Plot-4 generated.")
