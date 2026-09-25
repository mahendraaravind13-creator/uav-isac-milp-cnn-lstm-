"""Test-set metrics for a prediction file (MILP labels vs model labels).

Usage:
    python evaluate_predictions.py <prediction_file.xlsx> [more files ...]

Each file needs the columns Label, Predicted_Label, PsiComm and Rmin, with
15 consecutive rows per trajectory. Run on full_dec_cnn.xlsx it reproduces
the CNN numbers in the README (86.5% accuracy, 40.2% meeting R_min). Run on
master_table_FULLdec_with_predictions.xlsx it gives the Hybrid numbers.
Reading is fastest with `pip install python-calamine`; openpyxl also works.
"""
import sys

import numpy as np
import pandas as pd

WAYPOINTS = 15


def read(path):
    cols = ["PsiComm", "Rmin", "Label", "Predicted_Label"]
    try:
        return pd.read_excel(path, engine="calamine", usecols=cols)
    except (ImportError, ValueError):
        return pd.read_excel(path, usecols=cols)


def evaluate(path):
    df = read(path)
    n = len(df) // WAYPOINTS
    y = df["Label"].to_numpy(int)[: n * WAYPOINTS]
    p = df["Predicted_Label"].to_numpy(int)[: n * WAYPOINTS]
    print(f"\n{path}\n  trajectories: {n:,}")
    print(f"  per-waypoint accuracy vs MILP: {(y == p).mean() * 100:.1f}%")
    for k, name in [(1, "Sensing"), (2, "Comm"), (3, "Joint")]:
        m = y == k
        print(f"  recall {name:<8}: {(p[m] == k).mean() * 100:.1f}%")
    yy, pp = y.reshape(n, WAYPOINTS), p.reshape(n, WAYPOINTS)
    print(f"  exact 15-waypoint match: {(yy == pp).all(axis=1).mean() * 100:.1f}%")
    psi = df["PsiComm"].to_numpy(float)[: n * WAYPOINTS].reshape(n, WAYPOINTS)
    rmin = df["Rmin"].to_numpy(float)[: n * WAYPOINTS].reshape(n, WAYPOINTS)[:, 0]
    bits_milp = (psi * np.isin(yy, [2, 3])).sum(axis=1)
    bits_pred = (psi * np.isin(pp, [2, 3])).sum(axis=1)
    print(f"  meets R_min: MILP {(bits_milp >= rmin).mean() * 100:.1f}%, model {(bits_pred >= rmin).mean() * 100:.1f}%")
    print(f"  mean bits delivered, model / MILP: {bits_pred.mean() / bits_milp.mean() * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for f in sys.argv[1:]:
        evaluate(f)
