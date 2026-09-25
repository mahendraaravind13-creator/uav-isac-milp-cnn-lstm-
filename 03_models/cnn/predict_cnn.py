import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ================= CUSTOM LAYER =================
class TemperatureSoftmax(tf.keras.layers.Layer):
    def __init__(self, temperature=1.3, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        return tf.nn.softmax(inputs / self.temperature)

# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

MODEL_PATH = BASE_DIR / "model_final_cnn.h5"
DEC_FILE = DATASET_DIR / "master_table_FULLdec.xlsx"
OUT_FILE = DATASET_DIR / "full_dec_cnn.xlsx"

# ================= LOAD MODEL =================
model = load_model(
    MODEL_PATH,
    custom_objects={"TemperatureSoftmax": TemperatureSoftmax}
)

# ================= CONFIG =================
TIME_STEPS = 15
C1, C2 = 7, 3
NUM_CLASSES = 3

# ================= DROP COLS =================
cols_to_drop = [
    'CU1_x','CU1_y','CU2_x','CU2_y','CU3_x','CU3_y','CU4_x','CU4_y','CU5_x','CU5_y',
    'CU6_x','CU6_y','CU7_x','CU7_y','CU8_x','CU8_y','CU9_x','CU9_y','CU10_x','CU10_y',
    'ST1_x','ST1_y','ST2_x','ST2_y','ST3_x','ST3_y','ST4_x','ST4_y','ST5_x','ST5_y',
    'ST6_x','ST6_y','ST7_x','ST7_y','ST8_x','ST8_y','ST9_x','ST9_y','ST10_x','ST10_y'
]

# ================= HELPERS =================
def pad_to_multiple(X):
    r = X.shape[0] % TIME_STEPS
    if r != 0:
        pad = TIME_STEPS - r
        X = np.vstack([X, np.zeros((pad, X.shape[1]))])
    return X, r

def make_envs(X):
    X, _ = pad_to_multiple(X)
    n = X.shape[0] // TIME_STEPS
    return X.reshape(n, TIME_STEPS, X.shape[1])

def split_columns(X):
    return (
        X[:, :, :C1],
        X[:, :, C1:C1 + C2],
        X[:, :, C1 + C2:]
    )

def to_conv2d(x):
    # (N, T, F) → (N, T, F, 1)
    return x[..., np.newaxis].astype(np.float32)

# ================= LOAD DEC FILE =================
df_dec = pd.read_excel(DEC_FILE)
original_rows = df_dec.shape[0]

# Drop CU / ST columns
df_dec = df_dec.drop(columns=[c for c in cols_to_drop if c in df_dec.columns])

# Keep copy for output
df_out = df_dec.copy()

# Remove label column if present
if df_dec.columns[-1].lower().startswith("label"):
    df_dec = df_dec.iloc[:, :-1]

# Clean
df_dec = df_dec.replace([np.inf, -np.inf], 1e12).fillna(0.0)

X = df_dec.values.astype(float)

# ================= PREPROCESS =================
X, pad_rows = pad_to_multiple(X)
X_envs = make_envs(X)

p1, p2, p3 = split_columns(X_envs)

# IMPORTANT: fit scalers again (same as training)
sc1, sc2, sc3 = StandardScaler(), StandardScaler(), StandardScaler()
p1 = sc1.fit_transform(p1.reshape(-1, p1.shape[-1])).reshape(p1.shape)
p2 = sc2.fit_transform(p2.reshape(-1, p2.shape[-1])).reshape(p2.shape)
p3 = sc3.fit_transform(p3.reshape(-1, p3.shape[-1])).reshape(p3.shape)

p1i, p2i, p3i = map(to_conv2d, [p1, p2, p3])

# ================= PREDICT =================
pred_probs = model.predict(
    [p1i, p2i, p3i],
    batch_size=32,
    verbose=1
)

pred_labels = np.argmax(pred_probs, axis=-1).reshape(-1)

# Remove padding
pred_labels = pred_labels[:original_rows]

# Convert 0,1,2 → 1,2,3
pred_labels = pred_labels + 1

# ================= SAVE =================
df_out["Predicted_Label"] = pred_labels
df_out.to_excel(OUT_FILE, index=False)

