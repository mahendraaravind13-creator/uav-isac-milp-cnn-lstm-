import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tensorflow.keras.callbacks import CSVLogger
import pickle

# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

print("Loading BIG file for training (ALL ROWS)...")
df_big_file = pd.read_excel(DATASET_DIR / "Copy of UAV90BIG1.xlsx")
print("Total rows loaded:", df_big_file.shape[0])

# ================= CONFIG =================
TIME_STEPS = 15
LABEL_COL_INDEX = -1
C1, C2 = 7, 3
NUM_CLASSES = 3

EPOCHS = 500        
BATCH_SIZE = 16
LEARNING_RATE = 3e-4  
BATCH_NORM = True
DROPOUT_RATE = 0.5   
# ================= HELPERS =================
def clean_and_get_arrays(df):
    df = df.replace([np.inf, -np.inf], 1e12).fillna(0.0)
    y = df.iloc[:, LABEL_COL_INDEX].astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def pad_to_multiple(X, y):
    r = X.shape[0] % TIME_STEPS
    if r != 0:
        pad = TIME_STEPS - r
        X = np.vstack([X, np.zeros((pad, X.shape[1]))])
        y = np.hstack([y, np.zeros(pad, dtype=int)])
    return X, y

def make_envs(X, y):
    X, y = pad_to_multiple(X, y)
    n = X.shape[0] // TIME_STEPS
    return X.reshape(n, TIME_STEPS, X.shape[1]), y.reshape(n, TIME_STEPS)

def split_columns(X):
    return (
        X[:, :, :C1],
        X[:, :, C1:C1+C2],
        X[:, :, C1+C2:]
    )

def to_conv3d(x):
    return x.reshape(x.shape[0], x.shape[1], 1, x.shape[2], 1).astype(np.float32)

def build_branch(shape, name, f=(32,16)):
    i = layers.Input(shape=shape)

    x = layers.Conv3D(f[0], (3,1,3), padding="same", activation="relu")(i)
    if BATCH_NORM: x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Conv3D(f[1], (1,1,3), padding="same", activation="relu")(x)
    if BATCH_NORM: x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)

    return models.Model(i, x, name=name)

# ================= DATA PREP =================
X_raw, y_raw = clean_and_get_arrays(df_big_file)
X_envs, y_envs = make_envs(X_raw, y_raw)

y_envs = y_envs - 1
y_envs = np.clip(y_envs, 0, NUM_CLASSES - 1)

print("Final label range:", y_envs.min(), y_envs.max())
print("Total environments:", X_envs.shape[0])

p1, p2, p3 = split_columns(X_envs)

sc1, sc2, sc3 = StandardScaler(), StandardScaler(), StandardScaler()
p1 = sc1.fit_transform(p1.reshape(-1, p1.shape[-1])).reshape(p1.shape)
p2 = sc2.fit_transform(p2.reshape(-1, p2.shape[-1])).reshape(p2.shape)
p3 = sc3.fit_transform(p3.reshape(-1, p3.shape[-1])).reshape(p3.shape)

p1i, p2i, p3i = map(to_conv3d, [p1, p2, p3])

# ================= MODEL =================
b1 = build_branch(p1i.shape[1:], "b1")
b2 = build_branch(p2i.shape[1:], "b2", (16,8))
b3 = build_branch(p3i.shape[1:], "b3", (16,8))

i1, i2, i3 = [layers.Input(shape=x.shape[1:]) for x in [p1i, p2i, p3i]]
o = layers.Concatenate(axis=-1)([b1(i1), b2(i2), b3(i3)])

o = layers.TimeDistributed(layers.Dense(128, activation="relu"))(o)
if BATCH_NORM: o = layers.BatchNormalization()(o)
o = layers.Dropout(DROPOUT_RATE)(o)

o = layers.TimeDistributed(layers.Dense(NUM_CLASSES))(o)
o = layers.Activation("softmax")(o)

model = models.Model([i1, i2, i3], o)

model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
csv_logger = CSVLogger("training_log.csv", append=False)

print("\n--- TRAINING ON FULL BIG DATASET ---")
history = model.fit(
    [p1i, p2i, p3i],
    y_envs,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=2,
    callbacks=[csv_logger]
)

# ================= SAVE =================
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

model.save("model_trained_on_big.h5")
print("MODEL SAVED: model_trained_on_big.h5")
