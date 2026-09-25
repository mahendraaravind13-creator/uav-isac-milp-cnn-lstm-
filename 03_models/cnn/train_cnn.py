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

df_big_file = pd.read_excel(DATASET_DIR / "Copy of UAV90BIG1.xlsx")

# ================= CONFIG =================
TIME_STEPS = 15
LABEL_COL_INDEX = -1

C1, C2 = 7, 3
NUM_CLASSES = 3

EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
DROPOUT_RATE = 0.6

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
        X[:, :, C1:C1 + C2],
        X[:, :, C1 + C2:]
    )

def to_conv2d(x):
    return x[..., np.newaxis].astype(np.float32)

# ================= TEMPERATURE SOFTMAX =================
class TemperatureSoftmax(layers.Layer):
    def __init__(self, temperature=1.3):
        super().__init__()
        self.temperature = temperature

    def call(self, inputs):
        return tf.nn.softmax(inputs / self.temperature)

# ================= MODEL BLOCK =================
def build_branch(input_shape, filters):
    inp = layers.Input(shape=input_shape)

    x = layers.GaussianNoise(0.15)(inp)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 1),
        padding="same",
        activation="relu"
    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(16, activation="relu"))(x)

    return models.Model(inp, x)

# ================= DATA =================
X_raw, y_raw = clean_and_get_arrays(df_big_file)
X_envs, y_envs = make_envs(X_raw, y_raw)

y_envs = y_envs - 1
y_envs = np.clip(y_envs, 0, NUM_CLASSES - 1)

# ================= CLASS PRIOR BIAS =================
flat_labels = y_envs.flatten()
class_counts = np.bincount(flat_labels, minlength=NUM_CLASSES)
class_priors = class_counts / class_counts.sum()

#  strengthened bias
initial_bias = 1.5 * np.log(class_priors + 1e-8)

# ================= SPLIT + SCALE =================
p1, p2, p3 = split_columns(X_envs)

sc1, sc2, sc3 = StandardScaler(), StandardScaler(), StandardScaler()
p1 = sc1.fit_transform(p1.reshape(-1, p1.shape[-1])).reshape(p1.shape)
p2 = sc2.fit_transform(p2.reshape(-1, p2.shape[-1])).reshape(p2.shape)
p3 = sc3.fit_transform(p3.reshape(-1, p3.shape[-1])).reshape(p3.shape)

p1i, p2i, p3i = map(to_conv2d, [p1, p2, p3])

# ================= MODEL =================
b1 = build_branch(p1i.shape[1:], filters=6)
b2 = build_branch(p2i.shape[1:], filters=4)
b3 = build_branch(p3i.shape[1:], filters=4)

i1 = layers.Input(shape=p1i.shape[1:])
i2 = layers.Input(shape=p2i.shape[1:])
i3 = layers.Input(shape=p3i.shape[1:])

o = layers.Concatenate(axis=-1)([
    b1(i1),
    b2(i2),
    b3(i3)
])

o = layers.TimeDistributed(layers.Dense(32, activation="relu"))(o)
o = layers.Dropout(DROPOUT_RATE)(o)

o = layers.TimeDistributed(
    layers.Dense(
        NUM_CLASSES,
        bias_initializer=tf.keras.initializers.Constant(initial_bias)
    )
)(o)

o = TemperatureSoftmax(temperature=1.3)(o)

model = models.Model([i1, i2, i3], o)

# ================= COMPILE =================
optimizer = optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=0.3
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================
csv_logger = CSVLogger("training_log.csv", append=False)

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

model.save("model_final_cnn.h5")
