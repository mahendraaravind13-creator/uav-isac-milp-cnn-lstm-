import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.figure()
plt.plot(df["accuracy"], label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()
