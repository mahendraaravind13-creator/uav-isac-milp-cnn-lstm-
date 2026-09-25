import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.figure()
plt.plot(df["loss"], label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()
