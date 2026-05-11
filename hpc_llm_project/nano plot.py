import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("baseline_results.csv")

# Extract runtimes
cpu_time = df[df["device"] == "cpu"]["runtime_sec"].values[0]
gpu_time = df[(df["device"] == "cuda") & (df["distributed"] == False)]["runtime_sec"].values[0]
ddp_time = df[(df["device"] == "cuda") & (df["distributed"] == True)]["runtime_sec"].values[0]

# Labels
labels = ["CPU", "1 GPU", "3 GPU (DDP)"]
times = [cpu_time, gpu_time, ddp_time]

# Plot
plt.figure()
plt.bar(labels, times)
plt.xlabel("Configuration")
plt.ylabel("Runtime (seconds)")
plt.title("TinyGPT Scaling Performance")
plt.show()