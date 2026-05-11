import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# -------------------------
# 1. Strong scaling runtime
# -------------------------
runtime_data = {
    "Config": ["1 GPU", "3 GPUs", "6 GPUs"],
    "Runtime": [9.3948, 3.4188, 7.0924],
    "GPUs": [1, 3, 6],
}

df = pd.DataFrame(runtime_data)

plt.figure(figsize=(7, 5))
plt.bar(df["Config"], df["Runtime"])
plt.title("Strong Scaling Runtime")
plt.xlabel("Configuration")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
plt.savefig("plots/strong_scaling_runtime.png", dpi=300)
plt.close()

# -------------------------
# 2. Speedup
# -------------------------
baseline = 9.3948
df["Speedup"] = baseline / df["Runtime"]

plt.figure(figsize=(7, 5))
plt.bar(df["Config"], df["Speedup"])
plt.axhline(y=1, linestyle="--")
plt.title("Speedup Relative to 1 GPU")
plt.xlabel("Configuration")
plt.ylabel("Speedup")
plt.tight_layout()
plt.savefig("plots/speedup.png", dpi=300)
plt.close()

# -------------------------
# 3. Parallel efficiency
# -------------------------
df["Efficiency"] = df["Speedup"] / df["GPUs"] * 100

plt.figure(figsize=(7, 5))
plt.bar(df["Config"], df["Efficiency"])
plt.axhline(y=100, linestyle="--")
plt.title("Parallel Efficiency")
plt.xlabel("Configuration")
plt.ylabel("Efficiency (%)")
plt.tight_layout()
plt.savefig("plots/parallel_efficiency.png", dpi=300)
plt.close()

# -------------------------
# 4. Communication analysis
# -------------------------
comm = pd.read_csv("results/communication_analysis.csv")

plt.figure(figsize=(8, 5))
plt.bar(comm["world_size"].astype(str), comm["compute_sec"], label="Compute Time")
plt.bar(
    comm["world_size"].astype(str),
    comm["non_compute_sync_sec"],
    bottom=comm["compute_sec"],
    label="Non-compute / Sync Overhead"
)
plt.title("Compute vs Non-compute Runtime Breakdown")
plt.xlabel("World Size")
plt.ylabel("Time (seconds)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/communication_breakdown.png", dpi=300)
plt.close()

# -------------------------
# 5. I/O stall experiment
# -------------------------
io_data = {
    "Config": ["No I/O Delay", "Simulated I/O Delay"],
    "Runtime": [8.8062, 9.3334],
}

io = pd.DataFrame(io_data)

plt.figure(figsize=(7, 5))
plt.bar(io["Config"], io["Runtime"])
plt.title("Impact of Simulated I/O Delay")
plt.xlabel("Configuration")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
plt.savefig("plots/io_stall_experiment.png", dpi=300)
plt.close()

print("Plots saved in plots/")
print(df)
