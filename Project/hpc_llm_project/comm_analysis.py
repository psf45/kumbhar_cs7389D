import pandas as pd

files = [
    "results/baseline_big.csv",
    "results/ddp_big.csv",
    "results/ddp_2node.csv",
]

rows = []

for f in files:
    df = pd.read_csv(f)
    row = df.iloc[-1]

    total = row["runtime_sec"]
    compute = row["forward_sec"] + row["backward_sec"]
    non_compute = total - compute
    comm_ratio = non_compute / total * 100

    rows.append({
        "file": f,
        "world_size": row["world_size"],
        "runtime_sec": total,
        "compute_sec": compute,
        "non_compute_sync_sec": non_compute,
        "non_compute_percent": comm_ratio,
        "final_loss": row["final_loss"]
    })

out = pd.DataFrame(rows)
out.to_csv("results/communication_analysis.csv", index=False)
print(out)
