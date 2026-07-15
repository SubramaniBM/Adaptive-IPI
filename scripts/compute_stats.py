import json
import numpy as np

dirs = ["exp039", "exp041", "exp042", "exp043"]
metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auroc": [], "auprc": [], "ece": []}

for d in dirs:
    with open(f"experiments/{d}/summary.json", "r") as f:
        data = json.load(f)
        for k in metrics.keys():
            metrics[k].append(data[k])

print("--- MULTI-SEED AGGREGATION ---")
for k, vals in metrics.items():
    mean = np.mean(vals)
    std = np.std(vals)
    print(f"**{k.capitalize()}**: `{mean:.4f} ± {std:.4f}`")
