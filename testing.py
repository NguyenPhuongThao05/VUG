from recbole_cdr.quick_start.quick_start import run_recbole_cdr

# Run VUG_CLFM model (combining VUG's virtual user generation with CLFM's shared/specific embeddings)
res = run_recbole_cdr(
    model='VUG_CLFM',
    config_file_list=['./recbole_cdr/properties/dataset/Amazon.yaml',
                     './recbole_cdr/properties/model/VUG_CLFM.yaml']
)

import json, numpy as np, torch
import matplotlib.pyplot as plt

def _to_float(x):
    # handle numpy / torch scalars
    if isinstance(x, (np.generic,)):
        return float(x.item())
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return float(x.item())
    if isinstance(x, (int, float)):
        return float(x)
    return None

def flatten_numeric(d, prefix=""):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            items.extend(flatten_numeric(v, name))
    else:
        val = _to_float(d)
        if val is not None:
            items.append((prefix, val))
    return items


# Flatten and keep only numeric metrics
best_items = flatten_numeric(res.get("best_valid_result", {}))
test_items = flatten_numeric(res.get("test_result", {}))

# (Optional) sort for consistent display
best_items = sorted(best_items)
test_items = sorted(test_items)

def barplot_metrics(title, items):
    if not items:
        print(f"No numeric metrics to plot for: {title}")
        return
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]
    x = range(len(labels))
    plt.figure(figsize=(8,4))
    plt.bar(x, vals)
    plt.title(title)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

barplot_metrics("Best Valid Metrics", best_items)
barplot_metrics("Test Metrics",      test_items)
