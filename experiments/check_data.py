"""Kiểm tra dataset đã đặt đúng chỗ chưa; in số mẫu, dim, imbalance ratio so với Table 1."""
import _path  # noqa: F401
from _path import ROOT

from config import DATASET_INFO
from data import load_dataset

PAPER = {  # samples, dim, imbalance ratio
    "australian": (690, 42, 1.25),
    "loan": (1225, 28, 2.8),
    "taiwan": (30000, 91, 3.52),
    "gmsc": (150000, 67, 13.96),
}

print("ROOT:", ROOT)
print("DATASET_INFO:", DATASET_INFO)

for name in DATASET_INFO:
    print(f"\nDataset: {name}")
    print("Expected files:", DATASET_INFO[name]["files"])
    print("Directory exists:", (ROOT / "Dataset").exists())

for name in DATASET_INFO:
    try:
        X, y = load_dataset(name, ROOT / "Dataset")
        pos = int(y.sum())
        neg = len(y) - pos
        ratio = max(pos, neg) / max(min(pos, neg), 1)
        p = PAPER[name]
        print(f"{name:11s} samples={len(y)} (paper {p[0]}) dim={X.shape[1]} (paper {p[1]}) "
              f"imbalance={ratio:.2f} (paper {p[2]})")
    except Exception as e:
        print(f"{name:11s} LỖI: {e}")
