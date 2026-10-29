"""Chạy toàn bộ: 4 dataset x {IID, Non-IID} (Bảng 2-4), hoặc quét lambda (Bảng 5-6)."""
import argparse

import _path  # noqa: F401
from config import Config, DATASET_INFO, METHODS
from run_experiment import run

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(DATASET_INFO))
ap.add_argument("--settings", nargs="+", default=["iid", "noniid"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--rounds", type=int, default=50)
ap.add_argument("--device", default="cpu")
ap.add_argument("--sweep_lambda", action="store_true", help="chỉ chạy FedKT với lambda 0.1..0.9")
a = ap.parse_args()

for seed in a.seeds:
    for ds in a.datasets:
        for st in a.settings:
            if a.sweep_lambda:
                for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    cfg = Config(rounds=a.rounds, seed=seed, lam=lam, device=a.device)
                    run(ds, st, ["fedkt"], cfg, "results/raw_lambda", tag=f"_lam{lam}")
            else:
                cfg = Config(rounds=a.rounds, seed=seed, device=a.device)
                run(ds, st, METHODS, cfg, "results/raw")
