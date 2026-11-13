"""Chạy hàng loạt: dataset x {IID, Non-IID} x seed (Bảng 2-4), hoặc quét lambda (Bảng 5-6).

Mặc định chỉ chạy 3 dataset nhỏ (config.PROTOCOL['main_datasets']). GMSC rất lâu nên phải gọi rõ:
    --datasets gmsc

Chia việc:  python experiments/run_all.py --owner A   (FedAvg, FedKT)
            python experiments/run_all.py --owner B   (Client k, FedProx, FedCodl)
Kết quả cuối cùng theo quy trình chung:  --final   (đủ 5 seed theo config.PROTOCOL['seeds'])
"""
import argparse

import _path  # noqa: F401
from config import Config, DATASET_INFO, PROTOCOL, methods_of
from run_experiment import run

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=PROTOCOL["main_datasets"], choices=list(DATASET_INFO))
ap.add_argument("--settings", nargs="+", default=["iid", "noniid"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--final", action="store_true", help=f"dùng đủ seed của quy trình chung {PROTOCOL['seeds']}")
ap.add_argument("--owner", default="all", choices=["A", "B", "all"])
ap.add_argument("--rounds", type=int, default=Config.rounds)
ap.add_argument("--device", default="cpu")
ap.add_argument("--sweep_lambda", action="store_true",
                help="chỉ chạy FedKT (người A) với lambda 0.1..0.9, ghi vào results/raw_lambda")
a = ap.parse_args()

seeds = PROTOCOL["seeds"] if a.final else a.seeds
if a.sweep_lambda and a.owner == "B":
    raise SystemExit("Quét lambda chỉ có FedKT, thuộc người A (--owner A hoặc all).")

for seed in seeds:
    for ds in a.datasets:
        for st in a.settings:
            if a.sweep_lambda:
                for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    cfg = Config(rounds=a.rounds, seed=seed, lam=lam, device=a.device)
                    run(ds, st, ["fedkt"], cfg, "results/raw_lambda", tag=f"_lam{lam}")
            else:
                cfg = Config(rounds=a.rounds, seed=seed, device=a.device)
                run(ds, st, methods_of(a.owner), cfg, "results/raw")
