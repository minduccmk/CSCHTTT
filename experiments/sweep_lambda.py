"""Người A: quét lambda cho FedKT và so với FedAvg từng seed (Bảng 5-6 + điều kiện FedKT > FedAvg).

  python experiments/sweep_lambda.py --datasets australian loan --seeds 0 1 2 3 4 --variants keep global-mean-h1
Kết quả: results/calib/*.json (có cache) và results/tables/lambda_sweep_*.csv
"""
import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from calibrate import OUT, run_job

LAMS = [0.1, 0.3, 0.5, 0.7, 0.9]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["australian", "loan"])
    ap.add_argument("--parts", nargs="+", default=["iid", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["keep"])
    ap.add_argument("--lams", nargs="+", type=float, default=LAMS)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in a.seeds:
        for ds in a.datasets:
            for p in a.parts:
                jobs.append((ds, p, 32, "fedavg", "-", s))
                for v in a.variants:
                    for lam in a.lams:
                        jobs.append((ds, p, 32, "fedkt", f"{v}@{lam}", s))
    print(len(jobs), "jobs", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_job, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
