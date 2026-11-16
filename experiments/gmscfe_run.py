"""GMSC với đặc trưng rời rạc hoá (gmscfe): chạy đủ 5 mô hình, cùng giao thức, nhiều seed."""
import argparse
import multiprocessing as mp
import _path  # noqa: F401
from calibrate import OUT, run_job

M = [("fedavg", "-"), ("fedprox", "-"), ("fedcodl", "0.1"), ("fedcodl", "1.0"), ("fedkt", "keep")]
ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="+", default=["iid", "random"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--procs", type=int, default=2)
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
jobs = [("gmscfe", p, 32, m, v, s) for s in a.seeds for p in a.parts for m, v in M]
with mp.Pool(a.procs) as pool:
    for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
        print(f"{i}/{len(jobs)} {r}", flush=True)