"""Chạy lưới các lựa chọn bài báo không nêu rõ, nhiều seed, có cache.

Mỗi job = (dataset, partition, batch, method, variant, seed) -> 1 file JSON trong results/calib/.
  variant: fedcodl -> beta ; fedkt -> fc_mode ; còn lại -> "-"
"""
import argparse
import itertools
import json
import multiprocessing as mp
from dataclasses import replace
from functools import lru_cache

from _path import ROOT

OUT = ROOT / "results" / "calib"


@lru_cache(maxsize=4)
def _data(ds):
    from data import load_dataset
    return load_dataset(ds, ROOT / "Dataset")


@lru_cache(maxsize=8)
def _clients(ds, part, seed):
    from config import DATASET_INFO, Config
    from data import partition
    X, y = _data(ds)
    c = Config()
    return partition(X, y, DATASET_INFO[ds]["K"], part, seed, c.train_ratio, c.dirichlet_alpha), X.shape[1]


def run_job(job):
    import torch
    torch.set_num_threads(1)
    from config import Config, DATASET_INFO
    from federated import make_selections, run_federated
    ds, part, batch, method, variant, seed = job
    f = OUT / f"{ds}_{part}_b{batch}_{method}_{variant}_s{seed}.json"
    if f.exists():
        return str(f)
    cfg = Config(seed=seed, batch_size=batch)
    if method == "fedcodl":
        cfg = replace(cfg, codl_beta=float(variant))
    if method == "fedkt":
        cfg = replace(cfg, fc_mode=variant)
    clients, dim = _clients(ds, part, seed)
    sel = make_selections(DATASET_INFO[ds]["K"], cfg.rounds, cfg.participation, seed)
    h = run_federated(method, clients, dim, cfg, sel)
    f.write_text(json.dumps(h))
    return str(f)


def build_jobs(ds, parts, batches, seeds):
    methods = [("fedavg", "-"), ("fedprox", "-"), ("fedcodl", "0.1"), ("fedcodl", "1.0"),
               ("fedkt", "keep"), ("fedkt", "global"), ("fedkt", "reinit")]
    return [(ds, p, b, m, v, s) for s in seeds for p in parts for b in batches for m, v in methods]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parts", nargs="+", default=["iid", "random", "dirichlet", "sorted"])
    ap.add_argument("--batches", nargs="+", type=int, default=[32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(a.dataset, a.parts, a.batches, a.seeds)
    print(f"{len(jobs)} jobs", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
