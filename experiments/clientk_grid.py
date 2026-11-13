"""Chạy baseline Client k (học cục bộ thuần túy) với cache, cùng lưới (dataset, cách chia, seed) với calibrate.py.

  python experiments/clientk_grid.py --datasets australian loan taiwan --parts iid random dirichlet sorted --seeds 0 1 2 3 4
Kết quả: results/calib/{ds}_{part}_b32_clientk_-_s{seed}.json  ({"client_acc": ...})
"""
import argparse
import json
import multiprocessing as mp

import _path  # noqa: F401
from calibrate import OUT, _clients


def run(job):
    import torch
    torch.set_num_threads(1)
    from config import Config, validate
    from federated import run_client_k
    ds, part, seed = job
    f = OUT / f"{ds}_{part}_b32_clientk_-_s{seed}.json"
    if f.exists():
        return str(f)
    cfg = validate(Config(seed=seed, batch_size=32))
    clients, dim = _clients(ds, part, seed)
    f.write_text(json.dumps(run_client_k(clients, dim, cfg)))
    return str(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["australian", "loan"])
    ap.add_argument("--parts", nargs="+", default=["iid", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, p, s) for s in a.seeds for d in a.datasets for p in a.parts]
    print(len(jobs), "jobs", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"{i}/{len(jobs)}", flush=True)
