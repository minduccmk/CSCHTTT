"""Chạy một cấu hình (dataset x setting) với các phương pháp được chọn.

Mỗi phương pháp ghi ra MỘT FILE RIÊNG:
    results/raw/{dataset}_{setting}_{method}_seed{seed}{tag}.json
nên người A (FedAvg, FedKT) và người B (Client k, FedProx, FedCodl) không bao giờ ghi đè kết quả của nhau.
"""
import argparse
import json
from dataclasses import asdict

import _path  # noqa: F401
from _path import ROOT

from config import (Config, DATASET_INFO, METHODS, OWNER, is_frozen, methods_of, partition_of,
                    shared_diff, shared_hash, validate)
from data import load_dataset, partition
from federated import make_selections, run_federated, run_client_k

import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def run(dataset, setting, methods, cfg, out_dir, tag=""):
    validate(cfg)
    X, y = load_dataset(dataset, ROOT / "Dataset")
    K = DATASET_INFO[dataset]["K"]
    clients = partition(X, y, K, setting, cfg.seed, cfg.train_ratio, cfg.dirichlet_alpha)
    selections = make_selections(K, cfg.rounds, cfg.participation, cfg.seed)
    frozen = is_frozen(cfg)
    if not frozen:
        print(f"CẢNH BÁO: tham số chung khác giá trị đã khoá {shared_diff(cfg)}. "
              f"Kết quả sẽ bị loại khỏi bảng chính.", flush=True)
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in methods:
        print(f"[{dataset}|{setting}|seed={cfg.seed}] {m} (người {OWNER[m]}) ...", flush=True)
        if m == "client_k":
            res = run_client_k(clients, X.shape[1], cfg)
        else:
            res = run_federated(m, clients, X.shape[1], cfg, selections)
            print(f"    final global acc={res['global_acc'][-1]:.2f}  "
                  f"mean local acc={sum(res['local_acc']) / len(res['local_acc']):.2f}")
        payload = {"dataset": dataset, "setting": setting, "partition": partition_of(setting),
                   "seed": cfg.seed, "method": m, "in_dim": int(X.shape[1]),
                   "config": asdict(cfg), "shared_hash": shared_hash(cfg), "frozen": frozen,
                   "result": res}
        path = out / f"{dataset}_{setting}_{m}_seed{cfg.seed}{tag}.json"
        path.write_text(json.dumps(payload))
        print("saved", path)
        results[m] = res
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_INFO))
    ap.add_argument("--setting", default="iid", choices=["iid", "noniid", "dirichlet", "random", "sorted"])
    ap.add_argument("--owner", default="all", choices=["A", "B", "all"],
                    help="A = fedavg, fedkt; B = client_k, fedprox, fedcodl")
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS,
                    help="ghi đè --owner nếu muốn chọn phương pháp cụ thể")
    ap.add_argument("--rounds", type=int, default=Config.rounds)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=Config.lam)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = Config(rounds=a.rounds, seed=a.seed, lam=a.lam, device=a.device)
    run(a.dataset, a.setting, a.methods or methods_of(a.owner), cfg, a.out, a.tag)
