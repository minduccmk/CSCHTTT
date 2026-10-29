"""Chạy một cấu hình (dataset x setting) với các phương pháp được chọn."""
import argparse
import json
from dataclasses import asdict

import _path  # noqa: F401
from _path import ROOT

from config import Config, DATASET_INFO, METHODS
from data import load_dataset, partition
from federated import make_selections, run_federated, run_client_k


def run(dataset, setting, methods, cfg, out_dir, tag=""):
    X, y = load_dataset(dataset, ROOT / "Dataset")
    K = DATASET_INFO[dataset]["K"]
    clients = partition(X, y, K, setting, cfg.seed, cfg.train_ratio, cfg.dirichlet_alpha)
    selections = make_selections(K, cfg.rounds, cfg.participation, cfg.seed)
    result = {"dataset": dataset, "setting": setting, "in_dim": int(X.shape[1]),
              "config": asdict(cfg), "methods": {}}
    for m in methods:
        print(f"[{dataset}|{setting}|seed={cfg.seed}] {m} ...", flush=True)
        if m == "client_k":
            result["methods"][m] = run_client_k(clients, X.shape[1], cfg)
        else:
            h = run_federated(m, clients, X.shape[1], cfg, selections)
            print(f"    final global acc={h['global_acc'][-1]:.2f}  "
                  f"mean local acc={sum(h['local_acc'])/len(h['local_acc']):.2f}")
            result["methods"][m] = h
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset}_{setting}_seed{cfg.seed}{tag}.json"
    path.write_text(json.dumps(result))
    print("saved", path)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_INFO))
    ap.add_argument("--setting", default="iid", choices=["iid", "noniid", "dirichlet"])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = Config(rounds=a.rounds, seed=a.seed, lam=a.lam, device=a.device)
    run(a.dataset, a.setting, a.methods, cfg, a.out, a.tag)
