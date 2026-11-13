"""Chọn biến thể FedKT gần số của bài nhất, rồi kiểm chứng trên seed chưa dùng để chọn.

Mục tiêu: sai số tuyệt đối trung bình (MAE) của FedKT so với số FedKT của bài trên 8 ô
(Australian, Loan) x (IID, Non-IID) x (Bảng 3 local, Bảng 4 global), và FedKT có cao nhất trong ô đó không
(so với FedAvg, FedProx, FedCodl beta=1.0 cùng seed).

  python experiments/tune_report.py select  --seeds 0 1 2 3 4
  python experiments/tune_report.py heldout --seeds 5 6 7 8 9 --variants keep keep-sum-h2
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from compare_paper import PAPER

DS = ["australian", "loan"]
MAPPINGS = {
    "chính (IID=iid, Non-IID=random)": {"iid": "iid", "noniid": "random"},
    "tham khảo (đảo nghĩa)": {"iid": "random", "noniid": "iid"},
}
KEY = {"T3": "local_acc", "T4": "global_acc"}


def load(batch=32):
    R = defaultdict(dict)
    for f in glob.glob(str(ROOT / "results/calib/*.json")):
        ds, part, b, method, variant, s = Path(f).stem.split("_")
        if int(b[1:]) != batch:
            continue
        h = json.load(open(f))
        R[(ds, part, method, variant)][int(s[1:])] = {k: float(np.mean(h[k])) for k in ("local_acc", "global_acc")}
    return R


def cell(R, ds, part, method, variant, tab, seeds):
    d = R.get((ds, part, method, variant), {})
    v = [d[s][KEY[tab]] for s in seeds if s in d]
    return (np.mean(v), len(v)) if len(v) == len(seeds) else (None, len(v))


def evaluate(R, variant, mapping, seeds):
    errs, best, n, rows = [], 0, 0, []
    for ds in DS:
        for st in ("iid", "noniid"):
            part = mapping[st]
            for tab in ("T3", "T4"):
                kt, c = cell(R, ds, part, "fedkt", variant, tab, seeds)
                if kt is None:
                    return None
                base = [cell(R, ds, part, m, v, tab, seeds)[0] for m, v in
                        (("fedavg", "-"), ("fedprox", "-"), ("fedcodl", "1.0"))]
                paper = PAPER[tab][(ds, st)][3]
                best_base = None if any(b is None for b in base) else max(base)
                is_best = None if best_base is None else kt > best_base
                errs.append(abs(kt - paper))
                if is_best is not None:
                    n += 1
                    best += int(is_best)
                rows.append({"dataset": ds, "setting": st, "table": tab, "paper": paper, "ours": round(kt, 2),
                             "err": round(kt - paper, 2), "best_baseline": None if best_base is None else round(best_base, 2),
                             "fedkt_highest": is_best})
    return {"MAE": float(np.mean(errs)), "highest": f"{best}/{n}" if n else "n/a", "cells": pd.DataFrame(rows)}


def variants_in(R):
    return sorted({k[3] for k in R if k[2] == "fedkt"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "heldout"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--variants", nargs="+", default=None)
    a = ap.parse_args()
    R = load()
    variants = a.variants or variants_in(R)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, mp in MAPPINGS.items():
        rows = []
        for v in variants:
            r = evaluate(R, v, mp, a.seeds)
            if r:
                rows.append({"variant": v, "MAE_vs_paper": round(r["MAE"], 3), "fedkt_highest_cells": r["highest"]})
        if not rows:
            print(f"\n[{name}] chưa đủ dữ liệu cho seed {a.seeds}")
            continue
        df = pd.DataFrame(rows).sort_values("MAE_vs_paper")
        df.to_csv(out / f"tune_{a.mode}_{'main' if name.startswith('chính') else 'ref'}.csv", index=False)
        print(f"\n=== {a.mode.upper()} | seed {a.seeds} | cách chia {name} ===")
        print(df.to_string(index=False))
    if a.mode == "heldout" and a.variants:
        for v in a.variants:
            r = evaluate(R, v, MAPPINGS["chính (IID=iid, Non-IID=random)"], a.seeds)
            if r:
                print(f"\n--- Chi tiết từng ô: {v} (cách chia chính, seed {a.seeds}) | MAE {r['MAE']:.2f} | FedKT cao nhất {r['highest']}")
                print(r["cells"].to_string(index=False))


if __name__ == "__main__":
    main()
