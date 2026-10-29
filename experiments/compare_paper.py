"""So sánh kết quả calibrate với Bảng 3 và 4 của bài, chọn cấu hình gần bài nhất.

Điểm = sai số tuyệt đối trung bình (điểm phần trăm) trên các ô (method x setting) của Bảng 3 và 4.
"""
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from _path import ROOT

METHODS = ["fedavg", "fedprox", "fedcodl", "fedkt"]
PAPER = {  # (dataset, setting) -> [FedAvg, FedProx, FedCodl, FedKT]
    "T3": {("australian", "iid"): [78.07, 78.43, 80.42, 82.93], ("australian", "noniid"): [82.78, 82.64, 78.57, 87.64],
           ("loan", "iid"): [67.33, 66.88, 67.42, 72.21], ("loan", "noniid"): [65.29, 65.67, 69.33, 72.88],
           ("taiwan", "iid"): [78.73, 79.16, 77.82, 81.20], ("taiwan", "noniid"): [78.01, 78.99, 77.78, 80.85]},
    "T4": {("australian", "iid"): [82.43, 82.51, 84.44, 86.63], ("australian", "noniid"): [86.24, 85.80, 87.43, 89.66],
           ("loan", "iid"): [70.52, 70.73, 73.71, 73.79], ("loan", "noniid"): [69.70, 69.33, 73.51, 73.53],
           ("taiwan", "iid"): [80.99, 80.57, 81.64, 81.90], ("taiwan", "noniid"): [80.62, 80.53, 81.62, 81.61]},
}
KEY = {"T3": "local_acc", "T4": "global_acc"}


def load(batch=32):
    """res[(ds, part, method, variant)] -> {'local_acc': [seed means], 'global_acc': [...]}"""
    res = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(ROOT / "results/calib/*.json")):
        name = Path(f).stem
        ds, part, b, method, variant, s = name.split("_")
        if int(b[1:]) != batch:
            continue
        h = json.load(open(f))
        for k in ("local_acc", "global_acc"):
            res[(ds, part, method, variant)][k].append(float(np.mean(h[k])))
    return res


def val(res, ds, part, method, variant, tab):
    v = res.get((ds, part, method, variant if method in ("fedcodl", "fedkt") else "-"), {}).get(KEY[tab])
    return (np.mean(v), np.std(v), len(v)) if v else None


def score_combo(res, datasets, p_iid, p_non, fc, beta):
    errs, kt_best = [], 0
    n_cmp = 0
    for ds in datasets:
        for st, part in (("iid", p_iid), ("noniid", p_non)):
            for tab in ("T3", "T4"):
                vals = []
                for m, v in zip(METHODS, ["-", "-", beta, fc]):
                    x = val(res, ds, part, m, v, tab)
                    if x is None:
                        break
                    vals.append(x[0])
                else:
                    paper = PAPER[tab][(ds, st)]
                    errs += [abs(a - b) for a, b in zip(vals, paper)]
                    n_cmp += 1
                    kt_best += int(np.argmax(vals) == 3)
    return (np.mean(errs) if errs else np.nan), kt_best, n_cmp


def main():
    res = load()
    datasets = sorted({k[0] for k in res})
    parts = sorted({k[1] for k in res})
    print("datasets:", datasets, "parts:", parts)
    rows = []
    for p_iid, p_non, fc, beta in itertools.product(parts, parts, ["keep", "global", "reinit"], ["0.1", "1.0"]):
        mae, kb, n = score_combo(res, datasets, p_iid, p_non, fc, beta)
        rows.append({"iid_part": p_iid, "noniid_part": p_non, "fc_mode": fc, "codl_beta": beta,
                     "MAE_vs_paper": mae, "fedkt_best_cells": f"{kb}/{n}"})
    df = pd.DataFrame(rows).dropna().sort_values("MAE_vs_paper")
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out / "calibration_scores.csv", index=False)
    print(df.head(12).round(2).to_string(index=False))
    best = df.iloc[0]
    print("\nBEST:", dict(best))

    # bảng đối chiếu từng ô với cấu hình tốt nhất
    lines = []
    for tab, title in (("T3", "Local acc (Bảng 3)"), ("T4", "Global acc (Bảng 4)")):
        for ds in datasets:
            for st, part in (("iid", best.iid_part), ("noniid", best.noniid_part)):
                row = {"table": tab, "dataset": ds, "setting": st}
                for i, (m, v) in enumerate(zip(METHODS, ["-", "-", best.codl_beta, best.fc_mode])):
                    x = val(res, ds, part, m, v, tab)
                    row[f"{m}_paper"] = PAPER[tab][(ds, st)][i]
                    row[f"{m}_ours"] = f"{x[0]:.2f}±{x[1]:.2f}" if x else "n/a"
                lines.append(row)
    cmp_df = pd.DataFrame(lines)
    cmp_df.to_csv(out / "compare_with_paper.csv", index=False)
    print()
    print(cmp_df.to_string(index=False))

    # ảnh hưởng riêng từng lựa chọn (trung bình MAE)
    print("\nẢnh hưởng trung bình của từng lựa chọn lên MAE:")
    for col in ["iid_part", "noniid_part", "fc_mode", "codl_beta"]:
        print(df.groupby(col)["MAE_vs_paper"].mean().round(2).to_string(), "\n")


if __name__ == "__main__":
    main()
