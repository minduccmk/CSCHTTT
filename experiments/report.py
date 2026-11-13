"""Tổng hợp cuối: mean±std nhiều seed, so sánh với bài, ảnh hưởng của batch size, kiểm tra FedKT > FedAvg theo từng seed.

Cấu hình CHÍNH lấy từ config.PROTOCOL (đã khoá): IID = config.partition_of('iid'), Non-IID = config.partition_of('noniid').
Kết quả có tham số chung khác giá trị đã khoá bị loại (trừ nhóm batch size 16/64, vốn là phân tích độ nhạy).

Xuất results/tables/final_*.csv và in kết quả.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from compare_paper import PAPER, METHODS, KEY
from config import FROZEN_HASH, FROZEN_SHARED, partition_of

# Cấu hình chính (khoá): IID = stratified, Non-IID = random; FC giữ lại; beta = 1.0
MAIN = {"iid": partition_of("iid"), "noniid": partition_of("noniid"), "fc": "keep", "beta": "1.0"}
# Cấu hình khớp số nhất trong lưới hiệu chỉnh (chỉ để THAM KHẢO: IID = random, Non-IID = stratified,
# tức là đảo nghĩa so với tên gọi, nên không dùng làm cấu hình chính)
FIT = {"iid": "random", "noniid": "iid", "fc": "keep", "beta": "1.0"}
BATCH_MAIN = FROZEN_SHARED["batch_size"]


def load_all():
    """R[(ds, part, batch, method, variant)][seed] = {'local_acc': mean, 'global_acc': mean}"""
    R = defaultdict(dict)
    skipped = []
    for f in glob.glob(str(ROOT / "results/calib/*.json")):
        ds, part, b, method, variant, s = Path(f).stem.split("_")
        batch = int(b[1:])
        h = json.load(open(f))
        # file mới có shared_hash; chỉ nhóm batch size khác giá trị khoá được phép lệch (độ nhạy)
        if h.get("shared_hash") not in (None, FROZEN_HASH) and batch == BATCH_MAIN:
            skipped.append(Path(f).name)
            continue
        R[(ds, part, batch, method, variant)][int(s[1:])] = {
            k: float(np.mean(h[k])) for k in ("local_acc", "global_acc")}
    if skipped:
        print(f"CẢNH BÁO: bỏ qua {len(skipped)} file có tham số chung khác giá trị đã khoá (ví dụ {skipped[0]})")
    return R


def seeds_vals(R, ds, part, batch, method, variant, tab):
    v = "-" if method in ("fedavg", "fedprox") else variant
    d = R.get((ds, part, batch, method, v), {})
    return {s: x[KEY[tab]] for s, x in d.items()}


def cfg_variants(cfg):
    return {"fedavg": "-", "fedprox": "-", "fedcodl": cfg["beta"], "fedkt": cfg["fc"]}


def table_for(R, cfg, batch, datasets):
    rows = []
    for tab in ("T3", "T4"):
        for ds in datasets:
            for st in ("iid", "noniid"):
                part = cfg[st]
                row = {"table": tab, "dataset": ds, "setting": st}
                ok = True
                per = {}
                for i, m in enumerate(METHODS):
                    sv = seeds_vals(R, ds, part, batch, m, cfg_variants(cfg)[m], tab)
                    if not sv:
                        ok = False
                        break
                    per[m] = sv
                    a = np.array(list(sv.values()))
                    row[f"{m}_paper"] = PAPER[tab][(ds, st)][i]
                    row[f"{m}_ours"] = f"{a.mean():.2f}±{a.std(ddof=1) if len(a) > 1 else 0:.2f}"
                    row[f"{m}_mean"] = a.mean()
                if not ok:
                    continue
                common = sorted(set(per["fedkt"]) & set(per["fedavg"]))
                row["n_seeds"] = len(common)
                row["kt_gt_avg_seeds"] = f"{sum(per['fedkt'][s] > per['fedavg'][s] for s in common)}/{len(common)}"
                diff = np.array([per["fedkt"][s] - per["fedavg"][s] for s in common])
                row["kt_minus_avg"] = f"{diff.mean():+.2f}±{diff.std(ddof=1) if len(diff) > 1 else 0:.2f}"
                paper = PAPER[tab][(ds, st)]
                row["MAE"] = float(np.mean([abs(row[f'{m}_mean'] - p) for m, p in zip(METHODS, paper)]))
                rows.append(row)
    return pd.DataFrame(rows)


def batch_effect(R, cfg, datasets, batches=(16, 32, 64)):
    """MAE theo batch size, chỉ tính trên các ô (bảng, setting) có đủ dữ liệu ở CẢ ba batch size."""
    dfs = {b: table_for(R, cfg, b, datasets) for b in batches}
    rows = []
    for ds in datasets:
        keys = None
        for b in batches:
            d = dfs[b][dfs[b].dataset == ds]
            ks = set(zip(d.table, d.setting))
            keys = ks if keys is None else keys & ks
        if not keys:
            continue
        row = {"dataset": ds, "n_cells": len(keys)}
        for b in batches:
            d = dfs[b][dfs[b].dataset == ds]
            d = d[[(t, s) in keys for t, s in zip(d.table, d.setting)]]
            row[b] = round(d.MAE.mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def main():
    R = load_all()
    ds_all = list(PAPER["T3"].keys())
    ds_all = sorted({k[0] for k in ds_all}, key=["australian", "loan", "taiwan"].index)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    for name, cfg in (("main", MAIN), ("fit", FIT)):
        df = table_for(R, cfg, BATCH_MAIN, ds_all)
        df.to_csv(out / f"final_{name}_batch{BATCH_MAIN}.csv", index=False)
        tag = "CHÍNH (đã khoá)" if name == "main" else "THAM KHẢO (khớp số nhất, đảo nghĩa IID/Non-IID)"
        print(f"\n=== Cấu hình {tag} {cfg} | batch {BATCH_MAIN} | MAE tổng = {df.MAE.mean():.2f} ===")
        cols = ["table", "dataset", "setting", "n_seeds"] + [f"{m}_{k}" for m in METHODS for k in ("paper", "ours")] \
               + ["kt_minus_avg", "kt_gt_avg_seeds"]
        print(df[cols].to_string(index=False))

    print("\n=== Ảnh hưởng batch size (cấu hình chính) — MAE so với bài, trên các ô có đủ cả 3 batch ===")
    bdf = batch_effect(R, MAIN, ds_all)
    bdf.to_csv(out / "final_batch_effect.csv")
    print(bdf.to_string())


if __name__ == "__main__":
    main()
