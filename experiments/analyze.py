"""Tổng hợp kết quả -> bảng (CSV) tương ứng Table 2-6 và hình tương ứng Fig. 3-7."""
import matplotlib
# <-- THÊM: dùng để tính p-value giữa 2 phương pháp (cùng dataset, setting, seed) nếu muốn
from scipy.stats import ttest_rel
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import glob
import argparse
import matplotlib.pyplot as plt
from _path import ROOT
from config import FROZEN_HASH
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


matplotlib.use("Agg")

import _path  # noqa: F401

FED = ["fedavg", "fedprox", "fedcodl", "fedkt"]
DS = ["australian", "loan", "taiwan", "gmsc"]
ST = ["iid", "noniid"]


def load(pattern, include_nonfrozen=False):
    """Đọc kết quả, hỗ trợ cả 2 định dạng và GỘP các phương pháp của cùng (dataset, setting, seed):
       - mới : 1 file / phương pháp  ({"method":..., "result":...}) -> người A và B chạy riêng rồi gộp ở đây
       - cũ  : 1 file chứa mọi phương pháp ({"methods": {...}})
    File có tham số chung khác giá trị đã khoá bị bỏ qua (trừ khi include_nonfrozen=True)."""
    merged, skipped = {}, []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        r = json.load(open(f))
        if r.get("shared_hash") is not None and r["shared_hash"] != FROZEN_HASH and not include_nonfrozen:
            skipped.append(Path(f).name)
            continue
        key = (r["dataset"], r["setting"], r["config"]
               ["lam"], r["config"]["seed"])
        d = merged.setdefault(key, {"dataset": r["dataset"], "setting": r["setting"],
                                    "config": r["config"],
                                    "seed": r["config"]["seed"],
                                    "methods": {}})
        if "methods" in r:
            d["methods"].update(r["methods"])
        else:
            d["methods"][r["method"]] = r["result"]
    if skipped:
        print(f"CẢNH BÁO: bỏ qua {len(skipped)} file có tham số chung khác giá trị đã khoá "
              f"(ví dụ {skipped[0]}). Dùng --include-nonfrozen để đưa vào.")
    runs = defaultdict(list)
    for (ds, st, lam, _seed), d in merged.items():
        runs[(ds, st, lam)].append(d)
    return runs


def mean_over_seeds(rs, method, key):
    arr = [np.mean(r["methods"][method][key]) for r in rs if method in r["methods"]
           and key in r["methods"][method]]
    return float(np.mean(arr)) if arr else np.nan


def final_over_seeds(rs, method, key):
    arr = [r["methods"][method][key][-1] for r in rs if method in r["methods"]
           and key in r["methods"][method]]
    return float(np.mean(arr)) if arr else np.nan


def mean_metric_over_seeds(rs, method, history_key, metric):
    """Trung bình metric qua các round, sau đó trung bình qua các seed."""
    seed_means = []

    for r in rs:
        history = r["methods"].get(method, {}).get(history_key, [])

        values = [
            item[metric]
            for item in history
            if isinstance(item, dict)
            and metric in item
            and item[metric] is not None
        ]

        if values:
            seed_means.append(float(np.mean(values)))

    return float(np.mean(seed_means)) if seed_means else np.nan

def metric_getter(history_key, metric):
    def g(res):
        vals = [it[metric] for it in res.get(history_key, [])
                if isinstance(it, dict) and it.get(metric) is not None]
        return float(np.mean(vals)) if vals else None
    return g


GETTERS = {
    "final_global_acc": lambda res: res["global_acc"][-1],
    "avg_local_acc":    lambda res: float(np.mean(res["local_acc"])),
    "avg_global_acc":   lambda res: float(np.mean(res["global_acc"])),
}
for _scope in ("local_metrics", "global_metrics"):
    for _m in ("precision", "recall", "f1", "auc", "ks"):
        GETTERS[f"{_scope}_{_m}"] = metric_getter(_scope, _m)


def seed_values(rs, method, getter):
    """{seed: giá trị} của một phương pháp; bỏ seed thiếu hoặc nan."""
    out = {}
    for r in rs:
        res = r["methods"].get(method)
        if res is None:
            continue
        try:
            v = getter(res)
        except (KeyError, IndexError, TypeError):
            continue
        if v is not None and not np.isnan(v):
            out[r["seed"]] = float(v)
    return out


def export_stats(runs, out_t):
    # 1) Bảng dài: mỗi dòng = một (dataset, setting, method, metric, seed)
    rows = []
    for ds in DS:
        for st in ST:
            rs = next((v for (d, s, _), v in runs.items() if d == ds and s == st), None)
            if rs is None:
                continue
            for name, getter in GETTERS.items():
                for m in FED:
                    for seed, v in seed_values(rs, m, getter).items():
                        rows.append(dict(dataset=ds, setting=st, method=m,
                                         metric=name, seed=seed, value=v))
            for seed, v in seed_values(rs, "client_k", lambda res: res["client_acc"]).items():
                rows.append(dict(dataset=ds, setting=st, method="client_k",
                                 metric="client_acc", seed=seed, value=v))
    long = pd.DataFrame(rows)
    long.to_csv(out_t / "seed_values_long.csv", index=False)

    # 2) Trung bình ± độ lệch chuẩn, cùng bố cục dataset/setting x method
    summ = (long.groupby(["dataset", "setting", "method", "metric"])["value"]
                .agg(mean="mean", std="std", n="count").reset_index())
    summ["cell"] = summ.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f} (n={int(r['n'])})", axis=1)
    for metric, sub in summ.groupby("metric"):
        sub.pivot(index=["dataset", "setting"], columns="method", values="cell") \
           .to_csv(out_t / f"table_{metric}_mean_std.csv")

    # 3) t-test ghép cặp: FedKT so với từng baseline, ghép theo seed
    sig = []
    for (ds, st, metric), sub in long.groupby(["dataset", "setting", "metric"]):
        wide = sub.pivot(index="seed", columns="method", values="value")
        if "fedkt" not in wide:
            continue
        for base in [m for m in FED if m != "fedkt" and m in wide]:
            pair = wide[["fedkt", base]].dropna()
            n = len(pair)
            if n < 3:
                continue
            d = pair["fedkt"] - pair[base]
            if np.allclose(d, 0):
                t = p = np.nan
            else:
                t, p = ttest_rel(pair["fedkt"], pair[base])
            sig.append(dict(dataset=ds, setting=st, metric=metric, baseline=base,
                            n_seeds=n, mean_diff=d.mean(), std_diff=d.std(ddof=1),
                            t=t, p=p))
    pd.DataFrame(sig).round(4).to_csv(out_t / "significance.csv", index=False)


def main(include_nonfrozen=False):
    out_t = ROOT / "results" / "tables"
    out_f = ROOT / "results" / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    runs = load("results/raw/*.json", include_nonfrozen)
    if not runs:
        print("Chưa có kết quả trong results/raw/. Hãy chạy run_all.py trước.")
        return

    # ---- Table 2: accuracy vòng cuối (global acc) + Client k ----
    rows2, rows3, rows4 = [], [], []
    for ds in DS:
        for st in ST:
            rs = next((v for (d, s, _), v in runs.items()
                      if d == ds and s == st), None)
            if rs is None:
                continue
            ck = [r["methods"]["client_k"]["client_acc"]
                  for r in rs if "client_k" in r["methods"]]
            r2 = {"dataset": ds, "setting": st,
                  "client_k": np.mean(ck) if ck else np.nan}
            r3 = {"dataset": ds, "setting": st}
            r4 = {"dataset": ds, "setting": st}
            for m in FED:
                r2[m] = final_over_seeds(
                    rs, m, "global_acc")          # vòng cuối
                r3[m] = mean_over_seeds(rs, m, "local_acc")
                r4[m] = mean_over_seeds(rs, m, "global_acc")
            rows2.append(r2)
            rows3.append(r3)
            rows4.append(r4)
    pd.DataFrame(rows2).round(2).to_csv(
        out_t / "table2_final_round.csv", index=False)
    pd.DataFrame(rows3).round(2).to_csv(
        out_t / "table3_avg_local_acc.csv", index=False)
    pd.DataFrame(rows4).round(2).to_csv(
        out_t / "table4_avg_global_acc.csv", index=False)
    print(pd.DataFrame(rows3).round(2).to_string(index=False))

    # Các bảng metrics mới: local_metrics và global_metrics
    metrics = ["precision", "recall", "f1", "auc", "ks"]

    for scope in ["local_metrics", "global_metrics"]:
        for metric in metrics:
            rows = []

            for ds in DS:
                for st in ST:
                    rs = next(
                        (v for (d, s, _), v in runs.items()
                         if d == ds and s == st),
                        None
                    )

                    if rs is None:
                        continue

                    row = {"dataset": ds, "setting": st}

                    for m in FED:
                        row[m] = mean_metric_over_seeds(
                            rs, m, scope, metric
                        )

                    rows.append(row)

            df = pd.DataFrame(rows).round(2)
            filename = f"table_{scope}_{metric}.csv"
            df.to_csv(out_t / filename, index=False)

            print(f"Đã lưu {filename}")
            
    export_stats(runs, out_t)

    # ---- Fig 3-6: train loss / local acc / global acc ----
    for ds in DS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        ok = False
        for i, st in enumerate(ST):
            rs = next((v for (d, s, _), v in runs.items()
                      if d == ds and s == st), None)
            if rs is None:
                continue
            ok = True
            for j, (key, lab) in enumerate([("train_loss", "Train loss"),
                                            ("local_acc", "Local accuracy"),
                                            ("global_acc", "Global accuracy")]):
                for m in FED:
                    curves = [r["methods"][m][key]
                              for r in rs if m in r["methods"]]
                    if curves:
                        axes[i, j].plot(np.mean(curves, axis=0), label=m)
                axes[i, j].set_title(
                    f"{ds} ({'IID' if st == 'iid' else 'Non-IID'})")
                axes[i, j].set_xlabel("# Rounds")
                axes[i, j].set_ylabel(lab)
                axes[i, j].legend(fontsize=7)
        if ok:
            fig.tight_layout()
            fig.savefig(out_f / f"fig_{ds}.png", dpi=150)
        plt.close(fig)

    # ---- Fig 7: std của accuracy qua các round (Non-IID) ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, ds in zip(axes, DS):
        rs = next((v for (d, s, _), v in runs.items()
                  if d == ds and s == "noniid"), None)
        if rs is None:
            continue
        w = 0.2
        for i, m in enumerate(FED):
            loc = np.mean([np.std(r["methods"][m]["local_acc"][1:])
                          for r in rs if m in r["methods"]])
            glo = np.mean([np.std(r["methods"][m]["global_acc"][1:])
                          for r in rs if m in r["methods"]])
            ax.bar([0 + i * w, 1 + i * w], [loc, glo], width=w, label=m)
        ax.set_xticks([0.3, 1.3])
        ax.set_xticklabels(["Local acc", "Global acc"])
        ax.set_title(ds)
        ax.set_ylabel("Standard deviation")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_f / "fig7_std_noniid.png", dpi=150)
    plt.close(fig)

    # ---- Table 5-6: quét lambda ----
    lam_runs = load("results/raw_lambda/*.json", include_nonfrozen)
    if lam_runs:
        r5, r6 = [], []
        for ds in DS:
            for st in ST:
                a, b = {"dataset": ds, "setting": st}, {
                    "dataset": ds, "setting": st}
                for lam in [0.9, 0.7, 0.5, 0.3, 0.1]:
                    rs = lam_runs.get((ds, st, lam))
                    if rs:
                        a[lam] = mean_over_seeds(rs, "fedkt", "local_acc")
                        b[lam] = mean_over_seeds(rs, "fedkt", "global_acc")
                r5.append(a)
                r6.append(b)
        pd.DataFrame(r5).round(2).to_csv(
            out_t / "table5_lambda_local.csv", index=False)
        pd.DataFrame(r6).round(2).to_csv(
            out_t / "table6_lambda_global.csv", index=False)
    print("Đã lưu bảng vào", out_t, "và hình vào", out_f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-nonfrozen", action="store_true",
                    help="đưa cả kết quả có tham số chung khác giá trị đã khoá vào bảng")
    main(ap.parse_args().include_nonfrozen)
