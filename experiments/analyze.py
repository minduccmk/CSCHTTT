"""Tổng hợp kết quả -> bảng (CSV) tương ứng Table 2-6 và hình tương ứng Fig. 3-7."""
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _path  # noqa: F401
from _path import ROOT

FED = ["fedavg", "fedprox", "fedcodl", "fedkt"]
DS = ["australian", "loan", "taiwan", "gmsc"]
ST = ["iid", "noniid"]


def load(pattern):
    runs = defaultdict(list)
    for f in glob.glob(str(ROOT / pattern)):
        r = json.load(open(f))
        runs[(r["dataset"], r["setting"], r["config"]["lam"])].append(r)
    return runs


def mean_over_seeds(rs, method, key):
    arr = [np.mean(r["methods"][method][key]) for r in rs if method in r["methods"]
           and key in r["methods"][method]]
    return float(np.mean(arr)) if arr else np.nan


def final_over_seeds(rs, method, key):
    arr = [r["methods"][method][key][-1] for r in rs if method in r["methods"]
           and key in r["methods"][method]]
    return float(np.mean(arr)) if arr else np.nan


def main():
    out_t = ROOT / "results" / "tables"
    out_f = ROOT / "results" / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    runs = load("results/raw/*.json")
    if not runs:
        print("Chưa có kết quả trong results/raw/. Hãy chạy run_all.py trước.")
        return

    # ---- Table 2: accuracy vòng cuối (global acc) + Client k ----
    rows2, rows3, rows4 = [], [], []
    for ds in DS:
        for st in ST:
            rs = next((v for (d, s, _), v in runs.items() if d == ds and s == st), None)
            if rs is None:
                continue
            ck = [r["methods"]["client_k"]["client_acc"] for r in rs if "client_k" in r["methods"]]
            r2 = {"dataset": ds, "setting": st, "client_k": np.mean(ck) if ck else np.nan}
            r2_mean = dict(r2)
            r3 = {"dataset": ds, "setting": st}
            r4 = {"dataset": ds, "setting": st}
            for m in FED:
                r2[m] = final_over_seeds(rs, m, "global_acc")          # vòng cuối
                r2_mean[m] = mean_over_seeds(rs, m, "global_acc")      # trung bình các round
                r3[m] = mean_over_seeds(rs, m, "local_acc")
                r4[m] = mean_over_seeds(rs, m, "global_acc")
            rows2.append(r2)
            rows3.append(r3)
            rows4.append(r4)
    pd.DataFrame(rows2).round(2).to_csv(out_t / "table2_final_round.csv", index=False)
    pd.DataFrame(rows3).round(2).to_csv(out_t / "table3_avg_local_acc.csv", index=False)
    pd.DataFrame(rows4).round(2).to_csv(out_t / "table4_avg_global_acc.csv", index=False)
    print(pd.DataFrame(rows3).round(2).to_string(index=False))

    # ---- Fig 3-6: train loss / local acc / global acc ----
    for ds in DS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        ok = False
        for i, st in enumerate(ST):
            rs = next((v for (d, s, _), v in runs.items() if d == ds and s == st), None)
            if rs is None:
                continue
            ok = True
            for j, (key, lab) in enumerate([("train_loss", "Train loss"),
                                            ("local_acc", "Local accuracy"),
                                            ("global_acc", "Global accuracy")]):
                for m in FED:
                    curves = [r["methods"][m][key] for r in rs if m in r["methods"]]
                    if curves:
                        axes[i, j].plot(np.mean(curves, axis=0), label=m)
                axes[i, j].set_title(f"{ds} ({'IID' if st == 'iid' else 'Non-IID'})")
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
        rs = next((v for (d, s, _), v in runs.items() if d == ds and s == "noniid"), None)
        if rs is None:
            continue
        w = 0.2
        for i, m in enumerate(FED):
            loc = np.mean([np.std(r["methods"][m]["local_acc"][1:]) for r in rs if m in r["methods"]])
            glo = np.mean([np.std(r["methods"][m]["global_acc"][1:]) for r in rs if m in r["methods"]])
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
    lam_runs = load("results/raw_lambda/*.json")
    if lam_runs:
        r5, r6 = [], []
        for ds in DS:
            for st in ST:
                a, b = {"dataset": ds, "setting": st}, {"dataset": ds, "setting": st}
                for lam in [0.9, 0.7, 0.5, 0.3, 0.1]:
                    rs = lam_runs.get((ds, st, lam))
                    if rs:
                        a[lam] = mean_over_seeds(rs, "fedkt", "local_acc")
                        b[lam] = mean_over_seeds(rs, "fedkt", "global_acc")
                r5.append(a)
                r6.append(b)
        pd.DataFrame(r5).round(2).to_csv(out_t / "table5_lambda_local.csv", index=False)
        pd.DataFrame(r6).round(2).to_csv(out_t / "table6_lambda_global.csv", index=False)
    print("Đã lưu bảng vào", out_t, "và hình vào", out_f)


if __name__ == "__main__":
    main()
