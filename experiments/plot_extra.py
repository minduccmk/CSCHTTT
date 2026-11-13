"""Vẽ đồ thị bổ sung từ các bảng đã có:
  fig_lambda_sweep.png      : độ chính xác của FedKT theo lambda, so với FedAvg (từ results/tables/lambda_sweep.csv)
  fig_partition_study.png   : 5 mô hình theo từng cách chia dữ liệu (từ results/tables/partition_study.csv)

Chạy trước: sweep_lambda.py + sweep_report.py (hình 1) và clientk_grid.py + partition_report.py (hình 2).
  python experiments/plot_extra.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _path  # noqa: F401
from _path import ROOT

T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
F.mkdir(parents=True, exist_ok=True)
COL = {"clientk": "#bdbdbd", "fedavg": "#9ecae1", "fedprox": "#4292c6", "fedcodl": "#08519c", "fedkt": "#d95f02"}
NAME = {"clientk": "Client k", "fedavg": "FedAvg", "fedprox": "FedProx", "fedcodl": "FedCodl", "fedkt": "FedKT"}
DSN = {"australian": "Australian", "loan": "Loan", "taiwan": "Taiwan"}


def fig_lambda(variant="keep"):
    f = T / "lambda_sweep.csv"
    if not f.exists() or f.stat().st_size < 10:
        print("Bỏ qua fig_lambda_sweep: lambda_sweep.csv trống hoặc chưa có. Chạy sweep_lambda.py rồi sweep_report.py.")
        return
    df = pd.read_csv(f)
    df = df[df["variant"] == variant]
    cells = [("australian", "iid"), ("australian", "noniid"), ("loan", "iid"), ("loan", "noniid")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, squeeze=False)
    for r, (tab, tname) in enumerate((("T3", "Local"), ("T4", "Global"))):
        for c, (ds, st) in enumerate(cells):
            ax = axes[r, c]
            g = df[(df.dataset == ds) & (df.setting == st) & (df.table == tab)].sort_values("lam")
            ax.plot(g.lam, g.fedkt, "o-", color=COL["fedkt"], lw=2, label="FedKT")
            ax.axhline(g.fedavg.iloc[0], color=COL["fedprox"], ls="--", lw=1.5, label="FedAvg")
            p = g.paper_fedkt.dropna()
            if len(p):
                ax.plot([0.3], [p.iloc[0]], "k^", ms=8, label="FedKT trong bài (λ=0.3)")
            ax.set_title(f"{DSN[ds]} {'IID' if st == 'iid' else 'Non-IID'} | {tname}", fontsize=10)
            ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("Accuracy (%)")
            if r == 1:
                ax.set_xlabel("λ")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"FedKT theo λ so với FedAvg (biến thể {variant}, trung bình 5 seed)", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(F / "fig_lambda_sweep.png", dpi=150)
    plt.close(fig)


def fig_partition():
    f = T / "partition_study.csv"
    if not f.exists() or f.stat().st_size < 10:
        print("Bỏ qua fig_partition_study: partition_study.csv trống hoặc chưa có. Chạy partition_report.py sau khi "
              "results/calib/ đã có đủ kết quả (xem thông báo của script đó).")
        return
    df = pd.read_csv(f)
    parts = ["iid", "random", "dirichlet", "sorted"]
    pname = {"iid": "IID", "random": "Non-IID\n(random)", "dirichlet": "Dirichlet", "sorted": "Sorted"}
    dss = [d for d in DSN if d in set(df.dataset)]
    metrics = [("local", "Local (TB 50 vòng)"), ("global", "Global (TB 50 vòng)"), ("final", "Final")]
    fig, axes = plt.subplots(len(metrics), len(dss), figsize=(5.2 * len(dss), 3.6 * len(metrics)), squeeze=False)
    for r, (m, mname) in enumerate(metrics):
        for c, ds in enumerate(dss):
            ax = axes[r, c]
            g = df[(df.dataset == ds) & (df.metric == m)].set_index("partition").reindex(parts)
            methods = [k for k in COL if k in g.columns and g[k].notna().any()]
            w = 0.8 / len(methods)
            for i, k in enumerate(methods):
                ax.bar(np.arange(len(parts)) + (i - (len(methods) - 1) / 2) * w, g[k].values, w,
                       color=COL[k], label=NAME[k])
            ax.set_xticks(range(len(parts)))
            ax.set_xticklabels([pname[p] for p in parts], fontsize=8)
            lo = np.nanmin(g[methods].values)
            ax.set_ylim(max(0, lo - 5), min(100, np.nanmax(g[methods].values) + 3))
            ax.set_title(f"{DSN[ds]} | {mname}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel("Accuracy (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[2, 0].get_legend_handles_labels()
    seen = dict(zip(l + l2, h + h2))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle("So sánh 5 mô hình theo cách chia dữ liệu (batch 32; trục y không bắt đầu từ 0)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(F / "fig_partition_study.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_lambda()
    fig_partition()
    print("Đã ghi", F / "fig_lambda_sweep.png", "và", F / "fig_partition_study.png")