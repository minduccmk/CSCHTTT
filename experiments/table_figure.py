"""Xuất BẢNG KẾT QUẢ thành hình PNG (từ results/tables/partition_study.csv).
  python experiments/table_figure.py
-> results/figures/table_local_global.png   (Local và Global, 4 mô hình liên kết)
-> results/figures/table_final.png          (Final, có thêm Client k)
Số cao nhất mỗi hàng được tô cam và in đậm."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from _path import ROOT

NAVY, ORANGE = "#1c1f3f", "#d97706"
DS = {"australian": "Australian", "loan": "Loan data", "taiwan": "Taiwan", "gmscfe": "GMSC", "gmsc": "GMSC (10 đặc trưng)"}
SET = {"iid": "IID", "random": "Non-IID"}
NAME = {"clientk": "Client k", "fedavg": "FedAvg", "fedprox": "FedProx", "fedcodl": "FedCodl", "fedkt": "FedKT"}

d = pd.read_csv(ROOT / "results/tables/partition_study.csv")
d = d[d.partition.isin(SET)]
order = [k for k in DS if k in set(d.dataset)]
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)


def draw(groups, fname, title):
    """groups: list[(tên nhóm, metric, [model...])]"""
    models = [m for _, _, ms in groups for m in ms]
    cw = [1.9, 1.1] + [1.15] * len(models)
    nrow = len(order) * 2
    rh, hh = 0.55, 0.95
    W, H = sum(cw), hh + nrow * rh + 1.6
    fig, ax = plt.subplots(figsize=(W * 0.95, H * 0.95)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.3, title, ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
    top = H - 0.7
    x = 0
    heads = [("Tập dữ liệu", cw[0]), ("Cách chia", cw[1])]
    for t, w in heads:
        ax.add_patch(Rectangle((x, top - hh), w, hh, fc=NAVY, ec="white", lw=1.2))
        ax.text(x + w / 2, top - hh / 2, t, ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
        x += w
    for g, _, ms in groups:
        gw = cw[2] * len(ms)
        ax.add_patch(Rectangle((x, top - hh / 2), gw, hh / 2, fc="#34386b", ec="white", lw=1.2))
        ax.text(x + gw / 2, top - hh / 4, g, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        for m in ms:
            ax.add_patch(Rectangle((x, top - hh), cw[2], hh / 2, fc=NAVY, ec="white", lw=1.2))
            ax.text(x + cw[2] / 2, top - 3 * hh / 4, NAME[m], ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
            x += cw[2]
    r = 0
    for ds in order:
        for st in SET:
            y = top - hh - (r + 1) * rh
            band = "#f4f5fa" if (r // 2) % 2 == 0 else "white"
            x = 0
            for j, (txt, w) in enumerate([(DS[ds] if st == "iid" else "", cw[0]), (SET[st], cw[1])]):
                ax.add_patch(Rectangle((x, y), w, rh, fc=band, ec="#d8dae6", lw=0.8))
                ax.text(x + w / 2, y + rh / 2, txt if j == 0 else SET[st], ha="center", va="center", fontsize=11.5, fontweight="bold" if j == 0 else "normal")
                x += w
            for _, metric, ms in groups:
                row = d[(d.dataset == ds) & (d.partition == st) & (d.metric == metric)]
                vals = {m: (float(row[m].iloc[0]) if len(row) and pd.notna(row[m].iloc[0]) else None) for m in ms}
                best = max([v for v in vals.values() if v is not None], default=None)
                for m in ms:
                    v = vals[m]
                    hot = v is not None and v == best
                    ax.add_patch(Rectangle((x, y), cw[2], rh, fc="#fdeedd" if hot else band, ec="#d8dae6", lw=0.8))
                    ax.text(x + cw[2] / 2, y + rh / 2, "-" if v is None else f"{v:.2f}", ha="center", va="center", fontsize=11.5,
                            fontweight="bold" if hot else "normal", color=ORANGE if hot else "#222")
                    x += cw[2]
            r += 1
    ax.add_patch(Rectangle((0, top - hh - nrow * rh - 0.05), W, 0.05, fc="#f0921e", ec="none"))
    n = int(d.n_seeds.max())
    ax.text(W / 2, 0.3, f"Accuracy (%), trung bình trên tối đa {n} seed. Ô cam = cao nhất trong hàng. \"-\" = chưa chạy.", ha="center", fontsize=10, color="#666", style="italic")
    fig.savefig(ROOT / "results/figures" / fname, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("Đã lưu results/figures/" + fname)


L4 = ["fedavg", "fedprox", "fedcodl", "fedkt"]
draw([("Local (TB 50 vòng)", "local", L4), ("Global (TB 50 vòng)", "global", L4)], "table_local_global.png", "Độ chính xác cục bộ và toàn cục")
draw([("Final (10 vòng cuối)", "final", ["clientk"] + L4)], "table_final.png", "Độ chính xác cuối (final), có Client k")