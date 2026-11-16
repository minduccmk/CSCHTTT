"""Xuất BẢNG λ (giống Table 5/6 của bài) thành PNG, từ results/tables/lambda_sweep.csv.
  python experiments/sweep_report.py        # tạo lambda_sweep.csv (nếu chưa có)
  python experiments/table_lambda_figure.py [--variant keep]
-> results/figures/table_lambda.png
Ô cam = λ cho accuracy cao nhất trong hàng. Cột FedAvg là mốc tham chiếu (không phụ thuộc λ)."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from _path import ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="keep"); a = ap.parse_args()
NAVY, ORANGE = "#1c1f3f", "#d97706"
DSN = {"australian": "Australian", "loan": "Loan data", "taiwan": "Taiwan", "gmscfe": "GMSC", "gmsc": "GMSC (10 đặc trưng)"}
d = pd.read_csv(ROOT / "results/tables/lambda_sweep.csv")
d = d[d.variant == a.variant]
lams = sorted(d.lam.unique())
dss = [k for k in DSN if k in set(d.dataset)]
sets = [("iid", "IID"), ("noniid", "Non-IID")]
groups = [("Độ chính xác cục bộ (Local)", "T3"), ("Độ chính xác toàn cục (Global)", "T4")]
cw = [1.9, 1.1] + [1.0] * ((len(lams) + 1) * 2)
nrow = len(dss) * 2; rh, hh = 0.55, 0.95
W, H = sum(cw), hh + nrow * rh + 1.6
fig, ax = plt.subplots(figsize=(W * 0.95, H * 0.95)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.text(W / 2, H - 0.3, "Ảnh hưởng của λ đến độ chính xác của FedKT (so với FedAvg)", ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
top = H - 0.7
x = 0
for t, w in [("Tập dữ liệu", cw[0]), ("Cách chia", cw[1])]:
    ax.add_patch(Rectangle((x, top - hh), w, hh, fc=NAVY, ec="white", lw=1.2))
    ax.text(x + w / 2, top - hh / 2, t, ha="center", va="center", color="white", fontsize=11.5, fontweight="bold"); x += w
for g, _ in groups:
    gw = len(lams) + 1
    ax.add_patch(Rectangle((x, top - hh / 2), gw, hh / 2, fc="#34386b", ec="white", lw=1.2))
    ax.text(x + gw / 2, top - hh / 4, g, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    for l in lams:
        ax.add_patch(Rectangle((x, top - hh), 1, hh / 2, fc=NAVY, ec="white", lw=1.2))
        ax.text(x + 0.5, top - 3 * hh / 4, f"λ={l:g}", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold"); x += 1
    ax.add_patch(Rectangle((x, top - hh), 1, hh / 2, fc="#555", ec="white", lw=1.2))
    ax.text(x + 0.5, top - 3 * hh / 4, "FedAvg", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold"); x += 1
r = 0
for ds in dss:
    for key, lab in sets:
        y = top - hh - (r + 1) * rh
        band = "#f4f5fa" if (r // 2) % 2 == 0 else "white"
        x = 0
        for j, (txt, w) in enumerate([(DSN[ds] if key == "iid" else "", cw[0]), (lab, cw[1])]):
            ax.add_patch(Rectangle((x, y), w, rh, fc=band, ec="#d8dae6", lw=0.8))
            ax.text(x + w / 2, y + rh / 2, txt, ha="center", va="center", fontsize=11.5, fontweight="bold" if j == 0 else "normal"); x += w
        for _, tb in groups:
            sub = d[(d.dataset == ds) & (d.setting == key) & (d.table == tb)].set_index("lam")
            vals = {l: (float(sub.fedkt[l]) if l in sub.index else None) for l in lams}
            best = max([v for v in vals.values() if v is not None], default=None)
            for l in lams:
                v = vals[l]; hot = v is not None and v == best
                ax.add_patch(Rectangle((x, y), 1, rh, fc="#fdeedd" if hot else band, ec="#d8dae6", lw=0.8))
                ax.text(x + 0.5, y + rh / 2, "-" if v is None else f"{v:.2f}", ha="center", va="center", fontsize=11.5,
                        fontweight="bold" if hot else "normal", color=ORANGE if hot else "#222"); x += 1
            fa = float(sub.fedavg.iloc[0]) if len(sub) else None
            ax.add_patch(Rectangle((x, y), 1, rh, fc="#ececec", ec="#d8dae6", lw=0.8))
            ax.text(x + 0.5, y + rh / 2, "-" if fa is None else f"{fa:.2f}", ha="center", va="center", fontsize=11.5, color="#555", style="italic"); x += 1
        r += 1
ax.add_patch(Rectangle((0, top - hh - nrow * rh - 0.05), W, 0.05, fc="#f0921e", ec="none"))
ax.text(W / 2, 0.3, f"Accuracy (%), TB 50 vòng, trung bình các seed. FedKT cấu hình '{a.variant}'. Ô cam = λ tốt nhất trong hàng. '-' = chưa chạy.",
        ha="center", fontsize=10, color="#666", style="italic")
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results/figures/table_lambda.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Đã lưu results/figures/table_lambda.png")