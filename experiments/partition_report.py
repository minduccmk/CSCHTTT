"""So sánh FedKT với cả 4 phương pháp còn lại theo từng cách chia dữ liệu (báo cáo TẤT CẢ cách chia, không chọn lọc).

Chỉ số (khai báo trước khi xem kết quả):
  local  = accuracy cục bộ trung bình 50 round     (Bảng 3)
  global = accuracy toàn cục trung bình 50 round   (Bảng 4)
  final  = accuracy toàn cục trung bình 10 round cuối; Client k = accuracy của model riêng từng client trên test của nó (Bảng 2)
Với mỗi ô: trung bình qua seed, và số seed FedKT cao nhất trong cả các phương pháp.
"""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import _path  # noqa: F401
from _path import ROOT

M = ["clientk", "fedavg", "fedprox", "fedcodl", "fedkt"]
R = defaultdict(dict)
for f in glob.glob(str(ROOT / "results/calib/*_b32_*.json")):
    ds, part, b, m, v, s = Path(f).stem.split("_")
    if (m == "fedkt" and v != "keep") or (m == "fedcodl" and v != "1.0"):
        continue
    h = json.load(open(f))
    if m == "clientk":
        R[(ds, part, m)][int(s[1:])] = {"final": h["client_acc"]}
    else:
        R[(ds, part, m)][int(s[1:])] = {"local": np.mean(h["local_acc"]), "global": np.mean(h["global_acc"]),
                                       "final": np.mean(h["global_acc"][-10:])}
rows = []
for (ds, part) in sorted({(k[0], k[1]) for k in R}):
    for metric in ("local", "global", "final"):
        ms = [m for m in M if metric in next(iter(R.get((ds, part, m), {0: {}}).values()), {})]
        if "fedkt" not in ms:
            continue
        seeds = sorted(set.intersection(*[set(R[(ds, part, m)]) for m in ms]))
        if not seeds:
            continue
        mean = {m: np.mean([R[(ds, part, m)][s][metric] for s in seeds]) for m in ms}
        wins = sum(all(R[(ds, part, "fedkt")][s][metric] > R[(ds, part, m)][s][metric] for m in ms if m != "fedkt") for s in seeds)
        best_other = max((m for m in ms if m != "fedkt"), key=mean.get)
        rows.append(dict(dataset=ds, partition=part, metric=metric, n_seeds=len(seeds),
                         **{m: round(mean[m], 2) for m in ms}, best_other=best_other,
                         kt_minus_best=round(mean["fedkt"] - mean[best_other], 2),
                         kt_highest_seeds=f"{wins}/{len(seeds)}"))
if not rows:
    raise SystemExit("Chưa có dữ liệu để so sánh. Cần trong results/calib/ cùng lúc kết quả fedkt (biến thể keep), "
                     "fedavg, fedprox, fedcodl (1.0), và clientk cho các cách chia. Xem hướng dẫn chạy calibrate.py, "
                     "tune_fedkt.py --baselines và clientk_grid.py.")
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/tables/partition_study.csv", index=False)
pd.set_option("display.width", 250)
print(df.to_string(index=False))