"""Bảng chính: chỉ 2 cách chia của bài (IID, Non-IID) -> kiểm tra FedKT có cao nhất không."""
import pandas as pd
from _path import ROOT
d = pd.read_csv(ROOT / "results/tables/partition_study.csv")
d = d[d.partition.isin(["iid", "random"])].copy()
d["setting"] = d.partition.map({"iid": "IID", "random": "Non-IID"})
d["FedKT cao nhất (TB)"] = d.kt_minus_best > 0
cols = ["dataset", "setting", "metric", "n_seeds", "fedavg", "fedprox", "fedcodl", "fedkt", "kt_minus_best", "kt_highest_seeds", "FedKT cao nhất (TB)"]
out = d[cols].sort_values(["dataset", "setting", "metric"])
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(ROOT / "results/tables/main_table.csv", index=False)
print("\nSố ô FedKT cao nhất (trung bình):", int(out["FedKT cao nhất (TB)"].sum()), "/", len(out))