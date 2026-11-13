"""Đọc kết quả quét lambda: FedKT(lam) so với FedAvg theo từng seed."""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import _path  # noqa: F401
from _path import ROOT
from compare_paper import PAPER

R = defaultdict(dict)
for f in glob.glob(str(ROOT / "results/calib/*.json")):
    ds, part, b, m, v, s = Path(f).stem.split("_")
    if b != "b32" or ds not in ("australian", "loan"):
        continue
    h = json.load(open(f))
    R[(ds, part, m, v)][int(s[1:])] = {"T3": np.mean(h["local_acc"]), "T4": np.mean(h["global_acc"])}
rows = []
for (ds, part, m, v), d in R.items():
    if m != "fedkt" or "@" not in v:
        continue
    base, lam = v.split("@")
    av = R[(ds, part, "fedavg", "-")]
    seeds = sorted(set(d) & set(av))
    for t in ("T3", "T4"):
        k = np.array([d[s][t] for s in seeds]); a = np.array([av[s][t] for s in seeds])
        st = "iid" if part == "iid" else "noniid"
        rows.append(dict(variant=base, lam=float(lam), dataset=ds, setting=st, table=t, fedkt=k.mean(), fedavg=a.mean(),
                         diff=(k - a).mean(), wins=f"{int((k > a).sum())}/{len(seeds)}",
                         paper_fedkt=PAPER[t][(ds, st)][3] if abs(float(lam) - 0.3) < 1e-9 else np.nan))
df = pd.DataFrame(rows).sort_values(["variant", "dataset", "setting", "table", "lam"])
df.round(2).to_csv(ROOT / "results/tables/lambda_sweep.csv", index=False)
pd.set_option("display.width", 200)
for v, g in df.groupby("variant"):
    print(f"\n##### variant {v}")
    print(g.pivot_table(index=["dataset", "setting", "table"], columns="lam", values="fedkt").round(2).to_string())
    print("-- FedAvg:"); print(g.drop_duplicates(["dataset", "setting", "table"]).set_index(["dataset", "setting", "table"])["fedavg"].round(2).to_string())
    print("-- số seed FedKT > FedAvg:"); print(g.pivot_table(index=["dataset", "setting", "table"], columns="lam", values="wins", aggfunc="first").to_string())
