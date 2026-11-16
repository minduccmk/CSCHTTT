"""Chạy lại TẤT CẢ: 4 dataset x 2 cách chia (IID, Non-IID) x 6 mô hình (Client k, FedAvg, FedProx,
FedCodl beta 0.1 & 1.0, FedKT) x các seed. Có cache: job nào đã có file trong results/calib/ sẽ bị bỏ qua.

  python experiments/run_all.py                       # mặc định: 4 dataset, seed 0-2
  python experiments/run_all.py --datasets australian loan --seeds 0 1 2 3 4 --procs 4
  python experiments/run_all.py --fresh               # xoá cache của các dataset được chọn rồi chạy lại từ đầu
Sau đó: partition_report.py -> main_table.py -> plot_extra.py
"""
import argparse
import multiprocessing as mp

import _path  # noqa: F401
from calibrate import OUT, run_job
from clientk_grid import run as run_clientk

METHODS = [("fedavg", "-"), ("fedprox", "-"), ("fedcodl", "0.1"), ("fedcodl", "1.0"), ("fedkt", "keep")]


def work(job):
    return run_clientk(job[1:]) if job[0] == "clientk" else run_job(job[1:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["australian", "loan", "taiwan", "gmscfe"])
    ap.add_argument("--parts", nargs="+", default=["iid", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        for ds in a.datasets:
            for f in OUT.glob(f"{ds}_*.json"):
                f.unlink()
    jobs = []
    for s in a.seeds:
        for d in a.datasets:
            for p in a.parts:
                jobs += [("m", d, p, 32, m, v, s) for m, v in METHODS]
                jobs.append(("clientk", d, p, s))
    jobs.sort(key=lambda j: j[1] in ("gmscfe",))      # dataset nhỏ chạy trước, GMSC chạy cuối
    print(len(jobs), "jobs", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs), 1):
            print(f"{i}/{len(jobs)} {r}", flush=True)