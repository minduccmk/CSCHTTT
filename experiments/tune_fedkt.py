"""Người A: chạy lưới biến thể FedKT (các lựa chọn bài không nêu rõ) và, khi cần, các baseline trên seed mới.

Biến thể = fc_mode x kt_reduction x kt_copy = 3 x 2 x 2 = 12. Tên: "<fc>" nếu là (mean, h2) mặc định,
còn lại "<fc>-<reduction>-<copy>". Dùng lại calibrate.run_job nên có cache (không chạy lại file đã có).

  python experiments/tune_fedkt.py --datasets australian loan --seeds 0 1 2 3 4
  python experiments/tune_fedkt.py --datasets australian loan --seeds 5 6 7 8 9 \
        --variants keep keep-sum-h2 --baselines
"""
import argparse
import itertools
import multiprocessing as mp

import _path  # noqa: F401
from calibrate import OUT, run_job

FC = ("keep", "global", "reinit")
RED = ("mean", "sum")
COPY = ("h2", "h1")


def variant_name(fc, red, cp):
    return fc if (red, cp) == ("mean", "h2") else f"{fc}-{red}-{cp}"


ALL_VARIANTS = [variant_name(*t) for t in itertools.product(FC, RED, COPY)]


def build(datasets, parts, seeds, variants, baselines, batch=32):
    jobs = []
    for s in seeds:
        for ds in datasets:
            for p in parts:
                for v in variants:
                    jobs.append((ds, p, batch, "fedkt", v, s))
                if baselines:
                    jobs += [(ds, p, batch, "fedavg", "-", s), (ds, p, batch, "fedprox", "-", s),
                             (ds, p, batch, "fedcodl", "1.0", s)]
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["australian", "loan"])
    ap.add_argument("--parts", nargs="+", default=["iid", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    ap.add_argument("--baselines", action="store_true", help="chạy thêm FedAvg, FedProx, FedCodl (beta=1.0)")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = build(a.datasets, a.parts, a.seeds, a.variants, a.baselines)
    print(f"{len(jobs)} jobs", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_job, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
