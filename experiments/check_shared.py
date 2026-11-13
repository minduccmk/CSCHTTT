"""Kiểm tra khi GỘP kết quả của người A và người B.

In số file theo thư mục, theo phương pháp (và chủ sở hữu) và theo trạng thái:
  frozen    : có shared_hash và bằng FROZEN_HASH
  legacy    : file cũ không ghi cấu hình (chấp nhận, nhưng nên chạy lại bằng bản mới khi có thể)
  nonfrozen : tham số chung khác giá trị đã khoá. Hợp lệ DUY NHẤT trong results/calib với batch size != 32 (độ nhạy)

Thoát với mã 1 nếu có file không hợp lệ trong bảng chính (results/raw, results/raw_lambda,
results/calib batch 32). Dùng:  python experiments/check_shared.py
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT
from config import FROZEN_HASH, FROZEN_SHARED, OWNER, shared_diff, Config


def method_of(path, r):
    if "method" in r:
        return r["method"]
    if "meta" in r:
        return r["meta"]["method"]
    return None            # file cũ chứa nhiều phương pháp


def main():
    bad = []
    for d in ("raw", "raw_lambda", "calib"):
        files = sorted(glob.glob(str(ROOT / "results" / d / "*.json")))
        if not files:
            continue
        cnt = Counter()
        for f in files:
            r = json.load(open(f))
            h = r.get("shared_hash")
            batch = r.get("meta", {}).get("batch", r.get("config", {}).get("batch_size", FROZEN_SHARED["batch_size"]))
            if h is None:
                status = "legacy"
            elif h == FROZEN_HASH:
                status = "frozen"
            else:
                status = "nonfrozen"
                if d == "calib" and batch != FROZEN_SHARED["batch_size"]:
                    status = "nonfrozen (độ nhạy batch, hợp lệ)"
                else:
                    diff = shared_diff(Config(**{k: v for k, v in r["config"].items() if k in Config.__dataclass_fields__}))
                    bad.append((Path(f).name, diff))
            m = method_of(f, r)
            who = OWNER.get(m, "?") if m else "cũ"
            cnt[(status, who)] += 1
        print(f"\nresults/{d}: {len(files)} file")
        for (status, who), n in sorted(cnt.items()):
            print(f"  {status:36s} người {who}: {n}")
    print(f"\nFROZEN_HASH = {FROZEN_HASH}  ({FROZEN_SHARED})")
    if bad:
        print(f"\nKHÔNG HỢP LỆ: {len(bad)} file có tham số chung khác giá trị đã khoá:")
        for name, diff in bad[:10]:
            print(f"  {name}: {diff}")
        sys.exit(1)
    print("OK: không có file nào lệch tham số chung trong bảng chính.")


if __name__ == "__main__":
    main()
