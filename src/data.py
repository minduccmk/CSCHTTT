"""Đọc, tiền xử lý 4 dataset tín dụng và chia dữ liệu cho các client."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import DATASET_INFO


# từ khoá nhận diện file (đã chuẩn hoá: chữ thường, bỏ _, -, khoảng trắng)
_KEYWORDS = {
    "australian": [["australian"]],
    "loan": [["loandata"], ["loan"]],
    "taiwan": [["defaultofcreditcard"], ["ucicreditcard"], ["taiwan"]],
    "gmsc": [["cstraining"], ["gmsc"], ["givemesomecredit"]],
}
# đuôi file hợp lệ cho từng dataset (theo thứ tự ưu tiên)
_EXTS = {
    "australian": [".dat", ".csv", ".txt"],
    "loan": [".csv"],
    "taiwan": [".xls", ".xlsx", ".csv"],
    "gmsc": [".csv"],
}
_EXCLUDE = ("dictionary", "description", "readme", "sample", "test")


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_file(data_dir, name):
    d = Path(data_dir)
    # 1) tên chuẩn
    for f in DATASET_INFO[name]["files"]:
        if (d / f).exists():
            return d / f
    # 2) khớp theo từ khoá, tìm cả trong thư mục con
    everything = [p for p in d.rglob("*") if p.is_file()] if d.exists() else []
    cands = [p for p in everything
             if p.suffix.lower() in _EXTS[name] and not any(x in _norm(p.stem) for x in _EXCLUDE)]
    for ext in _EXTS[name]:
        for kws in _KEYWORDS[name]:
            for p in sorted(cands):
                if p.suffix.lower() == ext and all(k in _norm(p.stem) for k in kws):
                    return p
    found = [str(p.relative_to(d)) for p in everything] if d.exists() else "(thư mục Dataset không tồn tại)"
    print(f"\n[DEBUG] Dataset: {name}")
    print(f"[DEBUG] Directory: {d.resolve()}")
    print(f"[DEBUG] Extensions: {_EXTS[name]}")
    print(f"[DEBUG] Keywords: {_KEYWORDS[name]}")
    print("[DEBUG] Files found:")
    for p in everything:
        print(
            f"  File: {p.name!r}"
            f" | Extension: {p.suffix.lower()!r}"
            f" | Normalized name: {_norm(p.stem)!r}"
        )
    raise FileNotFoundError(
        f"Không thấy file cho '{name}' trong {d.resolve()}.\n"
        f"  Tên mong đợi, ví dụ: {DATASET_INFO[name]['files'][0]}\n"
        f"  Các file đang có trong thư mục: {found}"
        )


def _preprocess(df_x: pd.DataFrame, cat_cols):
    """Continuous -> MinMax [0,1]; categorical -> one-hot."""
    cat_cols = [c for c in cat_cols if c in df_x.columns]
    cont_cols = [c for c in df_x.columns if c not in cat_cols]
    parts = []
    if cont_cols:
        cont = df_x[cont_cols].astype(float)
        cont = cont.fillna(cont.median())
        rng = (cont.max() - cont.min()).replace(0, 1)
        parts.append(((cont - cont.min()) / rng).astype(np.float32))
    if cat_cols:
        parts.append(pd.get_dummies(df_x[cat_cols].astype(str)).astype(np.float32))
    return pd.concat(parts, axis=1).values.astype(np.float32)


def load_dataset(name, data_dir="Dataset"):
    path = _find_file(data_dir, name)
    if name == "australian":
        df = pd.read_csv(path, sep=r"\s+", header=None)
        y = df.iloc[:, -1].values
        x = df.iloc[:, :-1]
        x.columns = range(x.shape[1])
        # A1,A4,A5,A6,A8,A9,A11,A12 là categorical (8 cột) -> đúng 42 chiều như bài báo
        X = _preprocess(x, cat_cols=[0, 3, 4, 5, 7, 8, 10, 11])
    elif name == "taiwan":
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, header=1)
        df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() == "ID"])
        y = df.iloc[:, -1].values
        x = df.iloc[:, :-1]
        cats = [c for c in x.columns
                if str(c).upper() in {"SEX", "EDUCATION", "MARRIAGE"} or str(c).upper().startswith("PAY_")
                and not str(c).upper().startswith("PAY_AMT")]
        X = _preprocess(x, cat_cols=cats)
    elif name == "gmsc":
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
        y = df["SeriousDlqin2yrs"].values
        x = df.drop(columns=["SeriousDlqin2yrs"])
        X = _preprocess(x, cat_cols=[])
    elif name == "loan":
        df = pd.read_csv(path, sep=None, engine="python")   # file gốc dùng dấu ';'
        y = df.iloc[:, -1].values                           # cột BAD
        x = df.iloc[:, :-1]
        cats = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
        X = _preprocess(x, cat_cols=cats)
    else:
        raise ValueError(name)
    # chuẩn hoá nhãn về {0,1} (1 = default)
    classes = np.unique(y)
    if set(classes) == {1, 2}:       # Statlog: 1/2
        y = (y == 2).astype(np.int64)
    else:
        y = (y == classes.max()).astype(np.int64)
    return X, y


def partition(X, y, K, setting, seed, train_ratio=0.8, dirichlet_alpha=0.5):
    """Trả về list K client, mỗi client là dict (Xtr, ytr, Xte, yte).

    setting:
      iid       - stratified: mọi client có đủ nhãn, giữ đúng tỷ lệ imbalance
      noniid    - chia ngẫu nhiên thẳng, không stratify (diễn giải theo bài báo)
      dirichlet - label skew mạnh theo Dirichlet(alpha) (tuỳ chọn thêm)
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    if setting == "iid":
        idx_per_client = [[] for _ in range(K)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            for i, part in enumerate(np.array_split(idx, K)):
                idx_per_client[i].extend(part.tolist())
    elif setting in ("noniid", "random"):
        idx = rng.permutation(n)
        idx_per_client = [p.tolist() for p in np.array_split(idx, K)]
    elif setting == "sorted":
        # sắp theo nhãn rồi cắt K phần bằng nhau -> label skew rất mạnh
        idx = rng.permutation(n)
        idx = idx[np.argsort(y[idx], kind="stable")]
        idx_per_client = [p.tolist() for p in np.array_split(idx, K)]
    elif setting == "dirichlet":
        idx_per_client = [[] for _ in range(K)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(np.repeat(dirichlet_alpha, K))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                idx_per_client[i].extend(part.tolist())
    else:
        raise ValueError(setting)

    clients = []
    for idx in idx_per_client:
        idx = np.array(idx)
        if len(idx) < 5:                       # tránh client rỗng
            idx = rng.choice(n, 5, replace=False)
        yi = y[idx]
        try:
            tr, te = train_test_split(idx, train_size=train_ratio, stratify=yi,
                                      random_state=seed)
        except ValueError:                     # thiếu mẫu một lớp -> không stratify
            tr, te = train_test_split(idx, train_size=train_ratio, random_state=seed)
        clients.append({
            "Xtr": torch.tensor(X[tr]), "ytr": torch.tensor(y[tr]),
            "Xte": torch.tensor(X[te]), "yte": torch.tensor(y[te]),
        })
    return clients