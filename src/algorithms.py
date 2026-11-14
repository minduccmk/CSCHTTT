"""Huấn luyện cục bộ dùng chung cho mọi phương pháp.

File này KHÔNG chứa loss của từng phương pháp; loss nằm trong src/methods/<tên>.py.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch

from methods import get_method


def local_train(model, client, method, cfg, epochs, global_model=None):
    """Huấn luyện `model` tại chỗ; trả về loss trung bình epoch cuối.

    method: tên phương pháp (str) hoặc một instance của methods.Method.
    global_model: model global lúc bắt đầu round (teacher / điểm neo của FedProx).
    """
    m = get_method(method) if isinstance(method, str) else method
    dev = cfg.device
    model.to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    X, y = client["Xtr"].to(dev), client["ytr"].to(dev)
    n = len(y)

    ctx = m.begin_local(model, global_model, cfg)

    last_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        total, cnt = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            xb, yb = X[b], y[b]
            out = model(xb)
            loss = m.local_loss(ctx, model, xb, yb, out, cfg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            cnt += len(b)
        last_loss = total / max(cnt, 1)
    return last_loss


@torch.no_grad()
def accuracy(model, X, y, device="cpu"):
    model.to(device).eval()
    pred = model(X.to(device)).argmax(1).cpu()
    return (pred == y).float().mean().item() * 100.0


@torch.no_grad()
def evaluate(model, X, y, device="cpu", threshold=0.5, pos_label=1):
    """Metric đầy đủ trên một client. Lớp dương = pos_label (default).

    Quy ước (nhớ ghi vào báo cáo):
    - recall / AUC / KS = nan nếu tập test không có đủ 2 lớp (client lệch nhãn nặng)
    - precision = 0 nếu mô hình không dự đoán dương nào
    """
    model.to(device).eval()
    p = torch.softmax(model(X.to(device)), dim=1)[:, pos_label].cpu().numpy()
    yt = (y.cpu().numpy() == pos_label).astype(int)
    pred = (p > threshold).astype(int)

    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(rec):
        f1 = np.nan
    else:
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    if len(np.unique(yt)) < 2:
        auc, ks = np.nan, np.nan
    else:
        auc = roc_auc_score(yt, p) * 100
        fpr, tpr, _ = roc_curve(yt, p)
        ks = float(np.max(tpr - fpr)) * 100

    return {
        "precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
        "auc": auc, "ks": ks}


def mean_metrics(ds):
    """Trung bình theo client, bỏ qua nan; kèm số client hợp lệ cho mỗi metric."""
    out = {}
    for k in ds[0]:
        v = np.array([d[k] for d in ds], dtype=float)
        ok = ~np.isnan(v)
        out[k] = float(v[ok].mean()) if ok.any(
        ) else None   # None -> JSON null
        out[k + "_n"] = int(ok.sum())
    return out
