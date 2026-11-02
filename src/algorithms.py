"""Huấn luyện cục bộ dùng chung cho mọi phương pháp.

File này KHÔNG chứa loss của từng phương pháp; loss nằm trong src/methods/<tên>.py.
"""
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
