"""Hàm huấn luyện cục bộ cho từng phương pháp.

method:
  client_k / fedavg : CE
  fedprox           : CE + (mu/2)||w - w_global||^2
  fedcodl           : CE + beta * MSE(softmax_global, softmax_local)   (codistillation)
  fedkt             : lam * CE + (1-lam) * MSE(softmax_global, softmax_local)   (Eq. 5)
"""
import torch
import torch.nn.functional as F


def soft_prob(logits, T=1.0):
    return F.softmax(logits / T, dim=1)


def local_train(model, client, method, cfg, epochs, global_model=None):
    """Huấn luyện `model` tại chỗ; trả về loss trung bình epoch cuối."""
    dev = cfg.device
    model.to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    X, y = client["Xtr"].to(dev), client["ytr"].to(dev)
    n = len(y)

    teacher = None
    g_params = None
    if method in ("fedkt", "fedcodl"):
        teacher = global_model.to(dev).eval()
    if method == "fedprox":
        g_params = [p.detach().clone() for p in global_model.to(dev).parameters()]

    last_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        total, cnt = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            xb, yb = X[b], y[b]
            out = model(xb)
            hard = F.cross_entropy(out, yb)

            if method in ("client_k", "fedavg"):
                loss = hard
            elif method == "fedprox":
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), g_params))
                loss = hard + 0.5 * cfg.prox_mu * prox
            else:
                with torch.no_grad():
                    q_g = soft_prob(teacher(xb), cfg.temperature)
                q_l = soft_prob(out, cfg.temperature)
                soft = F.mse_loss(q_l, q_g)
                if method == "fedkt":
                    loss = cfg.lam * hard + (1 - cfg.lam) * soft
                else:  # fedcodl
                    loss = hard + cfg.codl_beta * soft

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
