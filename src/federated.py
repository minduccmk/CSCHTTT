"""Vòng lặp mô phỏng federated learning (1 server, K client).

File dùng chung: KHÔNG chứa logic riêng của phương pháp nào. Logic đó nằm trong src/methods/.
"""
import copy

import numpy as np
import torch

from algorithms import local_train, accuracy
from methods import get_method
from models import MLP


def make_selections(K, rounds, participation, seed):
    """Dãy client được chọn mỗi round; dùng chung cho mọi phương pháp (so sánh công bằng)."""
    rng = np.random.RandomState(seed + 12345)
    m = max(1, int(round(participation * K)))
    return [rng.choice(K, m, replace=False).tolist() for _ in range(rounds)]


def _avg_state(states, weights):
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states))
    return out


def run_federated(method, clients, in_dim, cfg, selections):
    """method in {fedavg, fedprox, fedcodl, fedkt}. Trả về dict lịch sử theo round."""
    m = get_method(method)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    global_model = MLP(in_dim, cfg.hidden, cfg.n_classes)
    K = len(clients)
    m.setup(global_model, K, cfg)

    hist = {"train_loss": [], "local_acc": [], "global_acc": []}
    for rnd, sel in enumerate(selections):
        states, sizes, losses, laccs = [], [], [], []
        # teacher cố định trong round = global model đầu round
        teacher = copy.deepcopy(global_model)
        for k in sel:
            local = m.init_local(k, global_model, cfg)
            loss = local_train(local, clients[k], m, cfg, cfg.local_epochs, global_model=teacher)
            losses.append(loss)
            laccs.append(accuracy(local, clients[k]["Xte"], clients[k]["yte"], cfg.device))
            states.append({kk: v.detach().cpu().clone() for kk, v in local.state_dict().items()})
            sizes.append(len(clients[k]["ytr"]))
        global_model.load_state_dict(_avg_state(states, sizes))   # Eq. (4)

        g_acc = float(np.mean([accuracy(global_model, c["Xte"], c["yte"], cfg.device) for c in clients]))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["local_acc"].append(float(np.mean(laccs)))
        hist["global_acc"].append(g_acc)
    return hist


def run_client_k(clients, in_dim, cfg):
    """Baseline 'Client k': mỗi client tự học trên dữ liệu riêng, không cộng tác."""
    m = get_method("client_k")
    accs = []
    for k, c in enumerate(clients):
        torch.manual_seed(cfg.seed + k)
        model = MLP(in_dim, cfg.hidden, cfg.n_classes)
        local_train(model, c, m, cfg, cfg.client_k_epochs)
        accs.append(accuracy(model, c["Xte"], c["yte"], cfg.device))
    return {"client_acc": float(np.mean(accs))}
