"""Vòng lặp mô phỏng federated learning (1 server, K client)."""
import copy

import numpy as np
import torch

from algorithms import local_train, accuracy
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
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    global_model = MLP(in_dim, cfg.hidden, cfg.n_classes)
    K = len(clients)
    # FedKT: mỗi client giữ model cục bộ riêng (lớp FC cuối được giữ lại giữa các round)
    local_models = {}
    if method == "fedkt":
        local_models = {k: copy.deepcopy(global_model) for k in range(K)}

    hist = {"train_loss": [], "local_acc": [], "global_acc": []}
    for rnd, sel in enumerate(selections):
        states, sizes, losses, laccs = [], [], [], []
        # teacher cố định trong round = global model đầu round
        teacher = copy.deepcopy(global_model)
        for k in sel:
            if method == "fedkt":
                local = local_models[k]
                # copy các lớp đầu (generic knowledge), giữ nguyên FC cuối của local
                local.features.load_state_dict(global_model.features.state_dict())
                if cfg.fc_mode == "global":
                    local.classifier.load_state_dict(global_model.classifier.state_dict())
                elif cfg.fc_mode == "reinit":
                    local.classifier.reset_parameters()
            else:
                local = copy.deepcopy(global_model)
            loss = local_train(local, clients[k], method, cfg, cfg.local_epochs, global_model=teacher)
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
    accs = []
    for k, c in enumerate(clients):
        torch.manual_seed(cfg.seed + k)
        model = MLP(in_dim, cfg.hidden, cfg.n_classes)
        local_train(model, c, "client_k", cfg, cfg.client_k_epochs)
        accs.append(accuracy(model, c["Xte"], c["yte"], cfg.device))
    return {"client_acc": float(np.mean(accs))}
