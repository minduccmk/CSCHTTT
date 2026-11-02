"""FedProx (Li et al., 2020): CE + (mu/2) * ||w - w_global||^2.

Tham số riêng: cfg.prox_mu.
"""
import torch.nn.functional as F

from .base import Method


class FedProx(Method):
    name = "fedprox"

    def begin_local(self, model, teacher, cfg):
        # tham số của global model lúc bắt đầu round (cố định trong suốt local update)
        return [p.detach().clone() for p in teacher.to(cfg.device).parameters()]

    def local_loss(self, ctx, model, xb, yb, out, cfg):
        hard = F.cross_entropy(out, yb)
        prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), ctx))
        return hard + 0.5 * cfg.prox_mu * prox
