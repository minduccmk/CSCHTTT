"""FedCodl (Ni et al., 2022): CE + beta * MSE(softmax_global, softmax_local) (codistillation).

Tham số riêng: cfg.codl_beta (bài không nêu; mặc định 1.0), cfg.temperature.
"""
import torch
import torch.nn.functional as F

from .base import Method, soft_prob


class FedCodl(Method):
    name = "fedcodl"

    def begin_local(self, model, teacher, cfg):
        return teacher.to(cfg.device).eval()

    def local_loss(self, ctx, model, xb, yb, out, cfg):
        hard = F.cross_entropy(out, yb)
        with torch.no_grad():
            q_g = soft_prob(ctx(xb), cfg.temperature)
        q_l = soft_prob(out, cfg.temperature)
        soft = F.mse_loss(q_l, q_g)
        return hard + cfg.codl_beta * soft
