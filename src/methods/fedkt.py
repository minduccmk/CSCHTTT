"""FedKT (phương pháp của bài): federated knowledge transfer.

1) Fine-tuning: local model bắt đầu từ các lớp đầu của global model (trừ lớp FC cuối).
2) Distillation: loss = lam * CE + (1 - lam) * MSE(softmax_global, softmax_local)   (Eq. 5)

Mỗi client giữ model cục bộ riêng giữa các round (lớp FC cuối nằm ở đây).
Tham số riêng (người A): cfg.lam (bài: 0.3), cfg.kt_temperature (bài: 1), cfg.fc_mode,
cfg.kt_reduction, cfg.kt_copy (hai tham số cuối: bài không nêu, mặc định giữ nguyên cách cài ban đầu).
  kt_copy      = h2 : copy cả 2 lớp ẩn của global (mặc định)
                 h1 : chỉ copy lớp ẩn đầu, lớp ẩn thứ hai và FC giữ của client
  kt_reduction = mean : F.mse_loss mặc định (trung bình trên mẫu x lớp)
                 sum  : tổng bình phương theo lớp, trung bình theo mẫu (gấp đôi "mean" với 2 lớp)
  fc_mode = keep   : giữ FC cuối của chính client từ round trước
            global : copy cả FC cuối từ global model
            reinit : khởi tạo lại FC cuối mỗi round
"""
import copy

import torch
import torch.nn.functional as F

from config import validate

from .base import Method, soft_prob


class FedKT(Method):
    name = "fedkt"

    def setup(self, global_model, n_clients, cfg):
        validate(cfg)          # bắt fc_mode gõ sai, lam ngoài [0,1]
        self.local_models = {k: copy.deepcopy(global_model) for k in range(n_clients)}

    def init_local(self, k, global_model, cfg):
        local = self.local_models[k]
        # copy các lớp đầu (generic knowledge), giữ nguyên FC cuối của local
        if cfg.kt_copy == "h1":
            local.features[0].load_state_dict(global_model.features[0].state_dict())
        else:
            local.features.load_state_dict(global_model.features.state_dict())
        if cfg.fc_mode == "global":
            local.classifier.load_state_dict(global_model.classifier.state_dict())
        elif cfg.fc_mode == "reinit":
            local.classifier.reset_parameters()
        return local

    def begin_local(self, model, teacher, cfg):
        return teacher.to(cfg.device).eval()

    def local_loss(self, ctx, model, xb, yb, out, cfg):
        hard = F.cross_entropy(out, yb)
        with torch.no_grad():
            q_g = soft_prob(ctx(xb), cfg.kt_temperature)
        q_l = soft_prob(out, cfg.kt_temperature)
        if cfg.kt_reduction == "sum":
            soft = F.mse_loss(q_l, q_g, reduction="sum") / q_l.shape[0]
        else:
            soft = F.mse_loss(q_l, q_g)
        return cfg.lam * hard + (1 - cfg.lam) * soft
