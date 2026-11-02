"""Giao diện chung cho mọi phương pháp huấn luyện liên kết.

Vòng lặp chung (federated.py, algorithms.py) chỉ gọi các hook dưới đây. Mỗi phương pháp
là một file riêng trong src/methods/ và chỉ ghi đè hook mà nó cần.

Hook:
  setup(global_model, n_clients, cfg)          gọi 1 lần đầu mỗi lần chạy (khởi tạo trạng thái riêng)
  init_local(k, global_model, cfg)             tạo model cục bộ cho client k ở đầu round
  begin_local(model, teacher, cfg)             chuẩn bị trước khi huấn luyện cục bộ, trả về ctx
  local_loss(ctx, model, xb, yb, out, cfg)     loss của một mini-batch
"""
import copy

import torch.nn.functional as F


def soft_prob(logits, T=1.0):
    return F.softmax(logits / T, dim=1)


class Method:
    name = "base"
    federated = True      # False: không có server/aggregation (Client k)

    def setup(self, global_model, n_clients, cfg):
        pass

    def init_local(self, k, global_model, cfg):
        # mặc định: local model bắt đầu từ bản sao toàn bộ global model
        return copy.deepcopy(global_model)

    def begin_local(self, model, teacher, cfg):
        return None

    def local_loss(self, ctx, model, xb, yb, out, cfg):
        return F.cross_entropy(out, yb)
