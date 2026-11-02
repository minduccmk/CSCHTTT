"""Client k: mỗi client tự học trên dữ liệu riêng, không có server (baseline không cộng tác).

Loss giống FedAvg. Tham số riêng: cfg.client_k_epochs (bài không nêu).
"""
from .base import Method


class ClientK(Method):
    name = "client_k"
    federated = False
