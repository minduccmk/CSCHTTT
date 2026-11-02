"""FedAvg (McMahan et al., 2017): local update chỉ dùng cross-entropy; server lấy trung bình trọng số."""
from .base import Method


class FedAvg(Method):
    name = "fedavg"
