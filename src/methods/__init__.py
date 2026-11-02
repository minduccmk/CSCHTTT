"""Registry các phương pháp. Thêm phương pháp mới = thêm 1 file + 1 dòng ở đây."""
from .base import Method
from .client_k import ClientK
from .fedavg import FedAvg
from .fedcodl import FedCodl
from .fedkt import FedKT
from .fedprox import FedProx

REGISTRY = {cls.name: cls for cls in (ClientK, FedAvg, FedProx, FedCodl, FedKT)}


def get_method(name):
    """Trả về một instance MỚI (FedKT giữ trạng thái riêng cho từng lần chạy)."""
    if name not in REGISTRY:
        raise ValueError(f"Phương pháp không có: {name}. Có: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["Method", "REGISTRY", "get_method"]
