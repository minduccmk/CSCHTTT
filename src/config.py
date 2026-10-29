"""Cấu hình thực nghiệm theo bài báo FedKT (credit scoring)."""
from dataclasses import dataclass

# Số client K theo Table 1 của bài báo
DATASET_INFO = {
    "australian": {"K": 10, "files": ["australian.dat", "australian.csv", "australian.txt"]},
    "loan": {"K": 20, "files": ["loan.csv", "loan_data.csv", "Loan.csv"]},
    "taiwan": {"K": 50, "files": ["default of credit card clients.xls",
                                   "default of credit card clients.xlsx",
                                   "taiwan.csv", "taiwan.xls", "taiwan.xlsx",
                                   "UCI_Credit_Card.csv"]},
    "gmsc": {"K": 150, "files": ["cs-training.csv", "gmsc.csv"]},
}

METHODS = ["client_k", "fedavg", "fedprox", "fedcodl", "fedkt"]


@dataclass
class Config:
    # --- Federated setup (Section 4.1.2) ---
    rounds: int = 50
    participation: float = 0.2      # 20% client mỗi round
    local_epochs: int = 10          # E = 10
    lr: float = 0.5
    momentum: float = 0.5
    hidden: int = 100               # 2 hidden layer x 100 unit
    n_classes: int = 2
    train_ratio: float = 0.8        # 80% train / 20% test trên mỗi client

    # --- Không nêu trong bài báo (giả định) ---
    batch_size: int = 32
    client_k_epochs: int = 100      # tổng epoch cho baseline "Client k"
    dirichlet_alpha: float = 0.5    # chỉ dùng cho setting "dirichlet"

    # --- FedKT ---
    fc_mode: str = "keep"           # lớp FC cuối của client: keep | global | reinit
    lam: float = 0.3               # lambda (trọng số hard loss)
    temperature: float = 1.0        # T = 1 (2 lớp nhãn)

    # --- Baselines ---
    prox_mu: float = 0.01           # FedProx
    codl_beta: float = 1.0          # FedCodl: hệ số distillation (giả định)

    seed: int = 0
    device: str = "cpu"
