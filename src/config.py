"""Cấu hình thực nghiệm FedKT (credit scoring).

QUY ƯỚC CHO HAI NGƯỜI (A và B) — đọc trước khi sửa file này:

  1. FROZEN_SHARED + PROTOCOL  : THAM SỐ DÙNG CHUNG, đã KHOÁ. Không ai tự sửa.
                                 Muốn đổi phải mở PR riêng, cả A và B cùng duyệt (xem CONTRIBUTING.md).
  2. Tham số riêng của A (FedAvg, FedKT)               : lam, kt_temperature, fc_mode, kt_reduction, kt_copy
  3. Tham số riêng của B (Client k, FedProx, FedCodl)  : prox_mu, codl_beta, codl_temperature, client_k_epochs
  4. Tham số theo từng lần chạy (tự do)                : seed, device, dirichlet_alpha

Mỗi file kết quả lưu kèm cấu hình và `shared_hash`. Nếu hash khác FROZEN_HASH thì file đó
không được dùng cho bảng chính (xem experiments/check_shared.py và experiments/report.py).

Chạy `python src/config.py` để in toàn bộ bảng tham số.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

# --------------------------------------------------------------------------------------
# Dữ liệu
# --------------------------------------------------------------------------------------
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

# Thứ tự phương pháp dùng trong bảng/hình
METHODS = ["client_k", "fedavg", "fedprox", "fedcodl", "fedkt"]

# Ai phụ trách phương pháp nào (khớp CODEOWNERS)
OWNER = {"fedavg": "A", "fedkt": "A", "client_k": "B", "fedprox": "B", "fedcodl": "B"}


def methods_of(owner):
    """owner in {'A','B','all'} -> danh sách phương pháp theo thứ tự METHODS."""
    if owner == "all":
        return list(METHODS)
    return [m for m in METHODS if OWNER[m] == owner]


# --------------------------------------------------------------------------------------
# 1) THAM SỐ DÙNG CHUNG (KHOÁ)
#    "paper" = bài báo ghi rõ (mục 4.1.2).  "assumed" = bài không nêu, hai người đã thống nhất chọn.
# --------------------------------------------------------------------------------------
FROZEN_SHARED = {
    "rounds": 50,           # paper   : 50 communication rounds
    "participation": 0.2,   # paper   : 20% client mỗi round, danh sách chọn cố định giữa các phương pháp
    "local_epochs": 10,     # paper   : E = 10
    "lr": 0.5,              # paper   : SGD lr = 0.5
    "momentum": 0.5,        # paper   : momentum = 0.5
    "hidden": 100,          # paper   : MLP 2 lớp ẩn x 100 đơn vị + 1 lớp FC ra
    "n_classes": 2,         # paper   : default / non-default
    "train_ratio": 0.8,     # paper   : 80% train / 20% test trên từng client
    "batch_size": 32,       # assumed : bài không nêu; thử 16/32/64, 32 gần bài nhất
}

# Quy trình thực nghiệm dùng chung (không nằm trong Config vì là lựa chọn của người chạy)
PROTOCOL = {
    # tập dataset của bảng chính; GMSC chưa hiệu chỉnh nên không nằm trong đây
    "main_datasets": ["australian", "loan", "taiwan"],
    # các seed của kết quả cuối cùng; mỗi seed ứng với 1 cách chia dữ liệu + 1 dãy client được chọn
    "seeds": [0, 1, 2, 3, 4],
    # assumed: tên setting của bài -> cách chia trong src/data.py
    #   iid    = stratified: mọi client có cùng tỷ lệ nhãn
    #   random = xáo trộn rồi chia K phần bằng nhau: tỷ lệ nhãn mỗi client lệch nhẹ do imbalance
    "partition_of": {"iid": "iid", "noniid": "random"},
    # assumed: tiền xử lý (cài trong src/data.py): biến liên tục -> MinMax [0,1], biến phân loại -> one-hot
    "preprocess": "minmax+onehot",
    # paper: accuracy trung bình trên 50 round; local = model cục bộ của các client được chọn trên test của chính nó;
    #        global = model toàn cục trên test của mọi client
    "metrics": ["train_loss", "local_acc", "global_acc"],
}


def partition_of(setting):
    """Tên setting ('iid'/'noniid') -> cách chia dữ liệu. Tên khác ('random','sorted','dirichlet') giữ nguyên."""
    return PROTOCOL["partition_of"].get(setting, setting)


# --------------------------------------------------------------------------------------
# Config: nhóm trường theo chủ sở hữu
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # ---- 1) CHUNG, KHOÁ (giá trị mặc định lấy từ FROZEN_SHARED; đừng đổi tại đây) ----
    rounds: int = FROZEN_SHARED["rounds"]
    participation: float = FROZEN_SHARED["participation"]
    local_epochs: int = FROZEN_SHARED["local_epochs"]
    lr: float = FROZEN_SHARED["lr"]
    momentum: float = FROZEN_SHARED["momentum"]
    hidden: int = FROZEN_SHARED["hidden"]
    n_classes: int = FROZEN_SHARED["n_classes"]
    train_ratio: float = FROZEN_SHARED["train_ratio"]
    batch_size: int = FROZEN_SHARED["batch_size"]

    # ---- 2) RIÊNG của A (FedAvg, FedKT) ----
    lam: float = 0.3                # paper   : λ = 0.3 (trọng số loss cứng của FedKT)
    kt_temperature: float = 1.0     # paper   : T = 1 (chỉ 2 nhãn)
    fc_mode: str = "keep"           # assumed : lớp FC cuối của client: keep | global | reinit
    kt_reduction: str = "mean"      # assumed : MSE trung bình trên (mẫu x lớp) = "mean"; tổng theo lớp, trung bình theo mẫu = "sum"
    kt_copy: str = "h2"             # assumed : copy "h2" = cả 2 lớp ẩn (mặc định) hay "h1" = chỉ lớp ẩn đầu

    # ---- 3) RIÊNG của B (Client k, FedProx, FedCodl) ----
    prox_mu: float = 0.01           # assumed : bài không nêu μ của FedProx
    codl_beta: float = 1.0          # assumed : bài không nêu hệ số distillation của FedCodl
    codl_temperature: float = 1.0   # assumed : bài không nêu
    client_k_epochs: int = 100      # assumed : bài không nêu số epoch của Client k

    # ---- 4) THEO TỪNG LẦN CHẠY (tự do) ----
    seed: int = 0
    device: str = "cpu"
    dirichlet_alpha: float = 0.5    # chỉ dùng cho setting 'dirichlet' (phân tích độ nhạy, không phải bảng chính)


FC_MODES = ("keep", "global", "reinit")
KT_REDUCTIONS = ("mean", "sum")
KT_COPIES = ("h2", "h1")


def validate(cfg):
    """Bắt các giá trị sai chính tả/vô lý sớm, thay vì chạy âm thầm với giá trị mặc định."""
    if cfg.fc_mode not in FC_MODES:
        raise ValueError(f"fc_mode='{cfg.fc_mode}' không hợp lệ. Chọn một trong {FC_MODES}")
    if cfg.kt_reduction not in KT_REDUCTIONS:
        raise ValueError(f"kt_reduction='{cfg.kt_reduction}' không hợp lệ. Chọn một trong {KT_REDUCTIONS}")
    if cfg.kt_copy not in KT_COPIES:
        raise ValueError(f"kt_copy='{cfg.kt_copy}' không hợp lệ. Chọn một trong {KT_COPIES}")
    if not 0.0 <= cfg.lam <= 1.0:
        raise ValueError(f"lam={cfg.lam} phải nằm trong [0, 1]")
    if not 0.0 < cfg.participation <= 1.0:
        raise ValueError(f"participation={cfg.participation} phải nằm trong (0, 1]")
    for name in ("kt_temperature", "codl_temperature", "prox_mu", "codl_beta"):
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} không được âm")
    return cfg


# --------------------------------------------------------------------------------------
# Kiểm tra tham số chung (dùng khi chạy và khi gộp kết quả)
# --------------------------------------------------------------------------------------
def shared_params(cfg):
    return {k: getattr(cfg, k) for k in FROZEN_SHARED}


def _hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


FROZEN_HASH = _hash(FROZEN_SHARED)


def shared_hash(cfg):
    return _hash(shared_params(cfg))


def shared_diff(cfg):
    """Các tham số chung của cfg khác giá trị đã khoá: {tên: (giá trị cfg, giá trị khoá)}."""
    return {k: (getattr(cfg, k), v) for k, v in FROZEN_SHARED.items() if getattr(cfg, k) != v}


def is_frozen(cfg):
    return shared_hash(cfg) == FROZEN_HASH


def describe():
    print(f"FROZEN_HASH = {FROZEN_HASH}\n")
    print("== 1) THAM SỐ CHUNG (khoá) ==")
    for k, v in FROZEN_SHARED.items():
        print(f"  {k:14s} = {v}")
    print("\n== PROTOCOL chung ==")
    for k, v in PROTOCOL.items():
        print(f"  {k:14s} = {v}")
    print("\n== Tham số riêng ==")
    c = Config()
    for owner, names in (("A", ["lam", "kt_temperature", "fc_mode", "kt_reduction", "kt_copy"]),
                         ("B", ["prox_mu", "codl_beta", "codl_temperature", "client_k_epochs"])):
        for n in names:
            print(f"  [{owner}] {n:16s} = {getattr(c, n)}")
    print("\n== Phân công ==")
    for owner in ("A", "B"):
        print(f"  {owner}: {', '.join(methods_of(owner))}")


if __name__ == "__main__":
    describe()
    print("\nConfig mặc định có khoá không?", is_frozen(Config()), "| dict:", asdict(Config()))
