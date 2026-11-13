"""Sinh docs/ket_qua_so_sanh.md từ các bảng final_*.csv (cấu hình chính đã khoá). Chạy sau experiments/report.py."""
import pandas as pd

import _path  # noqa: F401
from _path import ROOT
from config import FROZEN_SHARED, PROTOCOL

T = ROOT / "results" / "tables"
M = ["fedavg", "fedprox", "fedcodl", "fedkt"]
NAME = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcodl": "FedCodl", "fedkt": "FedKT"}
DS = {"australian": "Australian", "loan": "Loan", "taiwan": "Taiwan"}
ST = {"iid": "IID", "noniid": "Non-IID"}
main = pd.read_csv(T / f"final_main_batch{FROZEN_SHARED['batch_size']}.csv")
fit = pd.read_csv(T / f"final_fit_batch{FROZEN_SHARED['batch_size']}.csv")
bat = pd.read_csv(T / "final_batch_effect.csv")
n_seeds = int(main.n_seeds.min())


def table(tab):
    d = main[main.table == tab]
    hdr = "| Dataset | Setting | " + " | ".join(f"{NAME[m]} (bài / của bạn)" for m in M) + " |\n|---|---|" + "---|" * 4 + "\n"
    return hdr + "".join(
        f"| {DS[r.dataset]} | {ST[r.setting]} | " + " | ".join(f"{r[m + '_paper']:.2f} / {r[m + '_ours']}" for m in M) + " |\n"
        for _, r in d.iterrows())


def gains():
    hdr = "| Dataset | Setting | Local: bài | Local: của bạn | Global: bài | Global: của bạn | FedKT > FedAvg (số seed) |\n|---|---|---|---|---|---|---|\n"
    rows = ""
    for ds in DS:
        for st in ST:
            a = main[(main.table == "T3") & (main.dataset == ds) & (main.setting == st)].iloc[0]
            b = main[(main.table == "T4") & (main.dataset == ds) & (main.setting == st)].iloc[0]
            rows += (f"| {DS[ds]} | {ST[st]} | {a.fedkt_paper - a.fedavg_paper:+.2f} | {a.kt_minus_avg} | "
                     f"{b.fedkt_paper - b.fedavg_paper:+.2f} | {b.kt_minus_avg} | local {a.kt_gt_avg_seeds}, global {b.kt_gt_avg_seeds} |\n")
    return hdr + rows


def mae_by_ds(df):
    return df.groupby("dataset").MAE.mean()


all_seeds = (main.kt_gt_avg_seeds.str.split("/").str[0] == main.kt_gt_avg_seeds.str.split("/").str[1]).all()
mm, mf = mae_by_ds(main), mae_by_ds(fit)

# ô mà thứ tự IID so với Non-IID của FedAvg bị đảo so với bài
flips = []
for tab, label in (("T3", "local"), ("T4", "global")):
    for ds in DS:
        a = main[(main.table == tab) & (main.dataset == ds)].set_index("setting")
        if {"iid", "noniid"} <= set(a.index):
            ours = float(a.loc["iid", "fedavg_mean"] - a.loc["noniid", "fedavg_mean"])
            paper = float(a.loc["iid", "fedavg_paper"] - a.loc["noniid", "fedavg_paper"])
            if ours * paper < 0:
                flips.append(f"{DS[ds]} ({label})")

bt = "| Dataset | Số ô so sánh | Batch 16 | Batch 32 | Batch 64 |\n|---|---|---|---|---|\n" + "".join(
    f"| {DS[r.dataset]} | {int(r.n_cells)} | {r['16']:.2f} | {r['32']:.2f} | {r['64']:.2f} |\n" for _, r in bat.iterrows())

doc = f"""# Kết quả tái thực nghiệm FedKT (credit scoring) và so sánh với bài báo

Phạm vi: 3 dataset nhỏ (Australian, Loan, Taiwan). GMSC chưa được hiệu chỉnh (xem mục Hạn chế).
Số liệu là **trung bình ± độ lệch chuẩn qua {n_seeds} seed**, tính trên accuracy trung bình 50 round (tương ứng Bảng 3 và 4 của bài).
Toàn bộ tham số dùng chung được khoá trong `src/config.py` và mô tả ở `docs/SHARED_PARAMS.md`. Mọi con số dưới đây dùng **cấu hình chính đã khoá**:
IID = `{PROTOCOL['partition_of']['iid']}` (stratified), Non-IID = `{PROTOCOL['partition_of']['noniid']}` (xáo trộn rồi chia đều).

## 1. So sánh với bài (accuracy %, mean ± std)

### Bảng 3 — Local test accuracy trung bình
{table("T3")}
### Bảng 4 — Global test accuracy trung bình
{table("T4")}
## 2. Mức cải thiện của FedKT so với FedAvg (điểm phần trăm)
{gains()}
Cột "của bạn" là trung bình ± độ lệch chuẩn của chênh lệch FedKT − FedAvg **tính theo từng seed** (cùng chia dữ liệu, cùng danh sách client được chọn).

## 3. Ảnh hưởng của batch size
Sai số tuyệt đối trung bình so với Bảng 3 và 4 của bài (điểm phần trăm; càng nhỏ càng gần bài), chỉ tính trên các ô có đủ dữ liệu ở cả ba batch size:

{bt}
## 4. Nhận xét

**Khớp với bài**
- FedKT cao hơn FedAvg {"trong mọi ô và ở mọi seed" if all_seeds else "trong hầu hết các ô"} (cả local lẫn global), kể cả khi so theo từng seed.
- Sai số tuyệt đối trung bình so với bài: Australian {mm['australian']:.2f}, Loan {mm['loan']:.2f}, Taiwan {mm['taiwan']:.2f} điểm; Loan và Taiwan gần bài hơn nhiều.

**Không khớp / cần thận trọng**
- Australian lệch nhiều nhất và mức cải thiện của FedKT nhỏ hơn bài; dataset chỉ có 690 mẫu, độ lệch chuẩn qua seed lớn hơn khoảng cách giữa các baseline.
- Thứ tự IID so với Non-IID của FedAvg bị đảo so với bài ở: {", ".join(flips) if flips else "không ô nào"}. Điều này không kết luận được vì bài không mô tả rõ cách chia.
- Chênh lệch giữa FedCodl, FedProx và FedAvg thường nhỏ hơn độ lệch chuẩn, nên **thứ tự giữa các baseline không đáng tin**; chỉ thứ tự "FedKT cao nhất" là ổn định.
- FedCodl trong bài đôi khi thấp hơn FedAvg ở local accuracy, trong thực nghiệm này thì không; chưa giải thích được.

## 5. Hạn chế (cần nêu trong báo cáo)
1. Bài không công bố code. Các lựa chọn đánh dấu `assumed` trong `docs/SHARED_PARAMS.md` là suy đoán; đây là hiệu chỉnh, không phải kiểm chứng độc lập.
2. Cách chia dữ liệu: cách ghép tên IID/Non-IID đã được chọn theo nghĩa của tên (không theo mức khớp số). Cấu hình khớp số nhất trong lưới (IID = random, Non-IID = stratified, tức đảo nghĩa) cho sai số trung bình thấp hơn
   ({fit.MAE.mean():.2f} so với {main.MAE.mean():.2f} điểm); nó chỉ được giữ làm tham khảo (`results/tables/final_fit_batch{FROZEN_SHARED['batch_size']}.csv`). Chia theo skew nhãn mạnh (Dirichlet, sắp theo nhãn) lệch bài rất nhiều (sai số 5-17 điểm).
3. Các giá trị batch size, cách chia, β của FedCodl, μ của FedProx được chọn có tham chiếu số của bài (hiệu chỉnh), nên mức khớp có phần được tối ưu theo bài.
4. GMSC: chưa hiệu chỉnh; bài ghi 67 chiều còn file chỉ có 10 đặc trưng. Kết quả GMSC trong `results/raw` (nếu có) chỉ là lần chạy mặc định, 1 seed.
5. Chưa kiểm định thống kê chính thức (chỉ so theo từng seed và mean ± std). Taiwan ở batch 16/64 chỉ có 2 seed.

## 6. File liên quan
- `results/tables/final_main_batch{FROZEN_SHARED['batch_size']}.csv`: bảng đầy đủ ở mục 1-2 (cấu hình chính).
- `results/tables/final_fit_batch{FROZEN_SHARED['batch_size']}.csv`: cấu hình khớp số nhất (tham khảo).
- `results/tables/final_batch_effect.csv`: ảnh hưởng batch size.
- `results/tables/calibration_scores.csv`: xếp hạng các cấu hình chia dữ liệu / FC / β.
- Tái tạo: `python experiments/report.py` rồi `python experiments/make_report_md.py`.
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "ket_qua_so_sanh.md").write_text(doc, encoding="utf-8")
print("Đã ghi docs/ket_qua_so_sanh.md")
print(doc[doc.index("## 4. Nhận xét"):doc.index("## 5.")])
