# Tái thực nghiệm FedKT cho bài toán chấm điểm tín dụng

Tái hiện bài báo **"A novel federated learning approach with knowledge transfer for credit scoring"** (Wang, Xiao, Wang & Yao; *Decision Support Systems*, 177, 114084, 2024; Elsevier) bằng Python/PyTorch.

**Trạng thái: đã hoàn thiện.** Dự án phục vụ học tập, nghiên cứu và kiểm tra khả năng tái lập kết quả.

Nhóm thực hiện: Nguyễn Minh Đức (24022628), Nguyễn Khánh An (24022600).

## 1. Tổng quan

Dữ liệu khách hàng nằm rải rác ở nhiều tổ chức tài chính và không thể gộp lại vì quyền riêng tư. Học liên kết (Federated Learning) cho phép nhiều bên cùng huấn luyện mô hình mà không chia sẻ dữ liệu gốc.

**FedKT** là phương pháp trong bài báo. Mỗi vòng, client:

1. **Sao chép tham số:** lấy các lớp ẩn từ mô hình toàn cục, giữ lớp ra riêng của mình.
2. **Tối ưu cục bộ:** huấn luyện 10 epoch với hàm mất mát tổng hợp
   `L_local = λ·L_hard + (1−λ)·L_soft`,
   trong đó `L_hard` là cross-entropy với nhãn thật và `L_soft` là MSE giữa đầu ra xác suất của mô hình toàn cục (Q_g) và cục bộ (Q_l) (chưng cất tri thức, T = 1).

Server tổng hợp tham số bằng trung bình có trọng số theo kích thước dữ liệu.

## 2. Mô hình so sánh

| Phương pháp | Mô tả |
|---|---|
| Client k | Mỗi client huấn luyện riêng, không liên kết |
| FedAvg | Trung bình có trọng số tham số |
| FedProx | FedAvg có số hạng proximal (μ = 0.01, chưa tinh chỉnh) |
| FedCodl | Phương pháp chưng cất đối chứng |
| **FedKT** | Phương pháp đề xuất (sao chép lớp ẩn + L_local), λ = 0.3 |

## 3. Dữ liệu

| Bộ dữ liệu | Số mẫu | Không vỡ nợ / Vỡ nợ | Tỷ lệ mất cân bằng | Số chiều | Số client (K) |
|---|---|---|---|---|---|
| Australian | 690 | 383 / 307 | 1.25 | 42 | 10 |
| Loan | 1,225 | 902 / 323 | 2.80 | 28 | 20 |
| Taiwan | 30,000 | 23,364 / 6,636 | 3.52 | 91 | 50 |
| GMSC | 150,000 | 139,974 / 10,026 | 13.96 | 67 (bài báo) | 150 |

Tệp dữ liệu đặt trong thư mục `Dataset/`: `australian.dat`, `loan.csv`, `default of credit card clients.xls` (Taiwan), `cs-training.csv` (GMSC).

**Lưu ý về GMSC.** Với 10 đặc trưng gốc, mọi phương pháp đều trùng tỷ lệ lớp đa số (~93.3–93.5%), nên không phân biệt được. Nhóm dùng biến thể **`gmscfe`** (10 đặc trưng chuẩn hóa MinMax + one-hot 6 phân vị mỗi cột = 70 chiều, gần 67 chiều của bài báo). Bảng kết quả dùng `gmscfe`; `gmsc` (10 đặc trưng) được giữ để đối chiếu.

## 4. Thiết lập thực nghiệm (cố định)

- 50 vòng, 20% client tham gia mỗi vòng, E = 10 epoch cục bộ
- SGD, lr = 0.5, momentum = 0.5; batch size 32 (giả định của nhóm)
- MLP 2 lớp ẩn × 100 + 1 lớp FC; PyTorch 1.11.0
- Mỗi client chia 80% huấn luyện / 20% kiểm tra
- Kịch bản chia dữ liệu: **IID** (đủ hai nhãn theo tỷ lệ gốc) và **Non-IID** (lệch phân phối nhãn); ngoài ra có `dirichlet` và `sorted` trong nghiên cứu phân phối
- Chỉ số: accuracy cục bộ, accuracy toàn cục (trung bình 50 vòng), accuracy vòng cuối, độ lệch chuẩn (Non-IID)
- Số seed: 5–10 (Australian, Loan), 5 (Taiwan), 3 (gmscfe)

## 5. Kết quả chính

Accuracy trung bình (%), IID và Non-IID. FedKT cao nhất trên **24/24** ô xét theo trung bình, hơn phương pháp tốt nhì từ +0.34 đến +3.08 điểm. Ví dụ (IID, local):

| Dataset | FedAvg | FedProx | FedCodl | **FedKT** |
|---|---|---|---|---|
| Australian | 84.09 | 83.93 | 84.66 | **85.95** |
| Loan | 68.23 | 68.27 | 70.47 | **73.55** |
| Taiwan | 78.43 | 79.13 | 79.66 | **81.01** |
| GMSC (gmscfe) | 92.21 | 91.29 | 92.84 | **93.34** |

Bảng đầy đủ: `results/tables/main_table.csv`; hình: `results/figures/table_local_global.png`, `table_final.png`.

**Ảnh hưởng của λ** (`results/figures/table_lambda.png`): Australian tốt nhất ở λ = 0.3; Loan và Taiwan tốt nhất ở λ = 0.1. Khi λ = 0.9, FedKT có thể thấp hơn FedAvg (ví dụ Loan IID local 67.29 so với 67.49). Chưa chạy quét λ cho GMSC.

### Điều cần nói rõ

- Xét theo **từng seed**, FedKT không luôn cao nhất (ví dụ Australian IID global: 7/10 seed). Khẳng định "cao nhất" áp dụng cho **trung bình**.
- Với phân chia lệch nhãn mạnh (`dirichlet`, `sorted`), FedKT thấp hơn ở local và final.
- Số liệu không trùng hoàn toàn bài báo; bài báo không công bố đủ chi tiết (batch size, cách chia client, tiền xử lý) nên nhóm chọn giả định riêng. Việc FedKT đứng đầu là kết quả tái lập, không phải sao chép số liệu.
- FedProx (μ = 0.01) chưa tinh chỉnh; `gmscfe` chỉ 3 seed; Client k chưa chạy cho `gmscfe`.
- Cách giữ lớp ra riêng (`fc_mode=keep`) là cách diễn giải của nhóm.

## 6. Cấu trúc dự án

```
├── Dataset/            # Dữ liệu gốc
├── src/
│   ├── config.py       # Cấu hình, thông tin dataset
│   ├── data.py         # Đọc, tiền xử lý, chia client (gồm gmscfe)
│   ├── models.py       # MLP
│   ├── federated.py    # Vòng lặp học liên kết
│   ├── algorithms.py
│   └── methods/        # client_k, fedavg, fedprox, fedcodl, fedkt
├── experiments/
│   ├── run_all.py          # Chạy toàn bộ mô hình × dataset × kịch bản
│   ├── sweep_lambda.py     # Quét λ
│   ├── gmscfe_run.py       # Thực nghiệm GMSC mở rộng
│   ├── main_table.py, partition_report.py, sweep_report.py
│   ├── table_figure.py, table_lambda_figure.py, plot_extra.py
│   └── compare_paper.py    # So với số liệu bài báo
├── results/
│   ├── calib/, raw/, raw_lambda/   # JSON kết quả từng lần chạy
│   ├── tables/                     # Bảng CSV
│   └── figures/                    # Hình
└── docs/               # SHARED_PARAMS.md, ket_qua_so_sanh.md
```

## 7. Cài đặt

```bash
git clone YOUR_REPOSITORY_URL
cd FedKT-Credit-Scoring
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Thay `YOUR_REPOSITORY_URL` bằng đường dẫn repository của nhóm. Đặt 4 tệp dữ liệu vào `Dataset/` như mục 3 (tuân thủ giấy phép của từng bộ dữ liệu).

## 8. Chạy lại thực nghiệm

```bash
# 1) Chạy tất cả (ví dụ máy 16 nhân)
python experiments/run_all.py --procs 14 --seeds 0 1 2 3 4

# 2) Tạo bảng và hình
python experiments/partition_report.py
python experiments/main_table.py
python experiments/plot_extra.py
python experiments/table_figure.py

# 3) Quét λ và bảng λ
python experiments/sweep_lambda.py
python experiments/sweep_report.py
python experiments/table_lambda_figure.py --variant keep
```

Tùy chọn của `run_all.py`: `--datasets`, `--parts`, `--seeds`, `--procs`, `--fresh`.

## 9. Khả năng tái lập

Cố định hạt giống (seed) cho mỗi lần chạy; tham số giao thức đóng băng trong `src/config.py` (có băm kiểm tra); mọi kết quả thô lưu dạng JSON trong `results/`. Số liệu báo cáo đều lấy từ các tệp này.

## 10. Hạn chế

- Chỉ 4 bộ dữ liệu công khai và một MLP nhỏ.
- Chỉ đo accuracy; chưa có AUC, F1, Precision/Recall (accuracy dễ gây hiểu nhầm với dữ liệu mất cân bằng như GMSC).
- Quyền riêng tư và độ phức tạp truyền thông chưa được phân tích sâu.
- Khác biệt so với bài báo do tiền xử lý, cách chia client, batch size và siêu tham số.

Hướng phát triển: dữ liệu dị thể phức tạp hơn, giảm chi phí truyền thông, phân tích lý thuyết, bổ sung AUC/F1 và quét μ cho FedProx.

## 11. Trích dẫn

Wang, Z., Xiao, J., Wang, L., & Yao, J. (2024). A novel federated learning approach with knowledge transfer for credit scoring. *Decision Support Systems*, 177, 114084.

Mã nguồn thuộc giấy phép do nhóm chọn; bài báo và dữ liệu bên thứ ba theo giấy phép riêng.
