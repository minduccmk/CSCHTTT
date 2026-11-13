# Tham số dùng chung cho người A và người B

Nguồn duy nhất của các giá trị này là `src/config.py` (`FROZEN_SHARED` và `PROTOCOL`). Tài liệu này chỉ giải thích.
Chạy `python src/config.py` để in bảng hiện hành và `FROZEN_HASH`.

Phân công: **A** = FedAvg, FedKT. **B** = Client k, FedProx, FedCodl.

Nhãn nguồn: **paper** = bài báo ghi rõ (mục 4.1.2). **assumed** = bài không nêu, hai người đã thống nhất chọn.

## 1. Tham số huấn luyện chung (khoá, cả hai bắt buộc dùng đúng)

| Tham số | Giá trị | Nguồn | Ghi chú |
|---|---|---|---|
| `rounds` | 50 | paper | Số vòng giao tiếp |
| `participation` | 0.2 | paper | 20% client mỗi round; danh sách client chọn **cố định** giữa các phương pháp (`make_selections`) |
| `local_epochs` | 10 | paper | E = 10 |
| `lr` | 0.5 | paper | SGD |
| `momentum` | 0.5 | paper | SGD |
| `hidden` | 100 | paper | MLP: 2 lớp ẩn × 100 đơn vị + 1 lớp FC ra |
| `n_classes` | 2 | paper | default / non-default |
| `train_ratio` | 0.8 | paper | 80% train / 20% test trên **từng client** |
| `batch_size` | 32 | **assumed** | Bài không nêu. Thử 16/32/64 trên 3 dataset: 32 cho sai số trung bình thấp nhất so với bài |

Mọi file kết quả ghi kèm `shared_hash` của 9 tham số này. `FROZEN_HASH` hiện tại: `ba4b5f322e`.

## 2. Quy trình chung (khoá)

| Mục | Giá trị | Nguồn | Ghi chú |
|---|---|---|---|
| Dataset bảng chính | Australian (K=10), Loan (K=20), Taiwan (K=50) | — | GMSC (K=150) chưa hiệu chỉnh nên không vào bảng chính; chạy riêng bằng `--datasets gmsc` |
| Seed kết quả cuối | 0, 1, 2, 3, 4 | assumed | Mỗi seed quyết định cách chia dữ liệu, dãy client được chọn và khởi tạo model; `--final` dùng đủ 5 seed |
| IID | `iid`: stratified, mọi client cùng tỷ lệ nhãn | assumed | Bài chỉ nói "chia K phần bằng nhau" |
| Non-IID | `random`: xáo trộn rồi chia K phần bằng nhau, tỷ lệ nhãn mỗi client lệch nhẹ | assumed | Bài chỉ nói "label distribution skew do imbalance". Skew mạnh (Dirichlet, sắp theo nhãn) lệch bài rất nhiều (sai số 5-17 điểm), nên không dùng |
| Tiền xử lý | Biến liên tục → MinMax [0, 1]; biến phân loại → one-hot | assumed | Cài trong `src/data.py`. Số chiều: Australian 42, Loan 28, Taiwan 91 khớp Bảng 1 của bài; GMSC 10 (bài ghi 67) |
| Số đo | `train_loss`, `local_acc`, `global_acc`, lấy trung bình 50 round | paper | local = model cục bộ của client được chọn trên test của nó; global = model toàn cục trên test của mọi client |

Về cách ghép tên IID/Non-IID: cấu hình khớp số nhất trong lưới hiệu chỉnh đảo nghĩa (IID = random, Non-IID = stratified, sai số trung bình 1.24 điểm). Cấu hình đã khoá theo đúng nghĩa của tên cho sai số 1.72 điểm. Hai người chọn giữ **nghĩa của tên** để báo cáo không gây hiểu nhầm; cấu hình đảo nghĩa chỉ là tham khảo (`final_fit_*.csv`).

## 3. Tham số riêng (mỗi người tự tinh chỉnh, trong file của mình)

| Chủ | Tham số | Mặc định | Nguồn | Phạm vi tinh chỉnh |
|---|---|---|---|---|
| A | `lam` (λ) | 0.3 | paper | Chỉ quét {0.1, 0.3, 0.5, 0.7, 0.9} cho Bảng 5–6; bảng chính luôn 0.3 |
| A | `kt_temperature` | 1.0 | paper | Không đổi (chỉ 2 nhãn) |
| A | `fc_mode` | `keep` | assumed | `keep` / `global` / `reinit`; đã thử cả ba, chênh nhau ≤ 0.1 điểm sai số trung bình |
| B | `prox_mu` (μ của FedProx) | 0.01 | assumed | Bài không nêu |
| B | `codl_beta` (β của FedCodl) | 1.0 | assumed | Đã thử 0.1 và 1.0, chênh ≤ 0.1 điểm sai số trung bình |
| B | `codl_temperature` | 1.0 | assumed | Bài không nêu |
| B | `client_k_epochs` | 100 | assumed | Bài không nêu số epoch của Client k |

Quy tắc tinh chỉnh tham số riêng:
1. Tinh chỉnh trên **cùng lưới, cùng seed (0–4) và đúng các tham số chung ở mục 1–2**.
2. Ghi lại giá trị chọn và lý do (một dòng trong bảng trên, qua PR của mình).
3. Không chọn giá trị chỉ để một phương pháp khớp số của bài hơn; mục tiêu là so sánh công bằng giữa các phương pháp.

## 4. Chạy và đặt tên kết quả

| Việc | Lệnh |
|---|---|
| Xem tham số hiện hành | `python src/config.py` |
| Người A chạy phần của mình | `python experiments/run_all.py --owner A --final` |
| Người B chạy phần của mình | `python experiments/run_all.py --owner B --final` |
| Quét λ (A) | `python experiments/run_all.py --owner A --sweep_lambda --final` |
| Kiểm tra khi gộp | `python experiments/check_shared.py` |
| Tạo bảng và hình | `python experiments/analyze.py`, `python experiments/report.py` |

Mỗi phương pháp ghi một file riêng: `results/raw/{dataset}_{setting}_{method}_seed{n}.json`. Hai người ghi file khác nhau, nên không bao giờ ghi đè nhau. Mỗi file chứa `config`, `shared_hash`, `frozen` và `result`.

`check_shared.py` thoát với mã 1 nếu có file có tham số chung khác giá trị đã khoá. `analyze.py` và `report.py` tự bỏ qua các file đó (dùng `--include-nonfrozen` nếu cố ý). Ngoại lệ duy nhất: nhóm batch size 16/64 trong `results/calib` là phân tích độ nhạy, cố ý khác giá trị khoá.

## 5. Điều chưa thống nhất / còn mở
- GMSC: chưa rõ cách đạt 67 chiều như bài.
- Cách chia dữ liệu thật của bài (ảnh hưởng nhiều nhất đến Australian).
- Hệ số distillation của FedCodl và μ của FedProx trong bài gốc.
