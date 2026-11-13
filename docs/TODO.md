# TODO: Tái thực nghiệm "A federated learning approach with knowledge transfer for credit scoring" (FedKT)

## 0. Môi trường
- [ ] Python 3.9+, `pip install -r requirements.txt` (bài báo dùng PyTorch 1.11.0, bản mới hơn vẫn chạy được)
- [ ] Đặt dataset vào thư mục `Dataset/` (xem mục 1)

## 1. Dữ liệu (đặt trong `Dataset/`)
| Dataset | Nguồn | Tên file mong đợi | K clients | Dim sau tiền xử lý (bài báo) |
|---|---|---|---|---|
| Australian | UCI Statlog (Australian Credit Approval) | `australian.dat` | 10 | 42 |
| Loan data | Thomas et al., *Credit scoring and its applications* (SIAM 2002) | `loan.csv` (cột cuối = nhãn) | 20 | 28 |
| Taiwan | UCI "Default of Credit Card Clients" | `default of credit card clients.xls` (hoặc `.xlsx`/`.csv`) | 50 | 91 |
| GMSC | Kaggle "Give Me Some Credit" | `cs-training.csv` | 150 | 67 (xem lưu ý) |

- [ ] Tải/đặt 4 file trên; chạy `python experiments/check_data.py` để xem dim thực tế so với bài báo

## 2. Cài đặt thuật toán (đã có trong `src/`)
- [ ] MLP: 2 hidden layer x 100 unit + 1 FC đầu ra (`src/models.py`)
- [ ] Chia client IID (stratified) / Non-IID (`src/data.py`)
- [ ] Client k (học cục bộ, không cộng tác)
- [ ] FedAvg (McMahan 2017)
- [ ] FedProx (Li 2020), mu = 0.01
- [ ] FedCodl (Ni 2022): CE + distillation từ global model
- [ ] FedKT: copy các lớp đầu của global (trừ FC cuối) + loss `lambda*CE + (1-lambda)*MSE(softmax_global, softmax_local)`, lambda = 0.3, T = 1

## 3. Thiết lập thực nghiệm (theo bài báo)
- [ ] 50 communication rounds, 20% client chọn mỗi round (cố định giữa các phương pháp)
- [ ] E = 10 local epochs, SGD lr = 0.5, momentum = 0.5
- [ ] Mỗi client: 80% train / 20% test
- [ ] Metric: train loss, local test acc, global test acc (trung bình trên client)

## 4. Chạy
- [ ] Chạy 1 cấu hình: `python experiments/run_experiment.py --dataset australian --setting iid`
- [ ] Chạy tất cả (4 dataset x IID/Non-IID): `python experiments/run_all.py`
- [ ] Quét lambda {0.1,0.3,0.5,0.7,0.9}: `python experiments/run_all.py --sweep_lambda`
- [ ] Nhiều seed (khuyến nghị 3-5): `--seeds 0 1 2`

## 5. Phân tích
- [ ] `python experiments/analyze.py` -> `results/tables/*.csv` (Bảng 2-6) và `results/figures/*.png` (Hình 3-7)
- [ ] So sánh với số liệu bài báo, ghi lại chênh lệch vào README

## 6. Điểm bài báo KHÔNG nêu rõ (mình đã chọn giả định, có thể chỉnh trong `src/config.py`)
- Batch size: không nêu -> dùng 32
- Cách chuẩn hoá: "normalized" -> MinMax [0,1]
- Non-IID: bài báo chỉ nói "label distribution skew", "random sampling" -> mặc định `noniid` = chia ngẫu nhiên (không stratify); có thêm `dirichlet` (alpha=0.5) để skew nhãn mạnh hơn
- Bảng 2 (Client k, FedAvg...): không rõ là accuracy vòng cuối hay trung bình -> code lưu cả hai
- FedCodl: hệ số distillation không nêu -> beta = 1.0 (xem lại bài Ni et al. 2022 nếu muốn khớp chính xác)
- Số epoch của Client k: không nêu -> mặc định 100 epoch (~ tổng số epoch một client thực sự train trong FL)
- GMSC chỉ có 10 đặc trưng liên tục nhưng bài báo ghi dim = 67 -> chưa tái tạo được con số này
- Số mô tả đặc trưng Australian trong bài (8 liên tục/6 rời rạc) khác UCI; code dùng cột categorical sao cho ra đúng 42 chiều
