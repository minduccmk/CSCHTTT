# Kết quả tái thực nghiệm FedKT (credit scoring) và so sánh với bài báo

Phạm vi: 3 dataset nhỏ (Australian, Loan, Taiwan). GMSC chưa được hiệu chỉnh (xem mục Hạn chế).
Số liệu là **trung bình ± độ lệch chuẩn qua 5 seed**, tính trên accuracy trung bình 50 round (tương ứng Bảng 3 và 4 của bài).
Toàn bộ tham số dùng chung được khoá trong `src/config.py` và mô tả ở `docs/SHARED_PARAMS.md`. Mọi con số dưới đây dùng **cấu hình chính đã khoá**:
IID = `iid` (stratified), Non-IID = `random` (xáo trộn rồi chia đều).

## 1. So sánh với bài (accuracy %, mean ± std)

### Bảng 3 — Local test accuracy trung bình
| Dataset | Setting | FedAvg (bài / của bạn) | FedProx (bài / của bạn) | FedCodl (bài / của bạn) | FedKT (bài / của bạn) |
|---|---|---|---|---|---|
| Australian | IID | 78.07 / 85.21±3.16 | 78.43 / 84.84±3.34 | 80.42 / 85.74±2.52 | 82.93 / 87.37±2.31 |
| Australian | Non-IID | 82.78 / 83.39±2.07 | 82.64 / 83.30±1.72 | 78.57 / 83.90±1.48 | 87.64 / 85.09±1.25 |
| Loan | IID | 67.33 / 67.49±1.53 | 66.88 / 67.69±1.64 | 67.42 / 69.85±0.96 | 72.21 / 73.09±0.56 |
| Loan | Non-IID | 65.29 / 68.02±1.76 | 65.67 / 68.28±1.36 | 69.33 / 69.65±1.43 | 72.88 / 72.14±1.86 |
| Taiwan | IID | 78.73 / 78.43±0.36 | 79.16 / 79.13±0.47 | 77.82 / 79.66±0.35 | 81.20 / 81.01±0.55 |
| Taiwan | Non-IID | 78.01 / 78.73±0.42 | 78.99 / 79.24±0.41 | 77.78 / 79.77±0.30 | 80.85 / 81.26±0.29 |

### Bảng 4 — Global test accuracy trung bình
| Dataset | Setting | FedAvg (bài / của bạn) | FedProx (bài / của bạn) | FedCodl (bài / của bạn) | FedKT (bài / của bạn) |
|---|---|---|---|---|---|
| Australian | IID | 82.43 / 85.95±1.78 | 82.51 / 86.13±1.47 | 84.44 / 86.95±1.29 | 86.63 / 87.74±1.73 |
| Australian | Non-IID | 86.24 / 83.71±1.92 | 85.80 / 83.57±1.86 | 87.43 / 83.96±1.99 | 89.66 / 84.41±1.76 |
| Loan | IID | 70.52 / 71.20±0.81 | 70.73 / 71.54±0.77 | 73.71 / 72.84±0.74 | 73.79 / 74.56±0.96 |
| Loan | Non-IID | 69.70 / 70.44±1.36 | 69.33 / 70.65±1.18 | 73.51 / 71.83±1.38 | 73.53 / 73.39±0.68 |
| Taiwan | IID | 80.99 / 80.65±0.39 | 80.57 / 80.74±0.39 | 81.64 / 81.13±0.48 | 81.90 / 81.73±0.54 |
| Taiwan | Non-IID | 80.62 / 80.90±0.25 | 80.53 / 80.97±0.27 | 81.62 / 81.23±0.19 | 81.61 / 81.93±0.26 |

## 2. Mức cải thiện của FedKT so với FedAvg (điểm phần trăm)
| Dataset | Setting | Local: bài | Local: của bạn | Global: bài | Global: của bạn | FedKT > FedAvg (số seed) |
|---|---|---|---|---|---|---|
| Australian | IID | +4.86 | +2.16±1.13 | +4.20 | +1.79±1.38 | local 5/5, global 5/5 |
| Australian | Non-IID | +4.86 | +1.70±1.25 | +3.42 | +0.70±0.29 | local 5/5, global 5/5 |
| Loan | IID | +4.88 | +5.60±1.05 | +3.27 | +3.37±1.20 | local 5/5, global 5/5 |
| Loan | Non-IID | +7.59 | +4.12±1.07 | +3.83 | +2.95±1.09 | local 5/5, global 5/5 |
| Taiwan | IID | +2.47 | +2.58±0.34 | +0.91 | +1.08±0.23 | local 5/5, global 5/5 |
| Taiwan | Non-IID | +2.84 | +2.53±0.26 | +0.99 | +1.03±0.16 | local 5/5, global 5/5 |

Cột "của bạn" là trung bình ± độ lệch chuẩn của chênh lệch FedKT − FedAvg **tính theo từng seed** (cùng chia dữ liệu, cùng danh sách client được chọn).

## 3. Ảnh hưởng của batch size
Sai số tuyệt đối trung bình so với Bảng 3 và 4 của bài (điểm phần trăm; càng nhỏ càng gần bài), chỉ tính trên các ô có đủ dữ liệu ở cả ba batch size:

| Dataset | Số ô so sánh | Batch 16 | Batch 32 | Batch 64 |
|---|---|---|---|---|
| Australian | 4 | 3.48 | 3.54 | 3.91 |
| Loan | 4 | 1.39 | 1.11 | 1.59 |
| Taiwan | 2 | 0.82 | 0.60 | 0.64 |

## 4. Nhận xét

**Khớp với bài**
- FedKT cao hơn FedAvg trong mọi ô và ở mọi seed (cả local lẫn global), kể cả khi so theo từng seed.
- Sai số tuyệt đối trung bình so với bài: Australian 3.54, Loan 1.11, Taiwan 0.52 điểm; Loan và Taiwan gần bài hơn nhiều.

**Không khớp / cần thận trọng**
- Australian lệch nhiều nhất và mức cải thiện của FedKT nhỏ hơn bài; dataset chỉ có 690 mẫu, độ lệch chuẩn qua seed lớn hơn khoảng cách giữa các baseline.
- Thứ tự IID so với Non-IID của FedAvg bị đảo so với bài ở: Australian (local), Loan (local), Taiwan (local), Australian (global), Taiwan (global). Điều này không kết luận được vì bài không mô tả rõ cách chia.
- Chênh lệch giữa FedCodl, FedProx và FedAvg thường nhỏ hơn độ lệch chuẩn, nên **thứ tự giữa các baseline không đáng tin**; chỉ thứ tự "FedKT cao nhất" là ổn định.
- FedCodl trong bài đôi khi thấp hơn FedAvg ở local accuracy, trong thực nghiệm này thì không; chưa giải thích được.

## 5. Hạn chế (cần nêu trong báo cáo)
1. Bài không công bố code. Các lựa chọn đánh dấu `assumed` trong `docs/SHARED_PARAMS.md` là suy đoán; đây là hiệu chỉnh, không phải kiểm chứng độc lập.
2. Cách chia dữ liệu: cách ghép tên IID/Non-IID đã được chọn theo nghĩa của tên (không theo mức khớp số). Cấu hình khớp số nhất trong lưới (IID = random, Non-IID = stratified, tức đảo nghĩa) cho sai số trung bình thấp hơn
   (1.24 so với 1.72 điểm); nó chỉ được giữ làm tham khảo (`results/tables/final_fit_batch32.csv`). Chia theo skew nhãn mạnh (Dirichlet, sắp theo nhãn) lệch bài rất nhiều (sai số 5-17 điểm).
3. Các giá trị batch size, cách chia, β của FedCodl, μ của FedProx được chọn có tham chiếu số của bài (hiệu chỉnh), nên mức khớp có phần được tối ưu theo bài.
4. GMSC: chưa hiệu chỉnh; bài ghi 67 chiều còn file chỉ có 10 đặc trưng. Kết quả GMSC trong `results/raw` (nếu có) chỉ là lần chạy mặc định, 1 seed.
5. Chưa kiểm định thống kê chính thức (chỉ so theo từng seed và mean ± std). Taiwan ở batch 16/64 chỉ có 2 seed.

## 6. File liên quan
- `results/tables/final_main_batch32.csv`: bảng đầy đủ ở mục 1-2 (cấu hình chính).
- `results/tables/final_fit_batch32.csv`: cấu hình khớp số nhất (tham khảo).
- `results/tables/final_batch_effect.csv`: ảnh hưởng batch size.
- `results/tables/calibration_scores.csv`: xếp hạng các cấu hình chia dữ liệu / FC / β.
- Tái tạo: `python experiments/report.py` rồi `python experiments/make_report_md.py`.
