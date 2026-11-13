# Đóng góp: quy trình cho người A và người B

## Nguyên tắc
1. **Mỗi người chỉ sửa file của mình** (bảng trong `CODEOWNERS`). Người A: `src/methods/fedavg.py`, `fedkt.py`. Người B: `src/methods/client_k.py`, `fedprox.py`, `fedcodl.py`.
2. **File dùng chung không tự sửa.** `src/config.py`, `data.py`, `models.py`, `algorithms.py`, `federated.py`, `methods/base.py`, `experiments/` chỉ đổi qua một PR riêng nhỏ, cả hai cùng duyệt.
3. **Tham số chung đã khoá** (`FROZEN_SHARED`, `PROTOCOL` trong `src/config.py`, giải thích ở `docs/SHARED_PARAMS.md`). Muốn đổi, mở PR có tiêu đề `shared:` kèm lý do và số liệu; sau khi gộp, cả hai chạy lại phần của mình.

## Quy trình Git
1. Một người tạo repo từ thư mục này, commit và gắn tag `v0-shared`. Cả hai bắt đầu từ tag này.
2. Mỗi người làm trên nhánh riêng: `feat/fedkt-fedavg` (A), `feat/baselines` (B).
3. Trước khi mở PR: `python experiments/check_shared.py` phải in `OK`.
4. Sau khi cả hai gộp: `python experiments/analyze.py` và `python experiments/report.py` để tạo bảng từ kết quả của cả hai.

## Không commit
- `Dataset/` (dữ liệu), `.venv/`, `__pycache__/`.
- `results/tables/` và `results/figures/`: hai người cùng tạo ra cùng tên file nên sẽ xung đột; luôn tạo lại bằng lệnh ở trên.

## Có thể commit
- `results/raw/`, `results/raw_lambda/`, `results/calib/`: tên file có tên phương pháp nên không trùng giữa hai người. Mỗi file đã kèm cấu hình và `shared_hash`.

## Thêm phương pháp mới
Thêm một file trong `src/methods/` (ghi đè `init_local`, `begin_local`, `local_loss` khi cần), thêm một dòng vào `src/methods/__init__.py` và một dòng vào `OWNER` trong `src/config.py`.
