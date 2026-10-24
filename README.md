# Tái thực nghiệm FedKT cho bài toán chấm điểm tín dụng

## 1. Giới thiệu dự án

Dự án được thực hiện nhằm nghiên cứu, triển khai và đánh giá phương pháp học liên kết kết hợp chuyển giao tri thức (*Federated Learning with Knowledge Transfer*) được đề xuất trong bài báo:

**Tên bài báo:** A novel federated learning approach with knowledge transfer for credit scoring

**Tác giả:** Zhongyi Wang, Jin Xiao, Lu Wang và Jianrong Yao

**Tạp chí:** Decision Support Systems

**Tập:** 177

**Mã bài báo:** 114084

**Năm xuất bản:** 2024

**Nhà xuất bản:** Elsevier

Bài báo nghiên cứu phương pháp học liên kết kết hợp chuyển giao tri thức nhằm giải quyết bài toán chấm điểm tín dụng (*Credit Scoring*).

Trong thực tế, dữ liệu khách hàng thường được lưu trữ tại các tổ chức tài chính khác nhau. Việc chia sẻ trực tiếp dữ liệu có thể gặp những hạn chế liên quan đến quyền riêng tư, bảo mật và quy định quản lý dữ liệu.

Học liên kết (*Federated Learning*) cho phép nhiều bên phối hợp huấn luyện mô hình mà không cần tập trung toàn bộ dữ liệu gốc về một nơi. Việc kết hợp cơ chế chuyển giao tri thức có thể hỗ trợ quá trình học tập giữa các bên theo thiết kế của mô hình.

Trong dự án này, nhóm tập trung phân tích phương pháp FedKT được đề xuất trong bài báo, tái triển khai các thí nghiệm có thể kiểm chứng và đánh giá hiệu suất mô hình trên các bộ dữ liệu chấm điểm tín dụng.

Kết quả thực nghiệm của nhóm sẽ được so sánh với kết quả công bố trong bài báo gốc và các phương pháp đối chứng phù hợp.

## 2. Mục tiêu nghiên cứu

Dự án hướng đến các mục tiêu sau:

* Phân tích bài toán nghiên cứu, đóng góp và phương pháp được trình bày trong bài báo gốc.
* Tìm hiểu cơ chế học liên kết và chuyển giao tri thức trong phương pháp FedKT.
* Thu thập, khảo sát và tiền xử lý các bộ dữ liệu phục vụ thực nghiệm.
* Triển khai phương pháp FedKT dựa trên mô tả của tác giả.
* So sánh hiệu suất FedKT với các phương pháp đối chứng được lựa chọn.
* Đánh giá kết quả trên các bộ dữ liệu chấm điểm tín dụng.
* Thực hiện một số kịch bản thực nghiệm mở rộng nhằm khảo sát các yếu tố ảnh hưởng đến hiệu suất mô hình.
* Phân tích sự khác biệt giữa kết quả của nhóm và kết quả được công bố trong bài báo gốc.

### Các câu hỏi nghiên cứu

Dự án tập trung tìm hiểu những vấn đề sau:

1. Hiệu suất của FedKT so với các phương pháp đối chứng như thế nào?
2. Kết quả của FedKT có nhất quán trên các bộ dữ liệu khác nhau không?
3. Sự không đồng nhất trong phân phối dữ liệu giữa các bên ảnh hưởng như thế nào đến hiệu suất mô hình?
4. Nhóm có thể tái hiện các kết quả thực nghiệm được báo cáo trong bài báo gốc ở mức độ nào?
5. Những hạn chế và hướng cải tiến nào có thể được nhận diện từ kết quả thực nghiệm?

Các câu hỏi trên sẽ được đánh giá dựa trên những thí nghiệm thực tế được triển khai trong dự án.

## 3. Bài báo nghiên cứu gốc

Thông tin bài báo:

Wang, Z., Xiao, J., Wang, L., & Yao, J. (2024). *A novel federated learning approach with knowledge transfer for credit scoring*. Decision Support Systems, 177, 114084.

* Nhà xuất bản: Elsevier.
* Năm xuất bản: 2024.
* DOI: [Bổ sung DOI chính xác của bài báo].
* Trang công bố chính thức: [Bổ sung đường dẫn bài báo].

Bài báo gốc là tài liệu tham chiếu chính để nhóm tìm hiểu mô hình, thuật toán, thiết lập thực nghiệm và kết quả mà phương pháp FedKT đạt được.

## 4. Phương pháp nghiên cứu

Dự án tập trung vào phương pháp học liên kết kết hợp chuyển giao tri thức cho bài toán chấm điểm tín dụng.

Quy trình nghiên cứu dự kiến bao gồm các giai đoạn sau.

### 4.1. Phân tích bài báo

Nhóm phân tích các nội dung chính:

* Bối cảnh và động lực nghiên cứu.
* Phát biểu bài toán.
* Những hạn chế của các phương pháp trước đó.
* Đóng góp chính của bài báo.
* Kiến trúc mô hình FedKT.
* Thuật toán và cơ chế chuyển giao tri thức.
* Các bộ dữ liệu và phương pháp đối chứng.
* Thiết lập, kết quả và đánh giá thực nghiệm của tác giả.

### 4.2. Chuẩn bị dữ liệu

Dữ liệu được khảo sát và tiền xử lý trước khi đưa vào mô hình.

Các bước có thể bao gồm:

* Kiểm tra cấu trúc và chất lượng dữ liệu.
* Xử lý giá trị thiếu.
* Mã hóa các biến phân loại.
* Chuẩn hóa các đặc trưng số khi cần thiết.
* Xác định biến đầu vào và biến mục tiêu.
* Phân chia dữ liệu huấn luyện và đánh giá.
* Xây dựng các tập dữ liệu cục bộ tương ứng với những client tham gia học liên kết.

Quy trình thực tế sẽ được xác định dựa trên đặc điểm của từng bộ dữ liệu và thiết lập trong bài báo gốc.

Những thay đổi do nhóm tự thực hiện sẽ được ghi lại để phân biệt với quy trình của tác giả.

### 4.3. Triển khai FedKT

Nhóm triển khai các thành phần cần thiết của phương pháp FedKT theo mô tả trong bài báo, bao gồm những thành phần liên quan đến huấn luyện cục bộ, phối hợp giữa các client và server, chuyển giao tri thức và đánh giá mô hình.

Chi tiết thuật toán, cấu hình và cách triển khai sẽ được trình bày trong mã nguồn và tài liệu phương pháp của dự án.

### 4.4. So sánh với các phương pháp đối chứng

Hiệu suất FedKT sẽ được so sánh với các phương pháp đối chứng phù hợp.

Các phương pháp được sử dụng trong thí nghiệm tái thực nghiệm sẽ được xác định dựa trên bài báo gốc. Những phương pháp bổ sung do nhóm lựa chọn sẽ được phân biệt rõ với các phương pháp được tác giả sử dụng.

## 5. Dữ liệu thực nghiệm

Dự án dự kiến sử dụng bốn bộ dữ liệu liên quan đến bài toán tín dụng:

| STT | Bộ dữ liệu | Mô tả                                    |
| --- | ---------- | ---------------------------------------- |
| 1   | Australia  | Bộ dữ liệu chấp thuận tín dụng Australia |
| 2   | Taiwan     | Bộ dữ liệu tín dụng Taiwan               |
| 3   | Loan       | Bộ dữ liệu liên quan đến khoản vay       |
| 4   | GMSC       | Give Me Some Credit                      |

Các thông tin chi tiết về dữ liệu sẽ được thống kê từ chính những tệp mà nhóm sử dụng trong thực nghiệm.

Đối với mỗi bộ dữ liệu, nhóm sẽ ghi nhận:

* Nguồn công bố và đường dẫn tải dữ liệu.
* Phiên bản dữ liệu và định dạng tệp.
* Số lượng bản ghi.
* Số lượng đặc trưng.
* Biến mục tiêu và ý nghĩa các nhãn.
* Phân phối các lớp.
* Tình trạng giá trị thiếu.
* Quy trình tiền xử lý.
* Phương pháp phân chia dữ liệu giữa các client.

Tên bộ dữ liệu không đủ để khẳng định dữ liệu đang sử dụng hoàn toàn giống với dữ liệu của tác giả. Vì vậy, nhóm sẽ kiểm tra nguồn dữ liệu, phiên bản và quy trình tiền xử lý trước khi đưa ra kết luận về khả năng tái thực nghiệm.

### Quy định về dữ liệu

Tùy theo dung lượng và điều khoản sử dụng, dữ liệu gốc có thể không được lưu trực tiếp trong repository.

Trong trường hợp đó, nhóm sẽ cung cấp đường dẫn nguồn và hướng dẫn tải, tổ chức, tiền xử lý dữ liệu trong thư mục tài liệu của dự án.

Người sử dụng cần tuân thủ các điều khoản cấp phép tương ứng của từng bộ dữ liệu.

## 6. Công nghệ và công cụ

Python được sử dụng làm ngôn ngữ lập trình chính cho quá trình xử lý dữ liệu, triển khai mô hình và thực nghiệm.

Các thư viện và công cụ được lựa chọn tùy theo yêu cầu triển khai thực tế, có thể bao gồm:

| Công cụ          | Mục đích                                               |
| ---------------- | ------------------------------------------------------ |
| Python           | Ngôn ngữ lập trình chính                               |
| NumPy            | Tính toán số học                                       |
| pandas           | Đọc, phân tích và tiền xử lý dữ liệu                   |
| scikit-learn     | Các công cụ tiền xử lý và đánh giá mô hình khi phù hợp |
| PyTorch          | Triển khai mạng nơ-ron nếu phương pháp yêu cầu         |
| Matplotlib       | Trực quan hóa kết quả                                  |
| Jupyter Notebook | Khảo sát dữ liệu và phân tích kết quả                  |
| Git              | Quản lý phiên bản mã nguồn                             |
| GitHub           | Lưu trữ mã nguồn và cộng tác nhóm                      |

Danh sách thư viện chính thức sẽ được xác định từ môi trường thực nghiệm và lưu trong tệp `requirements.txt`.

## 7. Cấu trúc dự án

> Chỉnh sửa cây thư mục theo cách tổ chức mã nguồn, tài liệu và kết quả thực nghiệm thực tế.


FedKT-Credit-Scoring/
│
├── README.md
│   # Tài liệu giới thiệu tổng quan về dự án
│
├── requirements.txt
│   # Danh sách thư viện và phiên bản cần thiết
│
├── .gitignore
│   # Các tệp và thư mục không đưa lên GitHub
│
├── docs/
│   # Tài liệu phân tích bài báo và phương pháp nghiên cứu
│   
│
├── dataset/
│   # Hướng dẫn dữ liệu và dữ liệu thực nghiệm
│   
│
├── src/
│   # Mã nguồn triển khai
│   
│
├── configs/
│
├── experiments/
│
├── notebooks/
│
└── results/


## 8. Cài đặt môi trường

### 8.1. Tải mã nguồn

```bash
git clone YOUR_REPOSITORY_URL
cd FedKT-Credit-Scoring
```

Thay `YOUR_REPOSITORY_URL` bằng đường dẫn repository GitHub thực tế của nhóm.

### 8.2. Tạo môi trường Python

```bash
python -m venv .venv
```

Trên Windows, kích hoạt môi trường bằng PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Trên macOS hoặc Linux:

```bash
source .venv/bin/activate
```

### 8.3. Cài đặt thư viện

Nếu repository đã có tệp `requirements.txt`, chạy:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.4. Chuẩn bị dữ liệu

Thực hiện theo hướng dẫn trong `data/README.md` để tải và đặt dữ liệu vào đúng vị trí.

Đường dẫn dữ liệu đầu vào và định dạng tệp cần được thống nhất với cấu hình của chương trình.

## 9. Thiết kế thực nghiệm

Dự án chia các thí nghiệm thành hai nhóm chính.

### 9.1. Tái thực nghiệm bài báo gốc

Mục tiêu là triển khai lại các thí nghiệm của tác giả trong phạm vi thông tin, dữ liệu và tài nguyên tính toán mà nhóm có thể tiếp cận.

Các công việc bao gồm:

* Xác định thiết lập thực nghiệm trong bài báo.
* Triển khai phương pháp FedKT.
* Triển khai các phương pháp đối chứng được lựa chọn.
* Sử dụng các bộ dữ liệu tương ứng.
* Đánh giá mô hình theo các chỉ số phù hợp.
* So sánh kết quả của nhóm với kết quả được công bố trong bài báo.

Những điểm khác biệt về dữ liệu, cách chia client, cấu hình mô hình hoặc tài nguyên tính toán sẽ được ghi nhận.

### 9.2. Thực nghiệm mở rộng

Nhóm có thể thực hiện một số thí nghiệm bổ sung để đánh giá sâu hơn phương pháp FedKT, chẳng hạn:

* Thay đổi số lượng client.
* Thay đổi mức độ không đồng nhất dữ liệu.
* Thay đổi lượng dữ liệu huấn luyện.
* Thay đổi một số siêu tham số.
* So sánh tính ổn định của mô hình qua nhiều lần chạy.
* Đánh giá hiệu suất trên các bộ dữ liệu khác nhau.

Các hướng mở rộng cuối cùng sẽ được lựa chọn theo mục tiêu nghiên cứu, khả năng triển khai và tài nguyên tính toán của nhóm.

Kết quả của thí nghiệm mở rộng sẽ được phân biệt với kết quả tái thực nghiệm bài báo gốc.

## 10. Các chỉ số đánh giá

Tùy theo thiết lập của bài báo gốc, dự án có thể sử dụng các chỉ số phân loại sau:

| Chỉ số    | Ý nghĩa                                                                |
| --------- | ---------------------------------------------------------------------- |
| Accuracy  | Tỷ lệ dự đoán đúng trên tổng số mẫu                                    |
| Precision | Tỷ lệ dự đoán dương tính đúng trên tổng số mẫu được dự đoán dương tính |
| Recall    | Tỷ lệ mẫu dương tính thực tế được mô hình nhận diện                    |
| F1-score  | Trung bình điều hòa giữa Precision và Recall                           |
| ROC-AUC   | Đánh giá khả năng phân biệt hai lớp của mô hình                        |

Các chỉ số chính thức sẽ được xác định dựa trên bài báo gốc và mục tiêu thực nghiệm.

Đối với dữ liệu mất cân bằng, Accuracy có thể chưa phản ánh đầy đủ hiệu suất mô hình. Vì vậy, nhóm sẽ xem xét thêm các chỉ số phù hợp với bài toán chấm điểm tín dụng.

Định nghĩa lớp dương tính và cách tính các chỉ số sẽ được ghi rõ để bảo đảm tính nhất quán khi so sánh.

## 11. Kết quả thực nghiệm

Kết quả thực nghiệm sẽ được cập nhật sau khi các chương trình được triển khai và chạy thành công.

### 11.1. Kết quả tái thực nghiệm

Nhóm sẽ trình bày:

* Kết quả của FedKT trên từng bộ dữ liệu.
* Kết quả của các phương pháp đối chứng.
* So sánh với kết quả công bố trong bài báo gốc.
* Phân tích mức độ tương đồng và khác biệt.
* Giải thích những nguyên nhân có thể dẫn đến sai khác.

### 11.2. Kết quả thực nghiệm mở rộng

Nhóm sẽ phân tích ảnh hưởng của các yếu tố được khảo sát đến hiệu suất mô hình.

Các bảng số liệu và biểu đồ sẽ được lưu trong thư mục kết quả tương ứng.

Mọi số liệu được báo cáo phải xuất phát từ kết quả thực nghiệm thực tế và đi kèm thông tin về thiết lập thí nghiệm.

## 12. Khả năng tái lập thực nghiệm

Để hỗ trợ việc kiểm tra và tái lập kết quả, nhóm hướng đến việc công khai các thông tin sau:

* Nguồn và phiên bản dữ liệu.
* Các bước tiền xử lý.
* Cách phân chia dữ liệu giữa các client.
* Cấu hình mô hình và siêu tham số.
* Quy trình huấn luyện.
* Các chỉ số đánh giá.
* Phiên bản thư viện và môi trường lập trình.
* Hạt giống ngẫu nhiên khi phù hợp.
* Lệnh chạy các thí nghiệm.
* Kết quả thực nghiệm và cách phân tích.

Kết quả sẽ được phân biệt thành ba nhóm:

1. Kết quả được báo cáo trong bài báo gốc.
2. Kết quả do nhóm tự tái thực nghiệm.
3. Kết quả từ các thí nghiệm mở rộng.

Việc phân biệt này giúp người đọc đánh giá chính xác phạm vi và mức độ tái lập của dự án.

## 13. Nghiên cứu liên quan

Nhóm sẽ tìm hiểu và phân tích các nghiên cứu liên quan đến:

* Học liên kết trong lĩnh vực tài chính.
* Chuyển giao tri thức trong học máy phân tán.
* Chấm điểm tín dụng bằng các phương pháp học máy.
* Học máy bảo vệ quyền riêng tư.
* Các phương pháp học tập cộng tác trên dữ liệu không đồng nhất.

Phần tổng quan tài liệu sẽ bao gồm:

* Những nghiên cứu liên quan được bài báo gốc trích dẫn.
* Những nghiên cứu được công bố sau đó và có trích dẫn bài báo gốc.

Thông tin thư mục, đường dẫn nguồn và nội dung phân tích sẽ được bổ sung trong tài liệu nghiên cứu liên quan của dự án.

## 14. Hạn chế

Kết quả tái thực nghiệm có thể khác với kết quả của tác giả do sự khác biệt về:

* Phiên bản và cách tiền xử lý dữ liệu.
* Phương pháp phân chia dữ liệu giữa các client.
* Chi tiết triển khai mô hình.
* Siêu tham số và thiết lập huấn luyện.
* Phiên bản thư viện và môi trường phần mềm.
* Khởi tạo ngẫu nhiên và biến động giữa các lần chạy.

Nhóm sẽ mô tả rõ những khác biệt đã xác định được và tránh khẳng định tái thực nghiệm hoàn toàn thành công nếu chưa có đủ bằng chứng.

## 15. Thành viên nhóm

Dự án được thực hiện trong khuôn khổ bài tập lớn/báo cáo cuối kỳ của học phần.

| MSV        | Họ và tên             |
| ---        | --------------------- |
| 24022628   | Nguyễn Minh Đức       |
| 24022600   | Nguyễn Khánh An       |


## 16. Giấy phép và trích dẫn

Mã nguồn của dự án sẽ được cấp phép theo giấy phép do nhóm lựa chọn và công bố trong repository.

Bài báo gốc và các bộ dữ liệu bên thứ ba vẫn chịu sự điều chỉnh của điều khoản bản quyền và giấy phép tương ứng.

Khi sử dụng hoặc thảo luận về phương pháp được đề xuất trong nghiên cứu, cần trích dẫn bài báo gốc:

Wang, Z., Xiao, J., Wang, L., & Yao, J. (2024). *A novel federated learning approach with knowledge transfer for credit scoring*. Decision Support Systems, 177, 114084.

---

**Trạng thái dự án:** Đang thực hiện.

Dự án phục vụ mục đích học thuật, nghiên cứu thực nghiệm và tìm hiểu khả năng tái lập kết quả nghiên cứu.
