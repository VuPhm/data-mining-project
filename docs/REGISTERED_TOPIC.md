# Bản đăng ký với giảng viên
Nguồn: người dùng cung cấp trực tiếp ngày 28/10/2026. Đây là căn cứ phạm vi chính thức; đề xuất trong chat cũ không tự thay thế bản này.

**Bài toán:** Phân loại các hoạt động hằng ngày và thể thao của con người từ dữ liệu cảm biến chuyển động gắn trên cơ thể. Khảo sát ảnh hưởng của kỹ thuật giảm chiều và so sánh các hướng Machine Learning, CNN và Transformer trong bài toán nhận diện hoạt động.
**Thuật toán và Mô hình:** UMAP để giảm chiều và trực quan hóa đặc trưng; SVM làm mô hình Machine Learning baseline; InceptionTime thuộc nhóm CNN cho dữ liệu chuỗi thời gian; TST (Time Series Transformer) thuộc nhóm Transformer. So sánh hiệu quả giữa các hướng tiếp cận trên cùng tập dữ liệu.
**Tập dữ liệu:** Daily and Sports Activities Dataset (DSADS) – UCI Machine Learning Repository. Tập dữ liệu gồm 9.120 mẫu của 19 hoạt động do 8 người thực hiện, thu từ các cảm biến quán tính đặt tại nhiều vị trí trên cơ thể.
**Link tập dữ liệu:** [https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)
**Quy trình tiền xử lý:** Đọc và kiểm tra dữ liệu → tổ chức dữ liệu cảm biến theo hoạt động và người tham gia → chia Train/Validation/Test theo người tham gia → chuẩn hóa dữ liệu → áp dụng UMAP cho nhánh Machine Learning → chuẩn bị dữ liệu chuỗi thời gian cho CNN/Transformer → huấn luyện SVM, InceptionTime và TST → đánh giá và so sánh kết quả.
**Ứng dụng demo:** Ứng dụng web đơn giản cho phép chọn hoặc tải một mẫu dữ liệu cảm biến và dự đoán hoạt động tương ứng.
**Chức năng:** Hiển thị dữ liệu cảm biến → lựa chọn mô hình → dự đoán hoạt động và độ tin cậy → so sánh kết quả giữa SVM, InceptionTime và TST → trực quan hóa không gian đặc trưng sau giảm chiều.
**Đơn vị đo hiệu suất mô hình:** Accuracy, Precision, Recall, Macro F1-Score, Confusion Matrix; có thể bổ sung thời gian huấn luyện và thời gian dự đoán để so sánh các mô hình.
