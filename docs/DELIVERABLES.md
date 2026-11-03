# Đầu ra và tiêu chí hoàn thành
Cập nhật 28/10/2026 theo yêu cầu trực tiếp của người dùng. Phạm vi đối chiếu: REGISTERED_TOPIC.md.
Đây là tiêu chí cho sản phẩm sẽ triển khai, không phải xác nhận các sản phẩm đã hoàn thành.

## 1. Báo cáo LaTeX và PDF
- Bàn giao đầy đủ nguồn .tex, .bib, hình/bảng và PDF biên dịch thành công; có hướng dẫn biên dịch.
- Tiếng Việt hiển thị đúng; chương/mục, công thức, hình, bảng được đánh số và tham chiếu nhất quán; trích dẫn nguồn học thuật và dataset.
- Nội dung: bài toán và phạm vi; DSADS; nền tảng SVM/UMAP, InceptionTime, TST; tiền xử lý và thiết kế thực nghiệm; kết quả; phân tích; demo; hạn chế và kết luận; tài liệu tham khảo.
- So sánh SVM có/không UMAP để khảo sát ảnh hưởng giảm chiều; so sánh ba họ mô hình trên cùng split và metric.
- Số liệu và hình lấy từ kết quả thực nghiệm có nguồn trong project; lưu thông tin cấu hình, môi trường và seed. Phân biệt số liệu lịch sử chưa tái lập.
- Kiểm tra PDF trực quan, không lỗi font, tràn trang, tham chiếu hoặc trích dẫn chưa giải quyết.
- Chưa có mẫu trường/khoa/giảng viên: dùng cấu trúc báo cáo học thuật nhất quán làm mặc định, chưa tuyên bố tuân thủ mẫu riêng hoặc IEEE.

## 2. Notebook
- Các notebook khám phá dữ liệu, SVM/UMAP, InceptionTime, TST và tổng hợp đánh giá; có thể gộp khi ngắn gọn và dễ chạy.
- Code tối giản, giải thích mục đích từng bước bằng tiếng Việt; không chèn phân công nhóm hay kế hoạch tuần.
- **Giải trình con số & API:** Mọi siêu tham số và lựa chọn hàm/API trong notebook đều phải được giải thích rõ căn cứ (tính toán lý thuyết/đặc thù miền hoặc log thực nghiệm thu hẹp).
- **Trực quan hóa luồng & lý thuyết:** Tích hợp hệ thống sơ đồ chuyên dụng (Mermaid hoặc render từ module đồ họa `src/diagrams.py`) trước mỗi block mô hình/pipeline phức tạp để trình bày nhanh lý thuyết cốt lõi và luồng biến đổi tensor.
- Chạy theo thứ tự trong runtime được chỉ định; ghi đường dẫn dữ liệu và cách dùng cache.
- Có biểu đồ tín hiệu, UMAP 2D, confusion matrix và bảng Accuracy, Macro Precision/Recall/F1.
- Dùng chung mã nguồn với project để tránh lệch preprocessing giữa notebook, train và demo.

## 3. Mã nguồn và artifact
- Loader DSADS, kiểm tra dữ liệu, metadata subject/activity/segment, split, preprocessing và feature extraction.
- Train/evaluate ba họ mô hình; pipeline SVM có/không UMAP; lưu và tải lại model.
- Có cấu hình, seed, phụ thuộc theo môi trường, hướng dẫn chạy và log kết quả.
- Artifact bao gồm scaler, bộ trích xuất đặc trưng/cấu hình, UMAP, nhãn hoạt động, checkpoint và model xuất nếu dùng.
- Cache có thông tin nguồn và cấu hình để tránh sử dụng nhầm phiên bản.
- Xác minh prediction sau khi tải lại; nếu xuất ONNX phải đối chiếu với model gốc.
- Kiểm tra subject không giao nhau và preprocessing chỉ học trên train.

## 4. Ứng dụng web demo — bắt buộc
1. Chọn một mẫu có sẵn hoặc tải một mẫu cảm biến; công bố định dạng và kiểm tra shape/giá trị trước khi dự đoán.
2. Hiển thị tín hiệu cảm biến theo thời gian; cho chọn kênh khi cần.
3. Cho chọn SVM, InceptionTime hoặc TST và chạy inference bằng model đã huấn luyện.
4. Hiển thị nhãn dự đoán và độ tin cậy; giải thích cách tính. Điểm decision function SVM không được gắn nhãn xác suất. Nếu hiệu chỉnh xác suất, dùng dữ liệu trong train, giữ tách theo subject và không dùng test để hiệu chỉnh.
5. So sánh dự đoán và độ tin cậy của cả ba mô hình trên cùng mẫu; có thể bổ sung độ trễ.
6. Hiển thị không gian đặc trưng UMAP 2D và vị trí mẫu đang xét; bộ UMAP 2D phục vụ hiển thị có thể khác bộ giảm chiều dùng phân loại, phải ghi rõ. Fit từ train và transform mẫu mới.
7. Tải đúng preprocessing và mapping nhãn đã lưu; không huấn luyện lại mỗi lần người dùng dự đoán.
8. Có hướng dẫn chạy local và mẫu đầu vào; báo lỗi rõ khi thiếu model hoặc dữ liệu sai định dạng.

Demo phải chạy với dữ liệu và model thật trước khi được đánh dấu hoàn thành. Chưa chốt framework hoặc hosting; ứng dụng web chạy local đáp ứng phạm vi ban đầu.

## 5. Điều kiện bàn giao chung
- README chỉ rõ cách chạy notebook, train/evaluate, demo và biên dịch báo cáo.
- Báo cáo, notebook, mã nguồn và demo dùng cùng bộ nhãn và quy trình dữ liệu.
- Kết quả có thể truy về model, split và cấu hình đã tạo ra chúng.
- Chạy được quy trình được công bố trên môi trường đã ghi; nêu rõ phần nào cần Colab GPU.
