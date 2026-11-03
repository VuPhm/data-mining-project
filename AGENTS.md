# Hướng dẫn làm việc
- Đọc docs/PROJECT_CONTEXT.md và docs/HANDOFF.md trước khi thực hiện công việc.
- Phạm vi hiện tại: DSADS; SVM và UMAP + SVM, InceptionTime, TST. Không tự đổi SVM sang XGBoost vì đề xuất cũ.
- Notebook cá nhân ngắn, rõ, comment tiếng Việt vừa đủ để giải thích. Giữ nội dung quản lý ngoài notebook.
- Chia theo subject; chỉ fit preprocessing trên train và trong từng fold CV.
- Không dùng test để tuning; ghi rõ test baseline đã từng được xem.
- Không ghi số liệu lịch sử hoặc dữ liệu giả là kết quả chạy local.
- Ưu tiên khôi phục notebook có sẵn trước khi viết lại pipeline.
- Xác minh tương thích Python và gói trước khi cài; môi trường local và Colab có thể khác nhau.
- Không sửa cùng tệp đồng thời bằng nhiều công cụ. Cập nhật HANDOFF sau mỗi mốc.
- Không tự gọi agent, dịch vụ hoặc huấn luyện dài chỉ để đọc ngữ cảnh.

## Phạm vi đầu ra đã chốt
- Đọc docs/REGISTERED_TOPIC.md và docs/DELIVERABLES.md trước khi triển khai. Bản đăng ký người dùng cung cấp trực tiếp là nguồn phạm vi ưu tiên.
- Bắt buộc bàn giao báo cáo LaTeX chuẩn học thuật kèm PDF, notebook, code và web demo đủ chức năng; demo không phải phần tùy chọn.
- Chưa có mẫu báo cáo riêng của trường/khoa; không tự tuyên bố đáp ứng mẫu chưa được cung cấp.
- Báo cáo, notebook và demo phải thống nhất số liệu, preprocessing và nhãn; không dùng dữ liệu giả làm kết quả thật.

## Quy ước code nội bộ
- Áp dụng docs/INTERNAL_CODE_GUIDELINES.md: code tinh giản, tên biến nhất quán, dễ hiểu và dễ giải thích; tránh tối ưu nội bộ cực đoan.
- **Giải thích con số và API:** Mọi con số (siêu tham số, kích thước ma trận, window, threshold) và lựa chọn hàm/API đều phải có nguyên nhân rõ ràng (tính toán lý thuyết miền hoặc kết quả thử nghiệm thu hẹp có ghi nhận); thể hiện trong markdown hoặc comment.
- **Hệ thống sơ đồ, biểu đồ chuyên dụng:** Xây dựng module vẽ sơ đồ luồng/kiến trúc chuyên dụng (`src/diagrams.py`) và sơ đồ Mermaid để trực quan hóa nhanh lý thuyết cốt lõi trước mỗi notebook hoặc block code kế tiếp.
- Ghi chú này chỉ phục vụ triển khai, không đưa vào báo cáo, notebook nộp, demo hoặc tài liệu công khai.
