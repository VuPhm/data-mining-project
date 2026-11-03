# Ngữ cảnh đề tài DSADS
Cập nhật: 28/10/2026. Tổng hợp từ project ChatGPT “học và làm đồ án data mining”.

## Đề tài đã đăng ký
Nhận diện hoạt động con người từ dữ liệu cảm biến; khảo sát giảm chiều và so sánh Machine Learning, CNN, Transformer.
- Dữ liệu: Daily and Sports Activities Dataset (DSADS), UCI.
- Theo bản đăng ký: 9.120 mẫu, 19 hoạt động, 8 người; mỗi mẫu 125 bước thời gian × 45 kênh.
- Ba họ mô hình: SVM (baseline và nhánh UMAP + SVM), InceptionTime, Time Series Transformer (TST).
- SVM và SVM + UMAP là hai pipeline thí nghiệm, không phải hai họ mô hình khác nhau.
- Gợi ý XGBoost xuất hiện trong lịch sử nhưng chưa được người dùng chốt thay SVM. Bản đăng ký người dùng gửi sau đó vẫn ghi SVM.
- Demo dự kiến: chọn/tải một mẫu cảm biến, xem tín hiệu, chọn mô hình, dự đoán và so sánh; hiển thị độ tin cậy khi có cách tính phù hợp.

## Quy trình thực nghiệm
1. Đọc dữ liệu kèm activity, subject, segment; kiểm tra kích thước, NaN/Inf và file lỗi.
2. Chia theo người: train p1–p5, validation p6, test p7–p8 (phương án trong notebook cũ, cần đối chiếu notebook gốc khi khôi phục).
3. Chỉ fit scaler và UMAP trên train. Với GroupKFold, fit lại preprocessing trong từng fold.
4. SVM dùng đặc trưng thống kê; InceptionTime/TST giữ cấu trúc chuỗi thời gian.
5. Macro-F1 là chỉ số chính; thêm Accuracy, Macro Precision/Recall, confusion matrix, thời gian train/inference và phần cứng.
6. Tuning trên train/validation; không chạy lại test để chọn cấu hình. Test baseline đã từng được xem trong chat trước, cần công khai khi báo cáo.
7. Dùng cùng split cho các mô hình. LOSO là hướng đánh giá thêm, chưa có bằng chứng hoàn tất.

## Kết quả được ghi trong chat “Tối ưu notebook vòng 1”
Đây là số liệu được trợ lý trước báo cáo, CHƯA được tái lập hoặc kiểm chứng bằng artifact ở máy hiện tại.

| Pipeline | Validation Accuracy | Validation Macro-F1 | Test Accuracy đã báo cáo |
|---|---:|---:|---:|
| SVM RBF ban đầu | 92,63% | 0,9237 | 81,80% |
| SVM + UMAP | 85,00% | 0,8481 | 71,58% |
| SVM tối ưu vòng 1 | 95,35% | 0,9529 | Chưa báo cáo |

Cấu hình được báo cáo: RBF, C=3, gamma=0.0015; 9 cấu hình × 5 fold GroupKFold; CV Macro-F1 trung bình 0,8947, độ lệch chuẩn 0,0375.
Không xem các số liệu này là kết quả local, không tự bổ sung chi tiết feature extraction chưa đọc được.

## Yêu cầu cách viết
- Notebook cá nhân sạch, ngắn, dễ giải thích code khi thầy review.
- Không đưa phân công nhóm, kế hoạch tuần và nội dung thảo luận vào notebook.
- Biểu đồ chính: tín hiệu theo thời gian, UMAP 2D, confusion matrix; chỉ ghi kết quả thực khi đã chạy.
- Quy tắc nhóm ở tài liệu riêng: mỗi người học và làm toàn bộ ba họ mô hình chung; mở rộng tùy chọn sau đó.

## Runtime đã trao đổi
MacBook Pro 2019; hướng dự kiến là VS Code local cho EDA/SVM, Colab GPU cho InceptionTime/TST.
Chia sẻ dữ liệu đã xử lý, split, checkpoint và kết quả qua file; biến RAM không tự chia sẻ.
Xuất ONNX để demo local là phương án cần thử và kiểm chứng tương thích trước.
Chưa xác minh lại extension, phiên bản Python hay thư viện phù hợp với Mac này. Không cài deep learning theo suy đoán.

## Nguồn và artifact cần khôi phục
- [Hướng dẫn Mining DSADS](https://chatgpt.com/c/6adeda68-b464-83ec-a438-c80b08c8e8c9): bản đăng ký do người dùng gửi và yêu cầu notebook tối giản.
- [Tối ưu notebook vòng 1](https://chatgpt.com/c/6ae19eac-90ec-83ec-9168-ea9dbfb93a36): số liệu và cấu hình SVM vòng 1.
- [Cân nhắc chạy local VS Code](https://chatgpt.com/c/6ae1a2b4-8df0-83ec-9b15-5bb12f2346d4): hướng local/Colab/cache/inference.
- [Thiết lập Colab](https://chatgpt.com/c/6adf1deb-10e8-83ec-823f-1dfc94d39852): quy tắc thực nghiệm và lộ trình chung.
- [Tìm dataset datamining_verified](https://chatgpt.com/c/6aacae7a-e684-83ec-b0dd-b80956675ff3): lịch sử feedback nhóm 35; có đề xuất cũ khác bản hiện tại.
- [Notebook Colab người dùng cung cấp](https://colab.research.google.com/drive/1S0w9YNL9346PztEyJTW8yHPehz-6FZ4v?usp=sharing).
- [Notebook tối ưu trong Library](https://chatgpt.com/library?file_id=libfile_8c2ec53469348191955d7585103623c8&account_id=personal).
- [Dataset UCI](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities).

Các link artifact được khôi phục từ nội dung chat, chưa mở hoặc tải được tệp trong phiên này. Đường dẫn sandbox của phiên ChatGPT cũ không phải tệp trên máy Mac.

## Đầu ra đã xác nhận trực tiếp ngày 28/10/2026
Bản đăng ký nguyên văn: REGISTERED_TOPIC.md. Tiêu chí chi tiết: DELIVERABLES.md.
Bắt buộc: nguồn báo cáo LaTeX và PDF, notebook, mã nguồn liên quan và ứng dụng web demo. Demo gồm chọn/tải mẫu, hiển thị tín hiệu, chọn mô hình, dự đoán kèm độ tin cậy, so sánh SVM/InceptionTime/TST và trực quan hóa UMAP. Chưa có mẫu LaTeX riêng của trường/khoa.
