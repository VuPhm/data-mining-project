# Lộ trình triển khai
## 1. Khôi phục đầu vào
- Lấy notebook Exploration và Optimization Round1 từ nguồn trong PROJECT_CONTEXT.md.
- Giữ bản gốc riêng, xác nhận chính xác feature extraction, split và cấu hình.
- Lấy DSADS gốc; kiểm tra mẫu, nhãn và subject.
Điều kiện hoàn thành: có notebook và dữ liệu thật tại local, biết nguồn.

## 2. Môi trường và dữ liệu chung
- Kiểm tra kiến trúc máy, Python và khả năng cài các gói; chọn phiên bản phù hợp.
- Tách phụ thuộc local ML khỏi deep learning trên Colab khi cần.
- Tạo loader, cache float32, metadata và split cố định; lưu cấu hình/phiên bản cùng cache.
Điều kiện hoàn thành: đọc lại cache được và xác nhận các subject không giao nhau.

## 3. Tái lập SVM
- Chạy lại baseline từ notebook gốc.
- So sánh SVM và UMAP + SVM trên cùng validation.
- Thử cấu hình C=3, gamma=0.0015 sau khi đối chiếu đặc trưng.
- GroupKFold theo subject trong train; giữ scaler/UMAP trong pipeline của từng fold.
Điều kiện hoàn thành: có kết quả thật, cấu hình và thời gian thực thi; giải thích sai khác với số liệu chat.

## 4. CNN và Transformer
- Kiểm tra implementation InceptionTime/TST và shape input rõ ràng.
- Chạy thử một batch, lưu/tải checkpoint, thử export ONNX sớm.
- Train Colab với cùng split, normalization học từ train và early stopping theo validation.
Điều kiện hoàn thành: checkpoint và bảng validation có thể tái lập.

## 5. Đánh giá và demo
- Chốt mô hình trước khi đánh giá test.
- Báo cáo việc test baseline đã được xem trong quá khứ.
- So sánh metric, confusion matrix, thời gian, hạn chế.
- Tạo demo sau khi có model thật và xác minh dự đoán khi tải lại.

## 6. Hoàn thiện web demo
- Thực hiện đủ chức năng tại DELIVERABLES.md mục 4; dùng model và preprocessing đã lưu.
- Kiểm tra chọn mẫu/tải mẫu, dữ liệu lỗi, từng model, so sánh ba model và UMAP 2D.
Điều kiện hoàn thành: chạy local bằng hướng dẫn README, kết quả khớp pipeline đánh giá và giải thích được độ tin cậy.

## 7. Báo cáo LaTeX và bộ bàn giao
- Chuẩn bị nội dung phương pháp cùng quá trình triển khai; điền kết quả khi đã tái lập thực nghiệm.
- Tạo nguồn LaTeX, bibliography, hình/bảng từ kết quả project; mô tả và minh họa demo.
- Biên dịch và kiểm tra trực quan PDF; hoàn tất hướng dẫn chạy cho code/notebook/demo và hướng dẫn biên dịch.
Điều kiện hoàn thành: đáp ứng DELIVERABLES.md, số liệu thống nhất và có nguồn kiểm chứng.
