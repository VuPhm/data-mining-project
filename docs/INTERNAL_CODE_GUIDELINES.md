# Ghi chú nội bộ về cách viết code

Tài liệu làm việc dành cho người phát triển và công cụ hỗ trợ code; không đưa vào báo cáo, notebook nộp, giao diện demo hoặc tài liệu bàn giao công khai.

- Ưu tiên code tinh giản, dễ nắm bắt và dễ giải thích lại từng bước.
- Dùng tên biến, hàm và cấu hình rõ nghĩa, nhất quán; thống nhất cách gọi dữ liệu, nhãn, subject và các tập train/validation/test.
- Giữ luồng xử lý trực tiếp; tách hàm khi giúp hiểu mục đích hoặc tái sử dụng thực tế.
- Chọn cách triển khai thông dụng, dễ đọc; tránh abstraction, mẹo rút gọn hoặc tối ưu nội bộ cực đoan làm khó giải thích.
- Chỉ tối ưu khi có nhu cầu thực tế hoặc đo được điểm nghẽn; giữ giải pháp dễ theo dõi.
- Comment ngắn để giải thích mục đích, lý do và shape dữ liệu khi cần; tránh lặp lại điều code đã thể hiện.
- **Quy tắc giải thích con số, thuật toán và thuộc tính API (Yêu cầu giảng viên):**
  - Mọi con số/siêu tham số (window size, batch size, epochs, lr, kernel sizes, channels, $C$, $\gamma$, số chiều UMAP, dropout rate, weight decay, v.v.), lựa chọn thuật toán và **mọi attribute/tham số truyền vào khi gọi API** (ví dụ: `kernel='rbf'`, `decision_function_shape='ovr'`, `bias=False` trước BatchNorm, `padding='same'`, `batch_first=True`, `betas=(0.9, 0.999)`, `drop_last=False`, v.v.) BẮT BUỘC phải có nguyên nhân rõ ràng.
  - Căn cứ giải thích phải xuất phát từ:
    + (1) **Tính toán cụ thể từ lý thuyết miền/toán học** (ví dụ: tần số $25\text{ Hz} \times 5\text{s} = 125$ steps; kernel size 9, 19, 39 bao quát các pha bước chân và chu kỳ cử động $0.36\text{s} - 1.56\text{s}$; bottleneck giảm channel tránh bùng nổ tham số; `bias=False` do BatchNorm có phép trừ mean làm triệt tiêu bias; $d\_model=64$ và 4 heads; công thức gamma $\approx 1/(\text{n\_features} \times \text{var})$).
    + (2) **Lộ trình rà số tối ưu (Hyperparameter Tuning Roadmap)**: Mô tả phương pháp luận rà soát từ dải rộng (coarse sweep trên thang logarit) đến tinh chỉnh cục bộ (fine sweep quanh vùng tối ưu tiềm năng) qua kiểm chứng chéo theo đối tượng (GroupKFold CV trên tập Train) với tiêu chí Macro-F1.
- **Quy tắc giải thích lý do của các lựa chọn và các điểm dị thường (Anomalies):**
  - Mọi hiện tượng bất thường, dị thường hoặc kết quả đi ngược lại kỳ vọng thông thường đều phải được phân tích cặn kẽ về nguyên nhân toán học, vật lý và cơ sinh học:
    + *Dị thường UMAP + SVM sụt giảm mạnh hiệu năng*: Bản chất không giám sát (unsupervised) của UMAP bảo toàn topo đồ thị mờ không có nhãn, làm biến dạng ranh giới lớp giữa các hoạt động gần nhau; mất mát 94% số chiều làm mất các đặc trưng xung cực trị quan trọng cho SVM.
    + *Dị thường phương sai lớn giữa các Fold trong GroupKFold CV*: Biến thiên liên đối tượng (Inter-subject variability: chiều cao, cân nặng, thói quen vận động cá nhân).
    + *Dị thường nhầm lẫn giữa các cặp hoạt động trong Confusion Matrix*: Các hoạt động tĩnh (ngồi vs đứng, nằm ngửa vs nằm nghiêng) có vector trọng lực và biên độ tương đồng; các hoạt động đi bộ trên dốc vs phẳng có cùng tần số bước chân.
    + *Dị thường InceptionTime hội tụ nhanh hơn TST*: Inductive bias của CNN 1D (tính cục bộ thời gian và bất biến dịch chuyển) phù hợp với chuỗi rung động và cỡ mẫu vừa phải ($5.700$ mẫu), trong khi Transformer thiếu inductive bias cục bộ và rất đói dữ liệu (data-hungry).
- **Hệ thống sơ đồ, biểu đồ chuyên dụng (Yêu cầu đồ án):**
  - Xây dựng và duy trì module chuyên biệt `src/diagrams.py` để sinh các sơ đồ kiến trúc và luồng xử lý chuẩn học thuật.
  - Trước mỗi notebook hoặc trước block code thực thi mô hình/kỹ thuật phức tạp, phải có sơ đồ luồng dữ liệu (tensor shape qua từng tầng) và tóm tắt nhanh lý thuyết cốt lõi bằng Markdown + sơ đồ (Mermaid hoặc render từ matplotlib) để người đọc/giảng viên nắm bắt trực quan tức thì.
- Sự tinh giản không được làm sai thuật toán, mất kiểm tra dữ liệu cần thiết hoặc gây rò rỉ dữ liệu.
- Không hiển thị yêu cầu nội bộ này trong sản phẩm công khai; thể hiện bằng chất lượng và cấu trúc code.
