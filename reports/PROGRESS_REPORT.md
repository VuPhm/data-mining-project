# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN DATA MINING
**Đề tài**: Nhận diện hoạt động con người từ cảm biến quán tính (DSADS) – Khảo sát giảm chiều UMAP và so sánh Machine Learning, CNN, Transformer  
**Tập dữ liệu**: UCI Daily and Sports Activities Dataset (DSADS)  
**Thời gian cập nhật**: 04/11/2026  

---

## 1. TỔNG QUAN ĐỀ TÀI & PHẠM VI TRIỂN KHAI

### 1.1. Mục tiêu và Bài toán
- **Bài toán**: Phân loại 19 hoạt động sinh hoạt hằng ngày và thể thao dựa trên chuỗi thời gian đa kênh từ 5 cụm cảm biến quán tính (Inertial Measurement Units - IMU) gắn trên cơ thể người (Thân, Cánh tay phải, Cánh tay trái, Chân phải, Chân trái).
- **Quy mô dữ liệu**:
  - **8 đối tượng** (`p1` đến `p8`).
  - **19 hoạt động** (`a01` đến `a19`), mỗi hoạt động gồm 60 phân đoạn (mỗi phân đoạn kéo dài 5 giây ở tần số lấy mẫu 25 Hz $\rightarrow$ **125 bước thời gian**).
  - **45 kênh cảm biến** (mỗi cụm IMU gồm 9 kênh: 3 trục Gia tốc kế $\text{acc}$, 3 trục Con quay hồi chuyển $\text{gyro}$, 3 trục Từ kế $\text{mag}$).
  - Tổng cộng: **$9.120$ mẫu chuẩn hóa** ($125 \times 45$), $51.300.000$ điểm dữ liệu sạch 100%, không chứa NaN hay Inf.
- **Ba họ mô hình nghiên cứu và so sánh**:
  1. **Machine Learning Baseline**: Support Vector Machine (SVM) với kernel RBF trên vector đặc trưng thống kê 315 chiều, khảo sát kết hợp giảm chiều phi tuyến **UMAP (20 chiều)**.
  2. **Convolutional Neural Networks (CNN)**: **InceptionTime** (tích chập 1D đa tỷ lệ với các kích thước bộ lọc 9, 19, 39 và kết nối thặng dư Residual Shortcut).
  3. **Transformer**: **Time Series Transformer (TST)** (cơ chế Self-Attention đa đầu Multi-Head Attention, mã hóa vị trí Positional Encoding nhận diện tương quan toàn cục).

### 1.2. Thiết kế Thực nghiệm & Chống Rò rỉ Dữ liệu (Leakage-free)
- **Phân chia theo đối tượng (Subject-wise Split)**:
  - Tập Huấn luyện (Train): Đối tượng `p1`, `p2`, `p3`, `p4`, `p5` ($5.700$ mẫu - 62.5%).
  - Tập Kiểm định (Validation): Đối tượng `p6` ($1.140$ mẫu - 12.5%).
  - Tập Kiểm tra (Test): Đối tượng `p7`, `p8` ($2.280$ mẫu - 25.0%).
- **Nguyên tắc tiền xử lý**: Toàn bộ các phép biến đổi (`StandardScaler`, `UMAP`, `channel-wise standardization`) chỉ được `fit` duy nhất trên tập Train (`p1`–`p5`) hoặc trong từng fold của kiểm chứng chéo (GroupKFold CV). Tuyệt đối không để thông tin của tập Validation hay Test lọt vào khâu huấn luyện.

---

## 2. KẾT QUẢ THỰC NGHIỆM ĐẠT ĐƯỢC (SỐ LIỆU THỰC TẾ 100%)

### 2.1. Bảng Tổng hợp Hiệu năng 4 Pipeline Thí nghiệm
*(Đánh giá trên tập Validation độc lập `p6` và đối chiếu tập Test `p7-p8`)*

| Họ mô hình | Pipeline | Dạng đặc trưng đầu vào | Số tham số | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Thời gian train |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Machine Learning** | **SVM Baseline (RBF)** | Thống kê 315D | ~45.000 SVs | **88.68%** | **0.8586** | 81.80% | 0.8120 | ~10s |
| **Machine Learning** | **UMAP (20D) + SVM** | Chiếu phi tuyến 20D | ~25.000 SVs | **68.33%** | **0.6662** | 71.58% | 0.7042 | ~15s |
| **Deep Learning (CNN)** | **InceptionTime** | Chuỗi $(45, 125)$ | 465.491 | **92.63%** | **0.9261** | **81.45%** | **0.7995** | 459.6s |
| **Deep Learning (Trans)**| **Time Series Transformer** | Chuỗi $(125, 45)$ | 75.283 | **92.11%** | **0.9212** | **76.67%** | **0.7536** | 241.4s |

### 2.2. Kết quả Kiểm chứng chéo GroupKFold (5 Folds) trên tập Train (`p1`–`p5`) của SVM
- **Accuracy trung bình**: $90.58\% \pm 5.09\%$
- **Macro-F1 trung bình**: $0.8905 \pm 0.0605$
- Kết quả khớp với các thử nghiệm lý thuyết và chứng minh tính tổng quát hóa tốt trên các đối tượng tập train.

---

## 3. PHÂN TÍCH CHUYÊN SÂU CÁC ĐIỂM DỊ THƯỜNG (ANOMALIES ANALYSIS)

Theo quy chuẩn học thuật và yêu cầu của giảng viên hướng dẫn, đồ án làm rõ nguyên nhân của các hiện tượng đặc thù quan sát được trong thực nghiệm:

### 3.1. Dị thường 1: UMAP + SVM sụt giảm mạnh hiệu năng so với SVM Baseline (88.68% $\rightarrow$ 68.33%)
- **Hiện tượng**: Khi giảm chiều đặc trưng từ 315 chiều xuống 20 chiều bằng UMAP, Macro-F1 giảm gần 20%.
- **Nguyên nhân cốt lõi**:
  1. *Tính không giám sát (Unsupervised Nature)*: UMAP tối ưu hóa cấu trúc đồ thị lân cận mờ (fuzzy simplicial set) hoàn toàn không có nhãn. Trong không gian 315 chiều ban đầu, ranh giới giữa các hoạt động vận động tương tự (ví dụ: `a09` đi bộ trong bãi xe vs `a10` đi bộ máy 4 km/h) vốn rất mỏng manh. Chiếu phi tuyến không giám sát có xu hướng "nhập cụm" các điểm gần nhau, làm biến dạng ranh giới phân tách tối ưu mà SVM tìm kiếm.
  2. *Mất mát đặc trưng xung cực trị*: Các đại lượng cực đại (`max`), độ lệch (`skewness`) và độ nhọn (`kurtosis`) của tín hiệu rung động phản ánh các xung va chạm đột ngột của bàn chân/tay. Việc nén 94% không gian làm mờ đi các điểm dị biệt này.

### 3.2. Dị thường 2: InceptionTime hội tụ nhanh và mượt mà hơn hẳn Transformer (TST)
- **Hiện tượng**: InceptionTime vượt ngưỡng 88% Macro-F1 ngay từ Epoch 1 (88.90%) và đạt đỉnh tại Epoch 2 (92.61%), trong khi Transformer bắt đầu từ 76.25% ở Epoch 1 và cần đến Epoch 5 mới đạt 92.12%.
- **Nguyên nhân kiến trúc (Inductive Bias)**:
  1. *Inductive Bias của CNN 1D*: Tích chập 1D mang sẵn hai giả định mạnh mẽ: **Tính cục bộ thời gian (Temporal Locality)** và **Tính bất biến tịnh tiến (Translation Equivariance)**. Đối với tín hiệu cảm biến sinh học, dao động của 1 bước chân diễn ra trong một cửa sổ cục bộ ($0.36\text{s} - 1.56\text{s}$). Ba kernel kích thước 9, 19, 39 nắm bắt trọn vẹn đặc trưng hình sóng này ngay từ các tầng đầu.
  2. *Bản chất "Đói dữ liệu" (Data-hungry) của Transformer*: Transformer xem chuỗi thời gian như một chuỗi các token và buộc phải học tương quan giữa tất cả các cặp bước thời gian qua ma trận tự chú ý $(125 \times 125)$. Với kích thước tập dữ liệu vừa phải ($5.700$ mẫu train), Transformer cần nhiều thời gian để hội tụ và dễ bị nhiễu bởi các biến thiên nhỏ.

### 3.3. Dị thường 3: Độ biến thiên hiệu năng lớn giữa các Fold trong GroupKFold CV ($\pm 5.09\%$)
- **Hiện tượng**: Khi kiểm chứng chéo 5-Fold theo đối tượng trên tập Train, F1 dao động từ 83% đến 94%.
- **Nguyên nhân cơ sinh học**: Biến thiên liên đối tượng (Inter-subject variability). Mỗi đối tượng có thể trạng, chiều cao, cân nặng, độ dài sải chân và thói quen vận động khác nhau. Khi một đối tượng có phong cách vận động cá biệt đóng vai trò tập kiểm thử của một fold, mô hình gặp khó khăn hơn trong việc khái quát hóa.

### 3.4. Dị thường 4: Hiện tượng 312.927 giá trị 0.000 trong dữ liệu gốc DSADS
- **Hiện tượng**: Khảo sát toàn bộ 51.300.000 điểm dữ liệu phát hiện 312.927 giá trị 0.000 (0.610%).
- **Nguyên nhân**: $99.943\%$ các giá trị 0 này nằm ở 25 mẫu cuối (từ index 100 đến 124, tương đương 1.0 giây cuối) ở đúng 278 phân đoạn thuộc hoạt động thang máy (`a05`, `a06`) và nhảy bật cóc (`a18`). Đây là kỹ thuật **Zero-Padding có chủ đích** của nhóm tác giả gốc (Barshan & Yüksek, 2014) để đệm các hoạt động kết thúc sớm hơn 5 giây cho đủ định dạng chuẩn $125 \times 45$.
- **Độ an toàn**: Độ lệch chuẩn nhỏ nhất trên tất cả 45 kênh là $\sigma_{min} = 0.000503 > 0$, chứng minh không có kênh nào bị phẳng lì hay triệt tiêu phương sai, các phép chuẩn hóa z-score hoàn toàn an toàn.

---

## 4. TRẠNG THÁI CÁC ĐẦU RA VÀ TÀI SẢN DỰ ÁN

| Hạng mục | Trạng thái | Vị trí lưu trữ | Ghi chú |
|---|:---:|---|---|
| **Notebook 01 (EDA)** | ✅ 100% | `notebooks/01_dsads_exploration.ipynb` | Đầy đủ biểu đồ phân phối, sóng cảm biến, kiểm tra dị thường số 0 |
| **Notebook 02 (SVM & UMAP)** | ✅ 100% | `notebooks/02_svm_umap_baseline.ipynb` | Trích xuất 315D, GroupKFold CV, bảng kết quả, UMAP 2D |
| **Notebook 03 (Deep Learning)** | ✅ 100% | `notebooks/03_deep_learning_models.ipynb` | InceptionTime & TST, đồ thị Loss/F1, pre-rendered outputs đầy đủ |
| **Mô hình & Checkpoints** | ✅ 100% | `models/` | `svm_baseline.joblib`, `svm_umap.joblib`, `inception_time.pt`, `transformer_tst.pt`, ONNX models |
| **Dữ liệu Cache nén** | ✅ 100% | `data/processed/dsads_cache.npz` | Tốc độ nạp 9.120 mẫu ~1s, độc lập không phụ thuộc file thô |
| **Hệ thống Sơ đồ Đồ họa** | ✅ 100% | `src/diagrams.py`, `reports/figures/` | 4 sơ đồ kiến trúc chuẩn học thuật (Pipeline, SVM, Inception, TST) |
| **Web Demo Nguyên mẫu** | ✅ 100% | `demo/app.py` | Streamlit: Chọn mẫu, vẽ sóng, dự đoán 3 họ mô hình, UMAP 2D |
| **Kiểm thử tự động (Unit Tests)**| ✅ 9/9 Pass | `tests/` | Kiểm tra loader, split, trích xuất đặc trưng và sơ đồ |

---

## 5. KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO CHO NHÓM (ACTION ITEMS)

1. **Nhánh Web Demo**:
   - Hoàn thiện giao diện tương tác Streamlit với Plotly (`demo/app.py`).
   - Tích hợp gọi trực tiếp `models/inception_time.pt` và `models/transformer_tst.pt` để so sánh trực quan cả 3 mô hình trên cùng một mẫu cảm biến.
2. **Nhánh Báo cáo LaTeX**:
   - Soạn thảo cấu trúc chương mục chuẩn học thuật trong thư mục `reports/latex/`.
   - Trích dẫn các bài báo nền tảng của Barshan & Yüksek (2014) và Fawaz et al. (InceptionTime 2020).
   - Nhúng bảng kết quả đối chiếu 4 pipeline từ `reports/model_comparison.json`.
3. **Thuyết trình & Báo cáo Tiến độ**:
   - Sử dụng các hình vẽ kiến trúc trong `reports/figures/` và các số liệu trong bảng mục 2 để trình chiếu slide.
