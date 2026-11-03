# Phân Loại Hoạt Động Con Người Từ Cảm Biến Quán Tính (DSADS)
### Đồ Án Môn Học: Khai Phá Dữ Liệu (Data Mining)
Khảo sát kỹ thuật giảm chiều UMAP và so sánh ba họ mô hình: **Machine Learning (SVM)**, **Convolutional Neural Networks (InceptionTime)**, và **Transformer (Time Series Transformer - TST)** trên tập dữ liệu chuỗi thời gian cảm biến DSADS.

---

## 📌 1. Giới thiệu Đề tài & Bài toán

### 1.1. Bài toán (Problem Statement)
Nhận diện hoạt động con người (Human Activity Recognition - HAR) từ dữ liệu chuỗi thời gian đa kênh thu thập bởi các cảm biến quán tính (Inertial Measurement Units - IMU) gắn tại 5 vị trí trên cơ thể: **Thân người (Torso)**, **Cánh tay phải (Right Arm)**, **Cánh tay trái (Left Arm)**, **Chân phải (Right Leg)**, và **Chân trái (Left Leg)**.

### 1.2. Tập dữ liệu DSADS (Daily and Sports Activities Dataset)
- **Nguồn**: [UCI Machine Learning Repository - DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities).
- **Quy mô**:
  - **8 đối tượng** (`p1` đến `p8`).
  - **19 hoạt động** (`a01` đến `a19`), bao gồm các hoạt động tĩnh (ngồi, đứng, nằm), hoạt động chuyển tiếp (thang máy), đi lại/chạy bộ và các hoạt động thể thao (chèo thuyền, đạp xe, nhảy bật cóc, bóng rổ).
  - **60 phân đoạn** cho mỗi hoạt động (mỗi phân đoạn kéo dài 5 giây tại tần số 25 Hz $\rightarrow$ **125 bước thời gian**).
  - **45 kênh cảm biến** (5 vị trí $\times$ [3 trục gia tốc kế + 3 trục con quay hồi chuyển + 3 trục từ kế]).
  - Tổng cộng: **$9.120$ mẫu chuẩn hóa** ($125 \times 45$), $51.300.000$ điểm dữ liệu sạch $100\%$, không chứa giá trị NaN hay Inf.

### 1.3. Ba họ mô hình khảo sát
1. **Machine Learning Baseline**: **Support Vector Machine (SVM)** với kernel RBF trên vector đặc trưng thống kê 315 chiều (mean, std, min, max, median, skewness, kurtosis của 45 kênh). Khảo sát kết hợp kỹ thuật giảm chiều phi tuyến **UMAP (20 chiều)**.
2. **Convolutional Neural Networks (CNN)**: **InceptionTime** (tích chập 1D đa tỷ lệ với các kernel 9, 19, 39 và đường tắt thặng dư Residual Shortcut) tiếp nhận trực tiếp tensor $(45, 125)$.
3. **Transformer**: **Time Series Transformer (TST)** (Multi-Head Self-Attention với Sinusoidal Positional Encoding) tiếp nhận chuỗi token thời gian $(125, 45)$.

---

## 📊 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm

> **Quy chuẩn thực nghiệm chống rò rỉ dữ liệu (Leakage-free)**:
> Phân chia theo đối tượng (Subject-wise Split): **Train (`p1`–`p5`)** ($5.700$ mẫu), **Validation (`p6`)** ($1.140$ mẫu), **Test (`p7`–`p8`)** ($2.280$ mẫu).
> Mọi bộ chuẩn hóa (`StandardScaler`, `UMAP`, `channel-wise standardization`) đều chỉ fit duy nhất trên tập Train.

| Họ mô hình | Pipeline / Thuật toán | Dạng đầu vào | Số tham số | Val Accuracy | Val Macro-F1 | Test Accuracy | Đặc tính & Tốc độ |
|---|---|---|---|:---:|:---:|:---:|---|
| **Machine Learning** | **SVM Baseline (RBF)** | Thống kê 315D | ~45.000 SVs | **88.68%** | **0.8586** | 81.80% | $C=3.0, \gamma=0.0015$, train ~10s |
| **Machine Learning** | **UMAP (20D) + SVM** | Chiếu phi tuyến 20D | ~25.000 SVs | **67.81%** | **0.6608** | 71.58% | Giảm 94% số chiều, tốc độ suy luận nhanh |
| **Deep Learning (CNN)** | **InceptionTime** | Chuỗi $(45, 125)$ | 465.491 | **94.74%** | **0.9468** | ~92.5% | CNN 1D đa tỷ lệ, hội tụ mượt mà |
| **Deep Learning (Trans)**| **Time Series Transformer** | Chuỗi $(125, 45)$ | 75.283 | **89.21%** | **0.8875** | ~87.1% | 4 Attention heads, nắm bắt tương quan toàn cục |

---

## 🏗️ 3. Cấu Trúc Thư Mục Dự Án

```text
data-mining-project/
├── README.md                          # Tài liệu tổng quan dự án và hướng dẫn sử dụng
├── requirements.txt                   # Danh mục thư viện Python chuẩn hóa
├── config.json                        # Cấu hình đường dẫn và siêu tham số
├── AGENTS.md                          # Quy chuẩn làm việc và bảo mật
│
├── data/                              # Dữ liệu dự án (được loại trừ trên git)
│   ├── raw/                           # Dữ liệu gốc DSADS (19 thư mục a01-a19)
│   └── processed/
│       └── dsads_cache.npz            # Cache nén nạp 9.120 mẫu trong 1 giây
│
├── notebooks/                         # Sổ tay Jupyter thí nghiệm (Pre-rendered outputs)
│   ├── 01_dsads_exploration.ipynb     # Khám phá EDA, dạng sóng tín hiệu, phân tích dị thường số 0
│   ├── 02_svm_umap_baseline.ipynb     # Trích xuất 315D, GroupKFold CV, SVM Baseline, UMAP 2D
│   └── 03_deep_learning_models.ipynb  # Huấn luyện InceptionTime & TST, Loss/F1 curves, Confusion Matrix
│
├── src/                               # Mã nguồn tái sử dụng chất lượng cao
│   ├── dsads.py                       # Bộ nạp dữ liệu chuẩn hóa, metadata, split, kiểm tra toàn vẹn
│   ├── features.py                    # Trích xuất 315 đặc trưng thống kê miền thời gian
│   └── diagrams.py                    # Module vẽ sơ đồ kiến trúc luồng dữ liệu chuẩn học thuật
│
├── models/                            # Checkpoints mô hình đã huấn luyện sẵn
│   ├── scaler.joblib                  # StandardScaler 315D (fit trên p1-p5)
│   ├── svm_baseline.joblib            # Trọng số SVM Baseline
│   ├── svm_umap.joblib                # Trọng số SVM trên không gian UMAP
│   ├── umap_reducer.joblib            # UMAP Reducer 315D -> 20D
│   ├── umap_2d.joblib                 # UMAP Visualizer 315D -> 2D
│   ├── channel_scaler.npz             # Channel-wise mean & std cho Deep Learning
│   ├── inception_time.pt              # Trọng số PyTorch InceptionTime
│   └── transformer_tst.pt             # Trọng số PyTorch Time Series Transformer
│
├── reports/                           # Kết quả định lượng, biểu đồ và báo cáo
│   ├── PROGRESS_REPORT.md             # Báo cáo tiến độ chi tiết phục vụ họp nhóm & báo cáo thầy
│   ├── svm_results.json               # Số liệu thực nghiệm SVM
│   ├── dl_results.json                # Số liệu thực nghiệm Deep Learning
│   ├── model_comparison.json          # Bảng tổng hợp đối chiếu cả 4 pipeline
│   └── figures/                       # Sơ đồ kiến trúc độ phân giải cao
│       ├── diagram_overall_pipeline.png
│       ├── diagram_svm_umap_pipeline.png
│       ├── diagram_inception_architecture.png
│       └── diagram_transformer_architecture.png
│
├── demo/                              # Ứng dụng Web Demo tương tác (Streamlit)
│   └── app.py                         # Trực quan hóa sóng, dự đoán 3 họ model, UMAP 2D
│
└── tests/                             # Bộ kiểm thử tự động (Unit Tests)
    ├── test_dsads.py                  # Kiểm tra tính toàn vẹn loader, shape, subject split
    ├── test_features.py               # Kiểm tra kích thước và công thức 315 đặc trưng
    └── test_diagrams.py               # Kiểm tra module sinh hình vẽ sơ đồ
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### Bước 1: Khởi tạo Môi trường ảo Python
Khuyến nghị sử dụng **Python 3.10 – 3.12**:
```bash
# Clone repository
git clone <URL_REPOSITORY_CUA_NHOM>
cd data-mining-project

# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate  # Trên macOS/Linux
# .venv\Scripts\activate   # Trên Windows

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit  # Dành cho Web Demo
```

### Bước 2: Chạy Bộ Kiểm thử Tự động (Unit Tests)
Đảm bảo tất cả 9 tests đều vượt qua (100% Pass):
```bash
pytest tests/ -v
```

### Bước 3: Mở & Chạy các Notebooks Thí nghiệm
Toàn bộ 3 notebooks trong thư mục `notebooks/` đều đã được lưu trữ sẵn **kết quả thực thi (pre-rendered outputs)** kèm biểu đồ, bảng biểu và ma trận nhầm lẫn:
- **Notebook 01**: `notebooks/01_dsads_exploration.ipynb`
- **Notebook 02**: `notebooks/02_svm_umap_baseline.ipynb`
- **Notebook 03**: `notebooks/03_deep_learning_models.ipynb`

### Bước 4: Khởi động Ứng dụng Web Demo
Khởi chạy ứng dụng Web trực quan hóa và suy luận tương tác:
```bash
streamlit run demo/app.py
```
*Các tính năng của Web Demo:*
- Chọn mẫu thử từ tập Test (`p7`–`p8`) hoặc tải lên tệp cảm biến `.txt` kích thước $(125, 45)$.
- Xem biểu đồ dao động 3 trục gia tốc kế theo thời gian tại từng vị trí cơ thể.
- Dự đoán hoạt động và độ tin cậy từ 3 họ mô hình (SVM, InceptionTime, TST).
- Hiển thị tọa độ mẫu trên không gian đặc trưng UMAP 2D.

---

## 👥 5. Phân Công Công Việc & Kế Hoạch Tiếp Theo Cho Nhóm

| Thành viên / Nhóm nhỏ | Nhiệm vụ trọng tâm | Thời hạn | Ghi chú & Tài liệu liên quan |
|---|---|:---:|---|
| **Nhánh 1: Web Demo** | Tinh chỉnh giao diện Streamlit, kết nối trọn vẹn checkpoint PyTorch và biểu đồ Plotly tương tác | Tuần 2 | Tham khảo `demo/app.py` và `docs/DELIVERABLES.md` (Mục 4) |
| **Nhánh 2: Báo cáo LaTeX** | Soạn thảo báo cáo học thuật bằng LaTeX (`main.tex`), nhúng bảng kết quả và sơ đồ | Tuần 2 | Lấy số liệu từ `reports/PROGRESS_REPORT.md` và `figures/` |
| **Nhánh 3: Tối ưu & Mở rộng** | Thử nghiệm đánh giá bổ sung (LOSO - Leave-One-Subject-Out), phân tích lỗi Confusion Matrix | Tuần 3 | Xem phân tích dị thường trong Notebook 03 |
