# Bàn giao công việc
Cập nhật 04/11/2026.

## Đọc trước
1. PROJECT_CONTEXT.md: đề tài, quyết định, nguồn và số liệu lịch sử.
2. IMPLEMENTATION_PLAN.md: thứ tự triển khai.
3. AGENTS.md ở gốc: nguyên tắc làm việc.
4. reports/PROGRESS_REPORT.md: báo cáo tiến độ chi tiết phục vụ báo cáo giảng viên.

## Đã hoàn thành (Toàn bộ Round Thực nghiệm & Chuẩn bị Báo cáo Tiến độ)
- Workspace: /Users/vup/Documents/edu/data-mining/data-mining-project/
- Đã thiết lập virtual environment `.venv` chuẩn hóa với Python 3.12.14 (`/usr/local/bin/python3.12`) và cài đặt đầy đủ: `numpy<2` (1.26.4), `torch` (2.2.2), `scipy`, `pandas`, `scikit-learn`, `umap-learn`, `matplotlib`, `seaborn`, `jupyter`, `ipykernel`, `pytest` (tương thích hoàn toàn kiến trúc Intel Mac x86_64).
- Đã đăng ký kernelspec Jupyter chuyên dụng `Python (.venv) - DSADS` (`dsads_venv`) và bổ sung `.pth` file (`.venv/lib/python3.12/site-packages/dsads_src.pth`) liên kết trực tiếp tới `src/`. Bất kỳ môi trường hay kernel nào chạy cũng nạp được `dsads`, `diagrams`, `features` mà không bao giờ gặp lỗi thiếu module.
- Đã khảo sát toàn diện 51.300.000 điểm dữ liệu gốc DSADS, làm rõ hiện tượng 312.927 giá trị 0.000 (99.94% là Tail Zero-Padding chủ đích ở a05, a06, a18).
- **Hoàn thành và lưu trữ pre-rendered outputs 100% cho cả 3 Notebooks thí nghiệm**:
  1. `notebooks/01_dsads_exploration.ipynb`: Khám phá dữ liệu, dạng sóng gia tốc/góc quay, kiểm tra dị thường số 0, phân chia subject-wise.
  2. `notebooks/02_svm_umap_baseline.ipynb`: Trích xuất 315D đặc trưng thống kê, 5-Fold GroupKFold CV, SVM Baseline, UMAP (20D) + SVM, trực quan hóa UMAP 2D.
  3. `notebooks/03_deep_learning_models.ipynb`: Kiến trúc InceptionTime và Time Series Transformer (TST), huấn luyện 15 epochs, đồ thị Loss/F1 curves, phân tích dị thường Inductive bias vs Data-hungry.
- **Hoàn thành huấn luyện thực tế và lưu trữ toàn bộ Checkpoints của cả 3 họ mô hình vào `models/`**:
  - `scaler.joblib`: StandardScaler 315 chiều fit trên tập Train (p1-p5).
  - `svm_baseline.joblib`: Mô hình SVM RBF Baseline ($C=3.0, \gamma=0.0015$).
  - `svm_umap.joblib`: Mô hình SVM trên không gian 20 chiều UMAP.
  - `umap_reducer.joblib`: Bộ giảm chiều UMAP 315D -> 20D.
  - `umap_2d.joblib`: Bộ chiếu trực quan hóa UMAP 315D -> 2D.
  - `channel_scaler.npz`: Tham số chuẩn hóa Channel-wise (mean & std) cho Deep Learning.
  - `inception_time.pt` & `inception_time.onnx`: Trọng số và ONNX mô hình InceptionTime CNN 1D ($465.491$ tham số).
  - `transformer_tst.pt` & `transformer_tst.onnx`: Trọng số và ONNX mô hình Time Series Transformer ($75.283$ tham số).
- **Bảng số liệu thực nghiệm thực tế 100% (lưu tại `reports/model_comparison.json` và `reports/dl_results.json`)**:
  - **SVM Baseline**: Val Acc 88.68%, Val Macro-F1 0.8586, Test Acc 81.80%. GroupKFold CV: $90.58\% \pm 5.09\%$.
  - **UMAP + SVM**: Val Acc 68.33%, Val Macro-F1 0.6662, Test Acc 71.58%.
  - **InceptionTime (CNN)**: Val Acc 92.63%, Val Macro-F1 0.9261, Test Acc 81.45%, Test Macro-F1 0.7995.
  - **Time Series Transformer (TST)**: Val Acc 92.11%, Val Macro-F1 0.9212, Test Acc 76.67%, Test Macro-F1 0.7536.
- **Xây dựng ứng dụng Web Demo nguyên mẫu (`demo/app.py`)**:
  - Viết bằng Streamlit, hỗ trợ chọn mẫu kiểm thử từ Test p7-p8 hoặc tải file cảm biến $125 \times 45$.
  - Trực quan hóa sóng gia tốc đa kênh theo thời gian.
  - Nạp trực tiếp mô hình từ `models/` để suy luận và so sánh các pipeline.
  - Chiếu tọa độ mẫu lên không gian UMAP 2D.
- **Hệ thống sơ đồ kiến trúc chuẩn học thuật (`src/diagrams.py` và `reports/figures/`)**:
  - 4 sơ đồ độ phân giải cao: Luồng tổng thể, Nhánh SVM/UMAP, Kiến trúc InceptionTime, Kiến trúc TST.
  - Biểu đồ đường cong huấn luyện Loss/F1 `reports/figures/deep_learning_training_curves.png`.
- **Tài liệu Báo cáo Tiến độ hoàn chỉnh (`reports/PROGRESS_REPORT.md`)**: Sẵn sàng phục vụ thuyết trình tiến độ với giảng viên.
- **Chuẩn hóa Git & Repository**:
  - File `.gitignore` tối ưu, loại trừ 600MB dữ liệu thô và cache `data/`, giữ lại mã nguồn, models gọn nhẹ (<20MB), reports và notebooks.
  - File `README.md` chuyên nghiệp, đầy đủ hướng dẫn cài đặt, chạy thực nghiệm và phân công công việc tiếp theo cho nhóm.
  - Bộ kiểm thử tự động `pytest tests/` đạt 9/9 pass.

## Chưa hoàn thành (Các bước tiếp theo cho nhóm)
1. Tinh chỉnh giao diện Web Demo Streamlit (`demo/app.py`) với các biểu đồ tương tác nâng cao Plotly.
2. Soạn thảo Báo cáo LaTeX chuẩn học thuật (`reports/latex/main.tex`) kèm file PDF và trích dẫn `references.bib`.
3. Chuẩn bị slide thuyết trình tiến độ dựa trên các bảng kết quả và sơ đồ kiến trúc đã sinh.

## Bước tiếp theo
1. Đẩy mã nguồn lên GitHub cá nhân/nhóm và phân quyền cộng tác cho các thành viên.
2. Nhánh 1 tiếp tục hoàn thiện Web Demo theo các yêu cầu chi tiết trong `docs/DELIVERABLES.md` (Mục 4).
3. Nhánh 2 bắt đầu biên soạn báo cáo LaTeX từ số liệu trong `reports/PROGRESS_REPORT.md` và `reports/model_comparison.json`.
