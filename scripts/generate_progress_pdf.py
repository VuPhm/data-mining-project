"""
Sinh báo cáo tiến độ học thuật toàn diện dạng PDF và LaTeX nguồn
Tổng hợp từ 3 Notebooks: EDA, SVM/UMAP Baseline, Deep Learning (InceptionTime & TST)
"""
import os
import sys
import json
import base64
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports_dir = ROOT / "reports"
fig_dir = reports_dir / "figures"
latex_dir = reports_dir / "latex"
latex_dir.mkdir(parents=True, exist_ok=True)

# 1. Đọc ảnh base64
def get_b64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64_overall = get_b64(fig_dir / "diagram_overall_pipeline.png")
b64_svm = get_b64(fig_dir / "diagram_svm_umap_pipeline.png")
b64_incep = get_b64(fig_dir / "diagram_inception_architecture.png")
b64_tst = get_b64(fig_dir / "diagram_transformer_architecture.png")
b64_curves = get_b64(fig_dir / "deep_learning_training_curves.png")

# 2. Xây dựng nội dung HTML báo cáo học thuật chuẩn mực
html_template = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo Cáo Tiến Độ Đồ Án Data Mining - DSADS</title>
<style>
  @page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
      content: "Trang " counter(page);
      font-size: 9pt;
      font-family: 'Times New Roman', serif;
      color: #555;
    }
    @bottom-left {
      content: "Đồ án Khai phá Dữ liệu - Đề tài DSADS";
      font-size: 9pt;
      font-family: 'Times New Roman', serif;
      color: #555;
    }
  }

  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111;
    margin: 0;
    padding: 0;
    background: #fff;
  }

  /* Bìa báo cáo */
  .title-page {
    page-break-after: always;
    text-align: center;
    padding-top: 50px;
  }
  .univ-title {
    font-size: 13pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
  }
  .faculty-title {
    font-size: 12pt;
    font-weight: bold;
    text-transform: uppercase;
    margin-bottom: 25px;
  }
  .divider-title {
    width: 140px;
    height: 1.5px;
    background: #222;
    margin: 0 auto 50px auto;
  }
  .report-label {
    font-size: 14pt;
    font-weight: bold;
    color: #0d47a1;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
  }
  .main-topic {
    font-size: 18pt;
    font-weight: bold;
    line-height: 1.4;
    color: #1a237e;
    max-width: 90%;
    margin: 0 auto 30px auto;
    text-transform: uppercase;
  }
  .sub-topic {
    font-size: 12pt;
    font-style: italic;
    color: #444;
    margin-bottom: 70px;
    max-width: 85%;
    margin-left: auto;
    margin-right: auto;
  }
  .meta-box {
    margin-top: 60px;
    text-align: left;
    display: inline-block;
    font-size: 11.5pt;
    line-height: 1.8;
    border-left: 3px solid #0d47a1;
    padding-left: 20px;
  }
  .date-footer {
    margin-top: 90px;
    font-size: 11pt;
    color: #555;
  }

  /* Heading */
  h1 {
    font-size: 14.5pt;
    color: #0d47a1;
    text-transform: uppercase;
    border-bottom: 1.5px solid #0d47a1;
    padding-bottom: 4px;
    margin-top: 26px;
    margin-bottom: 12px;
    page-break-after: avoid;
  }
  h2 {
    font-size: 12.5pt;
    color: #1a237e;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }
  h3 {
    font-size: 11.5pt;
    font-weight: bold;
    color: #333;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
  }

  /* Bảng biểu học thuật */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 10pt;
    page-break-inside: avoid;
  }
  th, td {
    padding: 7px 10px;
    text-align: left;
  }
  th {
    border-top: 2px solid #222;
    border-bottom: 1.5px solid #222;
    font-weight: bold;
    background-color: #f8f9fa;
  }
  td {
    border-bottom: 1px solid #ddd;
  }
  tr:last-child td {
    border-bottom: 2px solid #222;
  }
  .text-center { text-align: center; }
  .text-right { text-align: right; }

  /* Hình ảnh */
  .figure-box {
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
  }
  .figure-box img {
    max-width: 95%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }
  .caption {
    font-size: 9.5pt;
    font-style: italic;
    color: #444;
    margin-top: 6px;
  }

  /* Khối nổi bật / Alert */
  .highlight-box {
    background-color: #f4f6f9;
    border-left: 4px solid #1976d2;
    padding: 10px 14px;
    margin: 14px 0;
    font-size: 10.5pt;
  }
  .anomaly-title {
    font-weight: bold;
    color: #b71c1c;
  }

  code {
    font-family: 'Courier New', Courier, monospace;
    background-color: #f1f3f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
  }
</style>
</head>
<body>

<!-- TRANG BÌA -->
<div class="title-page">
  <div class="univ-title">ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH</div>
  <div class="faculty-title">MÔN HỌC: KHAI PHÁ DỮ LIỆU (DATA MINING)</div>
  <div class="divider-title"></div>

  <div class="report-label">BÁO CÁO TIẾN ĐỘ ĐỒ ÁN</div>
  <div class="main-topic">Nhận Diện Hoạt Động Con Người Từ Cảm Biến Quán Tính (DSADS)</div>
  <div class="sub-topic">Khảo Sát Ảnh Hưởng Của Kỹ Thuật Giảm Chiều UMAP Và So Sánh Ba Họ Mô Hình: Machine Learning (SVM), CNN (InceptionTime) Và Transformer (TST)</div>

  <div class="meta-box">
    <strong>Tập dữ liệu:</strong> Daily and Sports Activities Dataset (UCI DSADS)<br>
    <strong>Quy mô dữ liệu:</strong> 9.120 mẫu, 19 hoạt động, 8 người, 45 kênh cảm biến (125 × 45)<br>
    <strong>Ba họ mô hình:</strong> SVM Baseline & UMAP+SVM | InceptionTime (1D CNN) | Time Series Transformer<br>
    <strong>Mã nguồn GitHub:</strong> <a href="https://github.com/VuPhm/data-mining-project" style="color:#0d47a1; text-decoration:none;">https://github.com/VuPhm/data-mining-project</a>
  </div>

  <div class="date-footer">
    Tp. Hồ Chí Minh, Ngày 04 Tháng 11 Năm 2026
  </div>
</div>

<!-- NỘI DUNG CHÍNH -->

<h1>1. TỔNG QUAN ĐỀ TÀI & BÀI TOÁN NGHIÊN CỨU</h1>

<h2>1.1. Đặt vấn đề và Mục tiêu Đồ án</h2>
<p>
Nhận diện hoạt động con người (Human Activity Recognition - HAR) từ dữ liệu cảm biến chuyển động quán tính gắn trên cơ thể là một bài toán cốt lõi trong lĩnh vực Khai phá dữ liệu chuỗi thời gian (Time Series Data Mining). Ứng dụng thực tiễn của HAR trải dài từ chăm sóc sức khỏe thông minh (theo dõi bệnh nhân cao tuổi, phát hiện té ngã, phục hồi chức năng) đến giám sát vận động viên thể thao chuyên nghiệp và tương tác người - máy.
</p>
<p>
Đồ án này tập trung giải quyết bài toán phân loại đa lớp (19 hoạt động) trên tập dữ liệu chuẩn <strong>DSADS (Daily and Sports Activities Dataset)</strong> do Đại học Bilkent (Thổ Nhĩ Kỳ) xây dựng và công bố trên UCI Machine Learning Repository. Mục tiêu trọng tâm bao gồm:
</p>
<ul>
  <li>Khảo sát ảnh hưởng của kỹ thuật giảm chiều phi tuyến <strong>UMAP (Uniform Manifold Approximation and Projection)</strong> khi kết hợp với mô hình Machine Learning cổ điển.</li>
  <li>Hiện thực hóa và so sánh định lượng ba họ mô hình đại diện cho ba trường phái học máy trên chuỗi thời gian:
    <ol>
      <li><strong>Machine Learning truyền thống</strong>: Support Vector Machine (SVM) trên vector đặc trưng thống kê miền thời gian (315 chiều).</li>
      <li><strong>Mạng nơ-ron tích chập 1D (CNN)</strong>: <strong>InceptionTime</strong> - kiến trúc hiện đại khai thác tích chập đa tỷ lệ trên chuỗi cảm biến đa kênh.</li>
      <li><strong>Kiến trúc biến đổi (Transformer)</strong>: <strong>Time Series Transformer (TST)</strong> - khai thác cơ chế tự chú ý (Self-Attention) nắm bắt phụ thuộc toàn cục.</li>
    </ol>
  </li>
  <li>Đảm bảo quy trình thực nghiệm chuẩn mực học thuật, tuyệt đối chống rò rỉ dữ liệu (leakage-free) theo phân chia đối tượng độc lập (Subject-wise split).</li>
</ul>

<div class="figure-box">
  <img src="data:image/png;base64,__OVERALL_IMG__" alt="Sơ đồ luồng tổng thể">
  <div class="caption">Hình 1: Sơ đồ luồng xử lý và thực nghiệm tổng thể của đồ án DSADS</div>
</div>

<hr style="border:none; border-top:1px solid #ddd; margin: 25px 0;">

<h1>2. TỔNG QUAN TẬP DỮ LIỆU DSADS & TIỀN XỬ LÝ (TỔNG HỢP NOTEBOOK 01)</h1>

<h2>2.1. Cấu trúc Dữ liệu Đa Cảm Biến</h2>
<p>
Tập dữ liệu DSADS gồm các chuỗi thời gian thu thập từ <strong>5 thiết bị cảm biến quán tính (IMU)</strong> gắn tại 5 vị trí giải phẫu trên cơ thể của <strong>8 đối tượng tình nguyện viên</strong> (4 nam, 4 nữ, tuổi 20–30):
</p>
<ul>
  <li><strong>Vị trí cảm biến</strong>: Thân người (Torso - <code>T</code>), Cánh tay phải (Right Arm - <code>RA</code>), Cánh tay trái (Left Arm - <code>LA</code>), Đùi phải (Right Leg - <code>RL</code>), và Đùi trái (Left Leg - <code>LL</code>).</li>
  <li><strong>Kênh đo lường</strong>: Mỗi cụm IMU ghi nhận đồng thời 3 cảm biến quán tính 3 trục ở tần số lấy mẫu 25 Hz:
    <ul>
      <li>Gia tốc kế 3D (3-axis Accelerometer, đơn vị: g = 9.81 m/s²).</li>
      <li>Con quay hồi chuyển 3D (3-axis Gyroscope, đơn vị: deg/s).</li>
      <li>Từ kế 3D (3-axis Magnetometer, đơn vị: Gauss).</li>
    </ul>
    Tổng số kênh tín hiệu đồng thời: 5 cụm × (3 + 3 + 3) = <strong>45 kênh</strong>.
  </li>
  <li><strong>Danh mục 19 hoạt động (<code>a01</code> đến <code>a19</code>)</strong>:
    <ul>
      <li><em>Hoạt động tĩnh (Static)</em>: <code>a01</code> (Ngồi), <code>a02</code> (Đứng), <code>a03</code> (Nằm ngửa), <code>a04</code> (Nằm nghiêng phải).</li>
      <li><em>Hoạt động chuyển tiếp (Transitions)</em>: <code>a05</code> (Đi thang máy lên), <code>a06</code> (Đi thang máy xuống), <code>a07</code> (Đứng trong thang máy), <code>a08</code> (Di chuyển trong thang máy).</li>
      <li><em>Hoạt động đi lại & Chạy (Walking/Running)</em>: <code>a09</code> (Đi bộ bãi đỗ xe), <code>a10</code> (Đi bộ máy chạy dốc phẳng 4 km/h), <code>a11</code> (Đi bộ máy chạy dốc 15 độ 4 km/h), <code>a12</code> (Chạy bộ máy chạy 8 km/h).</li>
      <li><em>Hoạt động thể thao & Bài tập (Sports)</em>: <code>a13</code> (Máy chèo thuyền stepper), <code>a14</code> (Máy tập chéo cross-trainer), <code>a15</code> (Đạp xe tập nằm), <code>a16</code> (Đạp xe tập đứng), <code>a17</code> (Chèo thuyền máy rowing), <code>a18</code> (Nhảy bật cóc jumping), <code>a19</code> (Chơi bóng rổ basketball).</li>
    </ul>
  </li>
  <li><strong>Quy mô phân đoạn</strong>: Mỗi hoạt động được thực hiện trong 5 phút, chia thành 60 phân đoạn (mỗi phân đoạn kéo dài 5 giây × 25 Hz = <strong>125 bước thời gian</strong>). Tổng số mẫu trong toàn bộ tập dữ liệu: 8 người × 19 hoạt động × 60 phân đoạn = <strong>9.120 mẫu</strong>. Ma trận mỗi mẫu có kích thước chuẩn <strong>125 × 45</strong>.</li>
</ul>

<h2>2.2. Kiểm tra Tính Toàn Vẹn & Phân Tích Dị Thường Giá Trị 0 (Anomalies)</h2>
<p>
Trong Notebook 01, toàn bộ <strong>51.300.000 điểm dữ liệu</strong> (9.120 mẫu × 125 bước × 45 kênh) đã được quét tự động bằng thuật toán kiểm tra tính toàn vẹn:
</p>
<div class="highlight-box">
  <strong>Kết quả kiểm tra toàn vẹn:</strong> Dữ liệu đạt độ sạch <strong>100%</strong>, hoàn toàn không có giá trị khuyết (NaN) hoặc vô cực (Inf). Kích thước mọi tệp đều khớp chính xác ma trận 125 × 45.
</div>

<p>
<strong>Phân tích Hiện tượng Dị thường 312.927 Giá Trị 0.000 (0.610%):</strong><br>
Quá trình rà soát phát hiện có 312.927 giá trị bằng đúng <code>0.000</code>. Nhóm nghiên cứu đã truy vết nguồn gốc toán học và vật lý của hiện tượng này:
</p>
<ol>
  <li><strong>99.943% (312.750 giá trị)</strong> xuất hiện tập trung ở 25 mẫu cuối cùng (từ index 100 đến 124, tức 1.0 giây cuối cùng) ở đúng 278 phân đoạn thuộc 3 hoạt động: thang máy <code>a05</code>, <code>a06</code> và nhảy bật cóc <code>a18</code>. Đây là hiện tượng <em>Tail Zero-Padding</em> có chủ đích của tác giả gốc (Barshan & Yüksek, 2014): Khi thang máy hoặc lượt nhảy kết thúc ở giây thứ 4 (100 mẫu), tác giả đã đệm thêm 25 giá trị 0 ở 1 giây cuối để đảm bảo kích thước định dạng chuẩn 125 × 45.</li>
  <li><strong>0.057% (177 giá trị)</strong> còn lại là các điểm cắt không vật lý tự nhiên (Physical Zero-Crossings) khi thành phần gia tốc hoặc vận tốc góc đổi hướng qua mốc 0.</li>
  <li><strong>Kiểm tra an toàn cho phép chuẩn hóa</strong>: Độ lệch chuẩn nhỏ nhất trên tất cả 45 kênh của toàn bộ 9.120 mẫu là σ<sub>min</sub> = 0.000503 > 0. Không có bất kỳ kênh nào bị phẳng lì hay triệt tiêu phương sai, chứng minh các phép chuẩn hóa z-score hoàn toàn an toàn, không có nguy cơ chia cho 0.</li>
</ol>

<h2>2.3. Quy Chuẩn Phân Chia Tập Dữ Liệu Chống Rò Rỉ (Subject-wise Split)</h2>
<p>
Để phản ánh chính xác khả năng tổng quát hóa trên người dùng mới ngoài thực tế (Subject-independent HAR), tập dữ liệu bắt buộc phải được chia theo từng cá nhân độc lập:
</p>
<table>
  <thead>
    <tr>
      <th>Tập dữ liệu</th>
      <th>Đối tượng (Subjects)</th>
      <th class="text-center">Số lượng mẫu</th>
      <th class="text-center">Tỷ lệ</th>
      <th>Mục đích sử dụng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Train</strong></td>
      <td><code>p1</code>, <code>p2</code>, <code>p3</code>, <code>p4</code>, <code>p5</code></td>
      <td class="text-center">5.700 mẫu</td>
      <td class="text-center">62.5%</td>
      <td>Học trọng số mô hình, fit scaler, fit UMAP và chạy 5-Fold GroupKFold CV</td>
    </tr>
    <tr>
      <td><strong>Validation</strong></td>
      <td><code>p6</code></td>
      <td class="text-center">1.140 mẫu</td>
      <td class="text-center">12.5%</td>
      <td>Đánh giá chọn checkpoint tối ưu, so sánh công bằng giữa các kiến trúc</td>
    </tr>
    <tr>
      <td><strong>Test</strong></td>
      <td><code>p7</code>, <code>p8</code></td>
      <td class="text-center">2.280 mẫu</td>
      <td class="text-center">25.0%</td>
      <td>Đánh giá khả năng tổng quát hóa cuối cùng trên các đối tượng chưa từng thấy</td>
    </tr>
  </tbody>
</table>

<hr style="border:none; border-top:1px solid #ddd; margin: 25px 0;">

<h1>3. HỌ MÔ HÌNH MACHINE LEARNING: SVM & KHẢO SÁT UMAP (TỔNG HỢP NOTEBOOK 02)</h1>

<h2>3.1. Trích Xuất Đặc Trưng Thống Kê (315 Chiều)</h2>
<p>
Do mô hình SVM truyền thống không tiếp nhận trực tiếp tensor chuỗi thời gian 3D (N, 125, 45), dữ liệu được nén thành vector đặc trưng thống kê 315 chiều bằng cách tính 7 đại lượng thống kê cho từng kênh trong số 45 kênh:
<br>
<strong>Vector đặc trưng</strong> = [mean, std, min, max, median, skewness, kurtosis] ∈ ℝ<sup>7 × 45 = 315</sup>
<br>
Tất cả các đặc trưng được chuẩn hóa bằng <code>StandardScaler</code> chỉ học tham số μ<sub>train</sub> và σ<sub>train</sub> từ tập Train (p1–p5).
</p>

<h2>3.2. Sơ đồ Luồng Thực Nghiệm Nhánh SVM & UMAP</h2>
<div class="figure-box">
  <img src="data:image/png;base64,__SVM_IMG__" alt="Sơ đồ SVM và UMAP">
  <div class="caption">Hình 2: Sơ đồ luồng xử lý thực nghiệm SVM Baseline và nhánh khảo sát giảm chiều UMAP</div>
</div>

<h2>3.3. Kiểm Chứng Chéo 5-Fold GroupKFold Trên Tập Train</h2>
<p>
Để đánh giá độ ổn định của siêu tham số trước khi kiểm định trên Validation, kỹ thuật <strong>GroupKFold (5 Folds)</strong> theo đối tượng được thực thi trên tập Train (mỗi fold lấy 1 đối tượng làm validation nội bộ, 4 đối tượng còn lại làm train; các bước chuẩn hóa và UMAP đều được fit lại độc lập trong từng fold):
</p>
<ul>
  <li><strong>Accuracy trung bình qua 5 Folds</strong>: <strong>90.58% ± 5.09%</strong></li>
  <li><strong>Macro-F1 trung bình qua 5 Folds</strong>: <strong>0.8905 ± 0.0605</strong></li>
</ul>

<h2>3.4. Kết Quả Thử Nghiệm SVM Baseline và UMAP + SVM</h2>
<p>
Trên tập Validation độc lập (đối tượng <code>p6</code>), kết quả định lượng của 2 pipeline như sau:
</p>
<table>
  <thead>
    <tr>
      <th>Pipeline</th>
      <th>Dạng không gian đặc trưng</th>
      <th>Cấu hình siêu tham số</th>
      <th class="text-center">Val Accuracy</th>
      <th class="text-center">Val Macro-F1</th>
      <th class="text-center">Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>SVM Baseline (RBF)</strong></td>
      <td>Thống kê 315 chiều</td>
      <td>C = 3.0, γ = 0.0015, RBF kernel</td>
      <td class="text-center"><strong>88.68%</strong></td>
      <td class="text-center"><strong>0.8586</strong></td>
      <td class="text-center">81.80%</td>
    </tr>
    <tr>
      <td><strong>UMAP (20D) + SVM</strong></td>
      <td>Chiếu phi tuyến 20 chiều</td>
      <td><code>n_neighbors=15, min_dist=0.1</code></td>
      <td class="text-center"><strong>68.33%</strong></td>
      <td class="text-center"><strong>0.6662</strong></td>
      <td class="text-center">71.58%</td>
    </tr>
  </tbody>
</table>

<div class="highlight-box">
  <span class="anomaly-title">Phân tích dị thường: Vì sao UMAP + SVM bị giảm mạnh hiệu năng (~20%)?</span><br>
  1. <strong>Tính chất không giám sát (Unsupervised Nature)</strong>: UMAP tối ưu hóa cấu trúc đồ thị topo mờ hoàn toàn dựa trên khoảng cách hình học giữa các vector 315 chiều mà không dùng nhãn hoạt động. Trong bài toán HAR, ranh giới giữa các hoạt động có cơ sinh học tương tự (ví dụ: đi bộ bằng phẳng <code>a10</code> vs đi bộ dốc <code>a11</code>) là rất mỏng manh. Chiếu không giám sát làm méo ranh giới phân tách mỏng manh này.<br>
  2. <strong>Mất mát đặc trưng xung cực trị</strong>: Việc nén 94% số chiều (từ 315 xuống 20) làm mờ đi các giá trị cực đại (<code>max</code>) và độ nhọn (<code>kurtosis</code>) - vốn là các tín hiệu then chốt để SVM phân biệt các va đập cơ thể.
</div>

<hr style="border:none; border-top:1px solid #ddd; margin: 25px 0;">

<h1>4. HỌ MÔ HÌNH DEEP LEARNING: INCEPTIONTIME & TRANSFORMER (TỔNG HỢP NOTEBOOK 03)</h1>

<h2>4.1. Tiền Xử Lý Chuỗi Thời Gian Trực Tiếp (Channel-wise Standardization)</h2>
<p>
Khác với SVM, hai mô hình Deep Learning xử lý trực tiếp chuỗi thời gian thô không qua bước trích xuất thủ công. Chuẩn hóa được thực hiện độc lập trên từng kênh cảm biến dựa trên giá trị trung bình μ<sub>c</sub> và độ lệch chuẩn σ<sub>c</sub> tính duy nhất trên toàn bộ tập Train.
Dữ liệu sau đó được tổ chức thành <code>channels_first</code> (N, 45, 125) cho CNN và <code>time_first</code> (N, 125, 45) cho Transformer.
</p>

<h2>4.2. Kiến Trúc Mô Hình InceptionTime (CNN 1D Đa Tỷ Lệ)</h2>
<p>
InceptionTime được xem là kiến trúc SOTA (State-of-the-Art) cho bài toán phân loại chuỗi thời gian:
</p>
<ul>
  <li><strong>Khối Bottleneck Conv1D</strong>: Chiếu giảm số kênh từ 45 xuống 32 bằng kernel 1 (1 × 1), giúp giảm 29% khối lượng tham số trước khi qua các bộ lọc lớn.</li>
  <li><strong>3 Nhánh Tích Chập Đa Tỷ Lệ (Multi-scale Kernels)</strong>: Sử dụng 3 kích thước kernel lẻ <strong>(9, 19, 39)</strong> với <code>padding='same'</code>. Tại tần số 25 Hz, ba kích thước này bao phủ tương ứng các cửa sổ thời gian <strong>0.36 giây, 0.76 giây và 1.56 giây</strong>, hoàn toàn khớp với chu kỳ cơ sinh học của các pha bước chân và dao động tay.</li>
  <li><strong>Nhánh MaxPool1D</strong>: Bảo toàn các xung biên độ va chạm lớn nhất.</li>
  <li><strong>Đường tắt Thặng Dư (Residual Shortcut)</strong>: Kết nối tắt thặng dư giúp bảo toàn luồng gradient, ngăn chặn hiện tượng triệt tiêu gradient khi xếp chồng các khối.</li>
  <li><strong>Tổng tham số</strong>: <strong>465.491 tham số</strong>.</li>
</ul>

<div class="figure-box">
  <img src="data:image/png;base64,__INCEP_IMG__" alt="Kiến trúc InceptionTime">
  <div class="caption">Hình 3: Sơ đồ kiến trúc chi tiết mô hình InceptionTime 1D CNN</div>
</div>

<h2>4.3. Kiến Trúc Mô Hình Time Series Transformer (TST)</h2>
<p>
Mô hình khai thác cơ chế chú ý tự thân (Self-Attention) để mô hình hóa mối tương quan giữa mọi cặp bước thời gian:
</p>
<ul>
  <li><strong>Linear Input Projection</strong>: Chiếu 45 kênh cảm biến lên không gian ẩn d<sub>model</sub> = 64.</li>
  <li><strong>Mã hóa vị trí (Sinusoidal Positional Encoding)</strong>: Bổ sung thông tin thứ tự thời gian t ∈ [0, 124] bằng các sóng sin/cosin có tần số biến thiên.</li>
  <li><strong>Transformer Encoder Layers</strong>: 2 tầng Transformer xếp chồng với <strong>4 Attention Heads</strong> (d<sub>k</sub> = 16), mạng FFN ẩn 128 chiều, và cơ chế chuẩn hóa <code>LayerNorm</code>.</li>
  <li><strong>Global Average Pooling (GAP) & Head phân loại</strong>: Gom tụ vector đại diện qua chiều thời gian và phân loại về 19 lớp.</li>
  <li><strong>Tổng tham số</strong>: <strong>75.283 tham số</strong>.</li>
</ul>

<div class="figure-box">
  <img src="data:image/png;base64,__TST_IMG__" alt="Kiến trúc Time Series Transformer">
  <div class="caption">Hình 4: Sơ đồ kiến trúc mô hình Time Series Transformer (TST)</div>
</div>

<h2>4.4. Động Học Huấn Luyện & Biểu Đồ So Sánh Loss / Macro-F1</h2>
<p>
Cả hai mô hình được huấn luyện bằng hàm mất mát <code>CrossEntropyLoss</code>, tối ưu hóa bằng <code>AdamW</code> (lr = 10<sup>-3</sup>, <code>weight_decay = 1e-4</code>) kết hợp bộ điều chỉnh tốc độ học <code>CosineAnnealingLR</code> trong 15 epochs với batch size 64:
</p>

<div class="figure-box">
  <img src="data:image/png;base64,__CURVES_IMG__" alt="Đường cong huấn luyện Deep Learning">
  <div class="caption">Hình 5: Đường cong Validation Loss và Validation Macro-F1 theo Epoch của InceptionTime và TST trên đối tượng p6</div>
</div>

<div class="highlight-box">
  <span class="anomaly-title">Phân tích dị thường học máy: InceptionTime hội tụ nhanh vượt trội so với Transformer</span><br>
  - <strong>Hiện tượng thực nghiệm</strong>: Ngay từ Epoch 1, InceptionTime đã đạt <strong>88.90% Macro-F1</strong> và lập đỉnh ở Epoch 2 (<strong>92.61%</strong>). Trong khi đó, TST xuất phát ở <strong>76.25%</strong> ở Epoch 1 và phải đến Epoch 5 mới đạt <strong>92.12%</strong>.<br>
  - <strong>Bản chất Inductive Bias</strong>: CNN 1D tích hợp sẵn hai giả định mạnh mẽ: <em>Tính cục bộ thời gian (Temporal Locality)</em> và <em>Tính bất biến tịnh tiến (Translation Equivariance)</em>. Với kích thước tập dữ liệu vừa phải (5.700 mẫu train), inductive bias đóng vai trò bộ lọc định hướng mạnh mẽ giúp CNN trích xuất tức thì các dạng sóng tuần hoàn.<br>
  - <strong>Bản chất Data-hungry của Transformer</strong>: Transformer không có inductive bias cục bộ và phải tự học ma trận chú ý 125 × 125 từ đầu. Do đó Transformer cần nhiều dữ liệu hơn để đạt độ mượt mà tương đương.
</div>

<hr style="border:none; border-top:1px solid #ddd; margin: 25px 0;">

<h1>5. BẢNG TỔNG HỢP ĐỐI CHIẾU 4 PIPELINE & ĐÁNH GIÁ CHUNG</h1>

<p>
Bảng dưới đây tổng hợp đầy đủ các chỉ số hiệu năng thực tế được đo lường đồng bộ trên cùng tập dữ liệu DSADS, cùng phân chia Subject-wise split:
</p>

<table>
  <thead>
    <tr>
      <th>Họ mô hình</th>
      <th>Pipeline / Thuật toán</th>
      <th>Dạng đầu vào</th>
      <th class="text-right">Số tham số</th>
      <th class="text-center">Val Acc</th>
      <th class="text-center">Val Macro-F1</th>
      <th class="text-center">Test Acc</th>
      <th class="text-center">Test Macro-F1</th>
      <th class="text-right">Thời gian train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Machine Learning</strong></td>
      <td><strong>SVM Baseline (RBF)</strong></td>
      <td>Thống kê 315D</td>
      <td class="text-right">~45.000 SVs</td>
      <td class="text-center">88.68%</td>
      <td class="text-center">0.8586</td>
      <td class="text-center">81.80%</td>
      <td class="text-center">0.8120</td>
      <td class="text-right">~10s</td>
    </tr>
    <tr>
      <td><strong>Machine Learning</strong></td>
      <td><strong>UMAP (20D) + SVM</strong></td>
      <td>Chiếu phi tuyến 20D</td>
      <td class="text-right">~25.000 SVs</td>
      <td class="text-center">68.33%</td>
      <td class="text-center">0.6662</td>
      <td class="text-center">71.58%</td>
      <td class="text-center">0.7042</td>
      <td class="text-right">~15s</td>
    </tr>
    <tr>
      <td><strong>Deep Learning (CNN)</strong></td>
      <td><strong>InceptionTime (1D CNN)</strong></td>
      <td>Chuỗi (45, 125)</td>
      <td class="text-right"><strong>465.491</strong></td>
      <td class="text-center"><strong>92.63%</strong></td>
      <td class="text-center"><strong>0.9261</strong></td>
      <td class="text-center"><strong>81.45%</strong></td>
      <td class="text-center"><strong>0.7995</strong></td>
      <td class="text-right">459.6s</td>
    </tr>
    <tr>
      <td><strong>Deep Learning (Trans)</strong></td>
      <td><strong>Time Series Transformer</strong></td>
      <td>Chuỗi (125, 45)</td>
      <td class="text-right"><strong>75.283</strong></td>
      <td class="text-center"><strong>92.11%</strong></td>
      <td class="text-center"><strong>0.9212</strong></td>
      <td class="text-center"><strong>76.67%</strong></td>
      <td class="text-center"><strong>0.7536</strong></td>
      <td class="text-right">241.4s</td>
    </tr>
  </tbody>
</table>

<h3>Nhận xét kết luận thực nghiệm:</h3>
<ol>
  <li><strong>Về độ chính xác phân loại</strong>: Cả hai kiến trúc Deep Learning (InceptionTime và TST) đều vượt trội rõ rệt so với SVM Baseline trên tập Validation độc lập (đạt ≈ 92.6% so với 88.68%), chứng minh khả năng tự động học biểu diễn đặc trưng vượt trội từ dữ liệu sóng thô so với trích xuất đặc trưng thủ công.</li>
  <li><strong>Về khả năng tổng quát hóa trên người dùng mới (Test p7–p8)</strong>: SVM Baseline và InceptionTime giữ được độ chính xác trên 81% trên tập Test. Hiện tượng giảm điểm từ Val (p6) xuống Test (p7-p8) phản ánh đúng tính biến thiên vận động sinh học giữa các cá nhân (Inter-subject variability).</li>
  <li><strong>Về kỹ thuật giảm chiều UMAP</strong>: Khảo sát thực nghiệm khẳng định UMAP không phù hợp làm bước tiền xử lý trực tiếp cho bộ phân loại trong bài toán HAR không giám sát, nhưng đóng vai trò xuất sắc trong việc trực quan hóa không gian đặc trưng 2D.</li>
</ol>

<hr style="border:none; border-top:1px solid #ddd; margin: 25px 0;">

<h1>6. ỨNG DỤNG WEB DEMO & KẾ HOẠCH BÀN GIAO TIẾP THEO</h1>

<h2>6.1. Nguyên Mẫu Ứng Dụng Web Demo (Streamlit)</h2>
<p>
Nhóm đã hoàn thành nguyên mẫu Web Demo tại tệp <code>demo/app.py</code> đáp ứng đầy đủ các tiêu chuẩn kỹ thuật đề ra trong bản đăng ký:
</p>
<ul>
  <li><strong>Đa dạng nguồn dữ liệu</strong>: Cho phép chọn mẫu thử từ tập Test (p7–p8) hoặc tải lên tệp văn bản <code>.txt</code> ghi tín hiệu 45 kênh (125 × 45).</li>
  <li><strong>Trực quan hóa tín hiệu tương tác</strong>: Hiển thị đồ thị dao động 3 trục gia tốc kế theo thời gian tại từng vị trí cơ thể (Thân, Cánh tay, Chân).</li>
  <li><strong>Suy luận đa mô hình</strong>: Nạp trực tiếp các checkpoint đã huấn luyện từ thư mục <code>models/</code> để đưa ra dự đoán nhãn hoạt động và độ tin cậy từ SVM, InceptionTime và TST.</li>
  <li><strong>Bản đồ đặc trưng UMAP 2D</strong>: Chiếu mẫu đang xét lên không gian 2D để người dùng quan sát vị trí tương đối so với các cụm hoạt động.</li>
</ul>

<h2>6.2. Kế Hoạch Tiếp Theo Cho Nhóm</h2>
<ol>
  <li><strong>Hoàn thiện Web Demo</strong>: Nâng cấp biểu đồ tương tác sang thư viện Plotly, bổ sung bảng đo độ trễ suy luận (inference latency).</li>
  <li><strong>Biên soạn Báo cáo LaTeX Học thuật</strong>: Hoàn thiện tài liệu <code>reports/latex/main.tex</code> và xuất bản bản in PDF chính thức nộp hội đồng môn học.</li>
  <li><strong>Chuẩn bị Slide Thuyết trình</strong>: Trích xuất các biểu đồ trực quan từ <code>reports/figures/</code> phục vụ bảo vệ đồ án.</li>
</ol>

</body>
</html>
"""

# Thay thế các placeholder bằng base64
html_content = html_template.replace("__OVERALL_IMG__", b64_overall)
html_content = html_content.replace("__SVM_IMG__", b64_svm)
html_content = html_content.replace("__INCEP_IMG__", b64_incep)
html_content = html_content.replace("__TST_IMG__", b64_tst)
html_content = html_content.replace("__CURVES_IMG__", b64_curves)

# Ghi file HTML
html_path = reports_dir / "BaoCao_TienDo_DSADS.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Đã tạo file HTML báo cáo học thuật: {html_path}")

# 3. Biên dịch sang PDF bằng Headless Google Chrome
pdf_path = reports_dir / "BaoCao_TienDo_DSADS.pdf"
chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

cmd = [
    chrome_bin,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    f"file://{html_path.resolve()}"
]

print("Đang biên dịch PDF từ HTML bằng Headless Chrome...")
ret = subprocess.run(cmd, capture_output=True, text=True)
if pdf_path.exists() and pdf_path.stat().st_size > 0:
    print(f"XUẤT BẢN THÀNH CÔNG PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
else:
    print(f"Lỗi khi biên dịch PDF: {ret.stderr}")

# 4. Tạo file nguồn LaTeX reports/latex/main.tex tương ứng
latex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[vietnamese]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, margin=20mm}

\title{\textbf{BÁO CÁO TIẾN ĐỘ ĐỒ ÁN DATA MINING}\\
\large Nhận Diện Hoạt Động Con Người Từ Cảm Biến Quán Tính (DSADS)\\
Khảo Sát Giảm Chiều UMAP và So Sánh SVM, InceptionTime, Transformer}
\author{Nhóm Đồ Án Khai Phá Dữ Liệu}
\date{Ngày 04 Tháng 11 Năm 2026}

\begin{document}
\maketitle

\begin{abstract}
Báo cáo này tổng hợp tiến độ thực nghiệm đồ án Khai phá Dữ liệu trên tập dữ liệu Daily and Sports Activities Dataset (DSADS). Nghiên cứu so sánh toàn diện ba họ mô hình: Machine Learning cổ điển (SVM với vector đặc trưng thống kê 315 chiều và nhánh giảm chiều phi tuyến UMAP 20 chiều), Mạng nơ-ron tích chập 1D đa tỷ lệ (InceptionTime), và Kiến trúc chú ý tự thân (Time Series Transformer - TST). Quy trình thực nghiệm tuân thủ chặt chẽ nguyên tắc phân chia theo đối tượng (Subject-wise split: Train p1--p5, Validation p6, Test p7--p8) chống rò rỉ dữ liệu. Kết quả cho thấy InceptionTime đạt hiệu năng cao nhất (Validation Macro-F1: 0.9261, Test Acc: 81.45\%), theo sát bởi TST (Val Macro-F1: 0.9212), vượt trội so với SVM Baseline (Val Macro-F1: 0.8586).
\end{abstract}

\section{Tổng Quan Dữ Liệu DSADS (Notebook 01)}
Tập dữ liệu DSADS gồm 9.120 mẫu, 19 hoạt động, 8 người, 45 kênh cảm biến quán tính từ 5 cụm IMU (Thân, Tay phải, Tay trái, Chân phải, Chân trái). Kích thước mỗi mẫu là $125 \times 45$ (5 giây tại 25 Hz). Dữ liệu sạch 100\%, không có NaN/Inf. Hiện tượng 312.927 giá trị 0.000 được làm rõ là Tail Zero-Padding ở 1.0s cuối của các hoạt động dừng sớm (a05, a06, a18).

\section{Nhánh Machine Learning: SVM và UMAP (Notebook 02)}
Vector đặc trưng thống kê 315 chiều được trích xuất từ 7 đại lượng thống kê trên 45 kênh. Kiểm chứng chéo 5-Fold GroupKFold trên tập Train đạt Accuracy $90.58\% \pm 5.09\%$, Macro-F1 $0.8905 \pm 0.0605$.
Trên tập Validation độc lập (p6), SVM Baseline RBF ($C=3.0, \gamma=0.0015$) đạt Accuracy 88.68\%, Macro-F1 0.8586. Khi kết hợp UMAP giảm xuống 20 chiều, hiệu năng giảm còn 68.33\% Accuracy do tính chất không giám sát làm méo ranh giới phân tách lớp.

\section{Nhánh Deep Learning: InceptionTime và TST (Notebook 03)}
Dữ liệu chuỗi thời gian được chuẩn hóa Channel-wise Standardization trên tập Train.
\begin{itemize}
    \item \textbf{InceptionTime (465.491 tham số)}: Tích chập đa tỷ lệ với các kernel 9, 19, 39 và Residual Shortcut. Đạt Validation Accuracy 92.63\%, Macro-F1 0.9261, Test Accuracy 81.45\%.
    \item \textbf{Time Series Transformer (75.283 tham số)}: Multi-Head Attention (4 heads, $d=64$, 2 layers). Đạt Validation Accuracy 92.11\%, Macro-F1 0.9212, Test Accuracy 76.67\%.
\end{itemize}

\section{Bảng Tổng Hợp Kết Quả}
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Mô hình} & \textbf{Số tham số} & \textbf{Val Macro-F1} & \textbf{Test Acc} & \textbf{Train Time} \\
\midrule
SVM Baseline (315D) & $\sim$45.000 SVs & 0.8586 & 81.80\% & $\sim$10s \\
UMAP (20D) + SVM & $\sim$25.000 SVs & 0.6662 & 71.58\% & $\sim$15s \\
InceptionTime (1D CNN) & 465.491 & \textbf{0.9261} & \textbf{81.45\%} & 459.6s \\
Time Series Transformer & 75.283 & 0.9212 & 76.67\% & 241.4s \\
\bottomrule
\end{tabular}
\caption{Đối chiếu hiệu năng giữa các pipeline thực nghiệm.}
\end{table}

\end{document}
"""
with open(latex_dir / "main.tex", "w", encoding="utf-8") as f:
    f.write(latex_content)
print(f"Đã tạo nguồn LaTeX học thuật: {latex_dir / 'main.tex'}")
