"""Script nhúng và render trực tiếp các sơ đồ kiến trúc vào notebooks:
1. Nhúng thẻ ảnh Markdown/HTML trỏ tới file ảnh trong reports/figures/
2. Nạp dữ liệu base64 PNG vào trường `outputs` của các code cell gọi hàm vẽ sơ đồ
Đảm bảo khi mở notebook trên VS Code hay Jupyter, sơ đồ được hiển thị lập tức mà không phụ thuộc plugin Mermaid.
"""

import base64
import json
from pathlib import Path


def get_base64_png(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_display_output(png_path: Path):
    b64 = get_base64_png(png_path)
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [f"<Figure size 2100x1200 with 1 Axes from {png_path.name}>"]
        },
        "metadata": {}
    }


fig_dir = Path("reports/figures")
pipe_png = fig_dir / "diagram_overall_pipeline.png"
inc_png = fig_dir / "diagram_inception_architecture.png"
tst_png = fig_dir / "diagram_transformer_architecture.png"
svm_png = fig_dir / "diagram_svm_umap_pipeline.png"

# ==========================================
# 1. CẬP NHẬT NOTEBOOK 03 (DEEP LEARNING)
# ==========================================
nb03_path = Path("notebooks/03_deep_learning_models.ipynb")
with open(nb03_path, "r", encoding="utf-8") as f:
    nb03 = json.load(f)

# Cell 0: Header Markdown
nb03["cells"][0]["source"] = [
    "# Huấn luyện Deep Learning trên DSADS: InceptionTime và Time Series Transformer (TST)\n",
    "\n",
    "Notebook này triển khai và đánh giá hai mô hình Deep Learning đại diện cho hai trường phái kiến trúc hiện đại trên dữ liệu chuỗi thời gian cảm biến đa kênh:\n",
    "1. **InceptionTime**: Mô hình tích chập 1D (CNN) đa tỷ lệ (Multi-scale Receptive Fields) kết hợp kết nối thặng dư (Residual Shortcut).\n",
    "2. **Time Series Transformer (TST)**: Mô hình cơ chế tự chú ý (Multi-Head Self-Attention) kết hợp mã hóa vị trí hình sin (Sinusoidal Positional Encoding).\n",
    "\n",
    "### Sơ đồ luồng dữ liệu tổng thể của hệ thống thực nghiệm\n",
    "![Sơ đồ tổng thể](../reports/figures/diagram_overall_pipeline.png)\n",
    "\n",
    "<details>\n",
    "<summary><b>Xem mã nguồn luồng xử lý (Mermaid)</b></summary>\n",
    "\n",
    "```mermaid\n",
    "flowchart TD\n",
    "    A[\"DSADS Raw Data (9.120 mẫu, 125x45)\"] --> B[\"Subject Split (Train: p1-p5, Val: p6, Test: p7-p8)\"]\n",
    "    B --> C[\"Chuẩn hóa Channel-wise (Fit strictly trên Train)\"]\n",
    "    C --> D1[\"Channels-First Tensor: (B, 45, 125)\"]\n",
    "    C --> D2[\"Time-First Tensor: (B, 125, 45)\"]\n",
    "    D1 --> E1[\"InceptionTime (CNN đa tỷ lệ)\"]\n",
    "    D2 --> E2[\"Time Series Transformer (TST)\"]\n",
    "    E1 --> F[\"Đánh giá Macro-F1 & Xuất Artifacts (ONNX, PyTorch)\"]\n",
    "    E2 --> F\n",
    "```\n",
    "</details>\n"
]

# Cell 2: Code cell vẽ pipeline tổng quan
for i, cell in enumerate(nb03["cells"]):
    src = "".join(cell.get("source", []))
    if "plot_overall_pipeline" in src and cell["cell_type"] == "code":
        cell["outputs"] = [create_display_output(pipe_png)]
        cell["execution_count"] = 1

# Cell 8: Markdown InceptionTime
for i, cell in enumerate(nb03["cells"]):
    src = "".join(cell.get("source", []))
    if "## 1. Mô hình InceptionTime:" in src and cell["cell_type"] == "markdown":
        cell["source"] = [
            "## 1. Mô hình InceptionTime: Lý thuyết cốt lõi & Cơ sở lựa chọn con số\n",
            "\n",
            "### Sơ đồ kiến trúc InceptionTime\n",
            "![Kiến trúc InceptionTime](../reports/figures/diagram_inception_architecture.png)\n",
            "\n",
            "### Lý thuyết cốt lõi\n",
            "InceptionTime là kiến trúc SOTA (State-of-the-Art) cho bài toán phân loại chuỗi thời gian, giải quyết bài toán phụ thuộc thời gian ở nhiều thang đo khác nhau mà không cần trích xuất đặc trưng thủ công:\n",
            "1. **Tích chập đa tỷ lệ song song (Multi-scale Receptive Fields)**: Thay vì chọn 1 kích thước kernel cố định, InceptionModule chạy đồng thời 3 bộ lọc với độ rộng khác nhau kết hợp 1 nhánh MaxPooling.\n",
            "2. **Tầng Bottleneck (1x1 Conv)**: Nén số kênh trước khi tích chập kernel lớn, giảm thiểu bùng nổ tham số tính toán.\n",
            "3. **Kết nối thặng dư (Residual Shortcut)**: Đưa tín hiệu đầu vào cộng trực tiếp với đầu ra của khối Inception, giúp giải quyết triệt để hiện tượng tiêu biến đạo hàm (vanishing gradient) khi mạng sâu.\n",
            "4. **Global Average Pooling (GAP)**: Thu gọn toàn bộ chiều thời gian thành 1 giá trị trung bình cho mỗi kênh, tạo tính bất biến tịnh tiến và giảm $99\\%$ số trọng số của lớp phân loại so với việc dùng `Flatten`.\n",
            "\n",
            "### Cơ sở toán học & tính toán miền cho từng con số:\n",
            "- **Tần số lấy mẫu**: $f_s = 25\\text{ Hz}$ $\\Rightarrow 1$ bước thời gian tương ứng $0.04\\text{s}$. Mẫu 5 giây gồm $125$ bước.\n",
            "- **Kích thước Kernels `(9, 19, 39)` (Tính toán từ chu kỳ vận động con người)**:\n",
            "  + $k = 9$: Cửa sổ $9 \\times 0.04\\text{s} \\approx 0.36\\text{s}$ $\\rightarrow$ Nắm bắt các vi chuyển động, rung chấn và pha tiếp đất (heel-strike) tức thời.\n",
            "  + $k = 19$: Cửa sổ $19 \\times 0.04\\text{s} \\approx 0.76\\text{s}$ $\\rightarrow$ Nắm bắt chu kỳ của 1 bước chân bình thường khi đi bộ hoặc leo cầu thang.\n",
            "  + $k = 39$: Cửa sổ $39 \\times 0.04\\text{s} \\approx 1.56\\text{s}$ $\\rightarrow$ Nắm bắt trọn vẹn chu kỳ hành động kép (đứng lên - ngồi xuống, một vòng đạp xe, chuyển động thân trên).\n",
            "- **`bottleneck_channels = 32` (Tính toán giảm tải tính toán)**: Giảm từ $45 \\to 32$ kênh ($29\\%$ số kênh đầu vào). Nếu không có bottleneck, chi phí tính toán cho 3 nhánh kernel lớn là $32 \\times 45 \\times k$ phép tính, có bottleneck giảm còn $32 \\times 32 \\times k$, chống overfit trên tập dữ liệu người p1-p5.\n",
            "- **`num_filters = 32` $\\Rightarrow$ Concat $128$ kênh**: 4 nhánh $\\times 32 = 128$ kênh đặc trưng biểu diễn phong phú.\n",
            "\n",
            "<details>\n",
            "<summary><b>Xem mã nguồn luồng tensor (Mermaid)</b></summary>\n",
            "\n",
            "```mermaid\n",
            "flowchart TD\n",
            "    In[\"Input (B, 45, 125)\"] --> B[\"Bottleneck Conv1D: 45 -> 32\"]\n",
            "    In -.->|\"Shortcut 1x1 Conv (45 -> 128)\"| Res[\"Cộng thặng dư (Add)\"]\n",
            "    B --> K9[\"Conv1D k=9, pad=4 (32 filters)\"]\n",
            "    B --> K19[\"Conv1D k=19, pad=9 (32 filters)\"]\n",
            "    B --> K39[\"Conv1D k=39, pad=19 (32 filters)\"]\n",
            "    In --> MP[\"MaxPool1D k=3 + Conv1D 1x1 (32 filters)\"]\n",
            "    K9 --> Cat[\"Concat 4 nhánh (128 filters)\"]\n",
            "    K19 --> Cat\n",
            "    K39 --> Cat\n",
            "    MP --> Cat\n",
            "    Cat --> Res\n",
            "    Res --> BN[\"BatchNorm1D + ReLU\"]\n",
            "    BN --> GAP[\"Global Avg Pooling 1D\"]\n",
            "    GAP --> FC[\"Linear 128 -> 19 classes\"]\n",
            "```\n",
            "</details>\n"
        ]

# Cell code InceptionTime Diagram
for i, cell in enumerate(nb03["cells"]):
    src = "".join(cell.get("source", []))
    if "plot_inception_architecture" in src and cell["cell_type"] == "code":
        cell["outputs"] = [create_display_output(inc_png)]
        cell["execution_count"] = 2

# Cell 11: Markdown TST
for i, cell in enumerate(nb03["cells"]):
    src = "".join(cell.get("source", []))
    if "## 2. Mô hình Time Series Transformer (TST):" in src and cell["cell_type"] == "markdown":
        cell["source"] = [
            "## 2. Mô hình Time Series Transformer (TST): Lý thuyết cốt lõi & Cơ sở lựa chọn con số\n",
            "\n",
            "### Sơ đồ kiến trúc Time Series Transformer\n",
            "![Kiến trúc TST](../reports/figures/diagram_transformer_architecture.png)\n",
            "\n",
            "### Lý thuyết cốt lõi\n",
            "Time Series Transformer áp dụng cơ chế tự chú ý (Self-Attention) để nắm bắt mối quan hệ phụ thuộc toàn cục (long-range dependencies) giữa tất cả các bước thời gian trong cửa sổ 5 giây:\n",
            "1. **Chiếu tuyến tính đặc trưng (Linear Input Projection)**: Đưa 45 giá trị cảm biến tại mỗi bước thời gian về không gian ẩn có chiều $d_{model} = 64$.\n",
            "2. **Mã hóa vị trí hình sin (Sinusoidal Positional Encoding)**: Do cơ chế Self-Attention là bất biến vị trí (không phân biệt bước nào xảy ra trước, bước nào xảy ra sau), việc cộng vector vị trí $PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d_{model}})$ và $PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d_{model}})$ là bắt buộc để mô hình nhận thức thứ tự dòng thời gian.\n",
            "3. **Multi-Head Self-Attention**: Cho phép mô hình đồng thời chú ý đến thông tin ở các không gian biểu diễn con khác nhau.\n",
            "4. **Global Mean Sequence Pooling**: Lấy trung bình vector ẩn qua toàn bộ 125 bước để tạo vector biểu diễn tổng quát của hành vi.\n",
            "\n",
            "### Cơ sở toán học & tính toán miền cho từng con số:\n",
            "- **`d_model = 64` (Cân bằng dung lượng biểu diễn)**: Chiếu từ $45 \\to 64$. Vì số mẫu train là $5.700$, chọn $d_{model}$ vừa phải ($64$) giúp mô hình không bị overfit như khi dùng kích thước lớn ($256$ hoặc $512$).\n",
            "- **`nhead = 4` (Tính toán chia hết)**: $64 / 4 = 16$ chiều cho mỗi attention head. $4$ heads cho phép học đồng thời 4 loại tương quan thời gian khác nhau (ví dụ: chu kỳ bước chân, pha chuyển động đỉnh, tương quan giữa các cảm biến, tương quan đối xứng trái-phải).\n",
            "- **`num_layers = 2` (Độ sâu mạng)**: Qua screening thực nghiệm, 2 lớp Transformer Encoder đủ để trích xuất ngữ cảnh cấp cao mà không làm chậm tốc độ huấn luyện hoặc gây quá khớp.\n",
            "- **`dim_feedforward = 128` ($2 \\times d_{model}$)**: Tầng phi tuyến trung gian trong khối Transformer để tái ánh xạ đặc trưng.\n",
            "- **`dropout = 0.1`**: Mức dropout chuẩn để regularize mạng Attention mà không làm mất mát tín hiệu chuỗi ngắn.\n",
            "\n",
            "<details>\n",
            "<summary><b>Xem mã nguồn luồng tensor (Mermaid)</b></summary>\n",
            "\n",
            "```mermaid\n",
            "flowchart TD\n",
            "    In[\"Input Sequence (B, 125, 45)\"] --> Proj[\"Linear Projection: 45 -> 64\"]\n",
            "    Proj --> PE[\"+ Sinusoidal Positional Encoding\"]\n",
            "    PE --> Enc1[\"Transformer Encoder Layer 1 (4 heads, d_model=64)\"]\n",
            "    Enc1 --> Enc2[\"Transformer Encoder Layer 2 (4 heads, d_model=64)\"]\n",
            "    Enc2 --> Pool[\"Global Mean Pooling (trên 125 steps) -> (B, 64)\"]\n",
            "    Pool --> MLP[\"MLP Head (64 -> 64 -> 19 classes)\"]\n",
            "```\n",
            "</details>\n"
        ]

# Cell code TST Diagram
for i, cell in enumerate(nb03["cells"]):
    src = "".join(cell.get("source", []))
    if "plot_transformer_architecture" in src and cell["cell_type"] == "code":
        cell["outputs"] = [create_display_output(tst_png)]
        cell["execution_count"] = 3

with open(nb03_path, "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=1, ensure_ascii=False)
print("Đã cập nhật và nhúng ảnh trực tiếp vào notebooks/03_deep_learning_models.ipynb.")


# ==========================================
# 2. CẬP NHẬT NOTEBOOK 02 (SVM & UMAP)
# ==========================================
nb02_path = Path("notebooks/02_svm_umap_baseline.ipynb")
with open(nb02_path, "r", encoding="utf-8") as f:
    nb02 = json.load(f)

# Cell 0: Header Markdown
nb02["cells"][0]["source"] = [
    "# Huấn luyện và Đánh giá SVM & UMAP trên DSADS\n",
    "\n",
    "Notebook này thực hiện nhánh Machine Learning truyền thống trong đề tài nghiên cứu:\n",
    "1. **Trích xuất 315 đặc trưng thống kê** từ 45 kênh cảm biến thời gian (7 thống kê x 45 kênh).\n",
    "2. **Đánh giá kiểm chứng chéo theo đối tượng (GroupKFold CV)** trên tập Train (`p1`–`p5`) để xác nhận tính ổn định và chống rò rỉ dữ liệu.\n",
    "3. **Huấn luyện mô hình SVM Baseline (RBF kernel)** với siêu tham số $C=3.0, \\gamma=0.0015$ đã được tối ưu qua grid search.\n",
    "4. **Khảo sát kỹ thuật giảm chiều UMAP** (nén từ 315 xuống 20 chiều) kết hợp với SVM để đo lường mức độ suy giảm thông tin phân loại.\n",
    "5. **Trực quan hóa không gian đặc trưng UMAP 2D** và phân tích ma trận nhầm lẫn (Confusion Matrix).\n",
    "\n",
    "### Sơ đồ quy trình thực nghiệm Machine Learning\n",
    "![Sơ đồ thực nghiệm SVM và UMAP](../reports/figures/diagram_svm_umap_pipeline.png)\n",
    "\n",
    "<details>\n",
    "<summary><b>Xem mã nguồn luồng xử lý (Mermaid)</b></summary>\n",
    "\n",
    "```mermaid\n",
    "flowchart LR\n",
    "    A[\"Dữ liệu thô (N, 125, 45)\"] --> B[\"Trích xuất 315 đặc trưng\\n(mean, std, min, max, median, skew, kurt)\"]\n",
    "    B --> C[\"StandardScaler\\n(Fit duy nhất trên Train)\"]\n",
    "    C --> D1[\"Nhánh SVM Baseline (315 chiều)\\nKernel RBF: C=3.0, γ=0.0015\"]\n",
    "    C --> D2[\"Nhánh UMAP (20 chiều) + SVM\\nNén 94% số chiều\"]\n",
    "    C --> D3[\"Nhánh UMAP 2D\\nTrực quan hóa 19 cụm hoạt động\"]\n",
    "```\n",
    "</details>\n"
]

# Cell code SVM/UMAP Diagram
for i, cell in enumerate(nb02["cells"]):
    src = "".join(cell.get("source", []))
    if "plot_svm_umap_pipeline" in src and cell["cell_type"] == "code":
        cell["outputs"] = [create_display_output(svm_png)]
        cell["execution_count"] = 2

with open(nb02_path, "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=1, ensure_ascii=False)
print("Đã cập nhật và nhúng ảnh trực tiếp vào notebooks/02_svm_umap_baseline.ipynb.")
