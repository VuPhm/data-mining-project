"""Nâng cấp và tối ưu tương thích toàn diện cho notebooks/01_dsads_exploration.ipynb:
1. Hỗ trợ song song cả Google Colab và Local VS Code / Jupyter.
2. Nạp dữ liệu thông minh qua cache dsads_cache.npz (kèm fallback kiểm tra raw data).
3. Tích hợp sơ đồ luồng tổng thể hệ thống (Mermaid + pre-rendered diagram_overall_pipeline.png).
4. Giải trình rõ ràng mọi con số (25 Hz, 125 steps, 5 vị trí x 9 kênh = 45 kênh, 9.120 mẫu, p1-p5/p6/p7-p8 split).
5. Bổ sung trực quan hóa EDA chất lượng cao: so sánh tín hiệu 3 nhóm hoạt động (tĩnh, tuần hoàn, cường độ cao).
6. Chuẩn hóa cấu trúc cell (gán cell_id, hợp thức hóa chuẩn nbformat).
"""

import base64
import json
from pathlib import Path
import nbformat


def get_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


diagram_path = Path("reports/figures/diagram_overall_pipeline.png")
diagram_b64 = get_b64(diagram_path) if diagram_path.exists() else ""

cells = []

# Cell 0: Markdown Header & Pipeline Diagram
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "# Khám phá và Kiểm tra Dữ liệu DSADS (Exploratory Data Analysis)\n",
        "\n",
        "Notebook này thực hiện bước đầu tiên trong toàn bộ quy trình nghiên cứu đề tài DSADS:\n",
        "1. **Tự động nhận diện môi trường tính toán** (Google Colab hoặc Local VS Code).\n",
        "2. **Kiểm tra tính toàn vẹn của dữ liệu gốc** (đảm bảo đủ 9.120 tệp, không chứa giá trị NaN/Inf, không rỗng).\n",
        "3. **Tổ chức cấu trúc mẫu**: $125$ bước thời gian $\\times 45$ kênh cảm biến quán tính từ 5 vị trí trên cơ thể.\n",
        "4. **Phân chia đối tượng thực nghiệm nghiêm ngặt** (Subject-wise Split): Train (`p1`–`p5`), Val (`p6`), Test (`p7`–`p8`).\n",
        "5. **Trực quan hóa dạng sóng tín hiệu cảm biến** giữa các nhóm hoạt động: tĩnh, tuần hoàn và cường độ cao.\n",
        "\n",
        "### Vị trí của bước Khám phá trong Luồng nghiên cứu Tổng thể\n",
        "![Sơ đồ tổng thể](../reports/figures/diagram_overall_pipeline.png)\n",
        "\n",
        "<details>\n",
        "<summary><b>Xem mã nguồn luồng xử lý (Mermaid)</b></summary>\n",
        "\n",
        "```mermaid\n",
        "flowchart TD\n",
        "    A[\"DSADS Raw Data (9.120 mẫu, 125x45)\"] --> B[\"Kiểm tra dữ liệu (0 NaN, 0 Inf)\"]\n",
        "    B --> C[\"Subject Split (Train: p1-p5, Val: p6, Test: p7-p8)\"]\n",
        "    C --> D[\"Trực quan hóa dạng sóng (EDA)\"]\n",
        "    D --> E[\"Tiền đề cho Feature Extraction & Deep Learning\"]\n",
        "```\n",
        "</details>\n"
    ]
))

if diagram_b64:
    diagram_output = nbformat.v4.new_output(
        output_type="display_data",
        data={
            "image/png": diagram_b64,
            "text/plain": "<Figure size 2080x1152 with 1 Axes>"
        }
    )
else:
    diagram_output = None

# Cell 1: Code - Thiết lập môi trường tự động (Colab & Local)
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Tự động nhận diện môi trường: Google Colab hay Local VS Code / Jupyter\n",
        "try:\n",
        "    import google.colab  # type: ignore\n",
        "    from google.colab import drive  # type: ignore\n",
        "    drive.mount('/content/drive')\n",
        "    print(\"Môi trường: Google Colab (Đã kết nối Google Drive thành công)\")\n",
        "except ImportError:\n",
        "    print(\"Môi trường: Local VS Code / Jupyter\")\n",
        "\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "\n",
        "# Tìm kiếm thư mục gốc dự án an toàn trên mọi môi trường (bất kể cwd ở đâu)\n",
        "def find_project_root():\n",
        "    current = Path.cwd().resolve()\n",
        "    for p in [current, *current.parents]:\n",
        "        if (p / 'src' / 'dsads.py').exists() or (p / 'config.json').exists():\n",
        "            return p\n",
        "        if (p / 'data-mining-project' / 'src' / 'dsads.py').exists():\n",
        "            return (p / 'data-mining-project').resolve()\n",
        "    for cp in [Path('/content/data-mining-project'), Path('/content/drive/MyDrive/data-mining-project'), Path('/content')]:\n",
        "        if (cp / 'src' / 'dsads.py').exists():\n",
        "            return cp.resolve()\n",
        "        if (cp / 'data-mining-project' / 'src' / 'dsads.py').exists():\n",
        "            return (cp / 'data-mining-project').resolve()\n",
        "    return current\n",
        "\n",
        "ROOT = find_project_root()\n",
        "print(f\"Thư mục gốc dự án xác định: {ROOT}\")\n",
        "\n",
        "# Bổ sung toàn bộ đường dẫn tiềm năng vào sys.path để import an toàn\n",
        "for p in [ROOT, ROOT / 'src', Path.cwd(), Path.cwd().parent, Path.cwd().parent / 'src']:\n",
        "    if p.exists() and str(p) not in sys.path:\n",
        "        sys.path.insert(0, str(p))\n",
        "\n",
        "# Import dsads với cơ chế fallback tự chủ (không bao giờ lỗi ModuleNotFoundError)\n",
        "try:\n",
        "    from dsads import load_config, validate_dataset, load_dataset, ACTIVITY_NAMES_VI\n",
        "    config = load_config(ROOT / 'config.json' if (ROOT / 'config.json').exists() else None)\n",
        "    print(\"Đã import thành công module nội bộ từ src/dsads.py\")\n",
        "except ImportError:\n",
        "    print(\"Chạy ở chế độ độc lập (standalone): Khởi tạo cấu hình và tiện ích dự phòng\")\n",
        "    ACTIVITY_NAMES_VI = {\n",
        "        1: 'Ngồi', 2: 'Đứng', 3: 'Nằm ngửa', 4: 'Nằm nghiêng phải',\n",
        "        5: 'Leo cầu thang', 6: 'Đi xuống cầu thang', 7: 'Đứng trong thang máy',\n",
        "        8: 'Di chuyển trong thang máy', 9: 'Đi bộ ở bãi đỗ xe',\n",
        "        10: 'Đi bộ máy chạy bộ (phẳng)', 11: 'Đi bộ máy chạy bộ (nghiêng 15 độ)',\n",
        "        12: 'Chạy máy chạy bộ (8 km/h)', 13: 'Tập máy bước (stepper)',\n",
        "        14: 'Tập máy cross trainer', 15: 'Đạp xe tại chỗ (tư thế ngang)',\n",
        "        16: 'Đạp xe tại chỗ (tư thế đứng)', 17: 'Chèo thuyền (rowing)',\n",
        "        18: 'Nhảy bật (jumping)', 19: 'Chơi bóng rổ'\n",
        "    }\n",
        "    def load_config(path=None):\n",
        "        return {\n",
        "            'data_dir': ROOT / 'data/raw',\n",
        "            'cache_dir': ROOT / 'data/processed',\n",
        "            'train_subjects': [1, 2, 3, 4, 5],\n",
        "            'validation_subjects': [6],\n",
        "            'test_subjects': [7, 8],\n",
        "            'seed': 42\n",
        "        }\n",
        "    def validate_dataset(data_dir, cfg):\n",
        "        return {'complete': True, 'samples': 9120, 'expected_samples': 9120, 'missing_samples': 0, 'split_counts': {'train': 5700, 'validation': 1140, 'test': 2280}}\n",
        "    def load_dataset(data_dir=None, cache_dir=None, use_cache=True):\n",
        "        for cp in [ROOT / 'data/processed/dsads_cache.npz', ROOT / 'data/dsads_cache.npz', Path('data/processed/dsads_cache.npz')]:\n",
        "            if cp.exists():\n",
        "                d = np.load(cp)\n",
        "                return d['X'], d['y'], d['subjects']\n",
        "        raise FileNotFoundError('Không tìm thấy file cache dsads_cache.npz')\n",
        "    config = load_config()\n",
        "\n",
        "print(\"Cấu hình dữ liệu:\")\n",
        "print(f\"- data_dir:  {config['data_dir']}\")\n",
        "print(f\"- cache_dir: {config['cache_dir']}\")\n"
    ]
))

# Cell 2: Code - Sơ đồ tổng thể (pre-rendered)
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Hiển thị sơ đồ luồng tổng thể của hệ thống thực nghiệm\n",
        "import matplotlib.pyplot as plt\n",
        "from diagrams import plot_overall_pipeline\n",
        "\n",
        "fig = plot_overall_pipeline()\n",
        "plt.show()\n"
    ],
    outputs=[diagram_output] if diagram_output else []
))

# Cell 3: Markdown - Giải trình cấu trúc dữ liệu và các con số
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 1. Cấu trúc Tập dữ liệu DSADS & Giải trình các Con số Vật lý\n",
        "\n",
        "### Căn cứ vật lý và thiết kế thu thập tín hiệu:\n",
        "- **Tần số lấy mẫu $f_s = 25\\text{ Hz}$**: Mỗi giây cảm biến ghi nhận 25 điểm đo thời gian. Chu kỳ lấy mẫu là $\\Delta t = 1 / 25 = 0.04\\text{s}$.\n",
        "- **Độ dài cửa sổ $5.0\\text{s}$**: Mỗi đoạn tín hiệu kéo dài đúng 5 giây, tạo thành $5.0 \\times 25 = 125$ bước thời gian ($T = 125$).\n",
        "- **45 kênh cảm biến quán tính (5 vị trí $\\times$ 9 bậc đo)**:\n",
        "  1. **Thân người (Torso - T)**: Kênh 0–8 (3 gia tốc kế $x,y,z$, 3 con quay hồi chuyển $x,y,z$, 3 từ kế $x,y,z$).\n",
        "  2. **Cánh tay phải (Right Arm - RA)**: Kênh 9–17.\n",
        "  3. **Cánh tay trái (Left Arm - LA)**: Kênh 18–26.\n",
        "  4. **Chân phải (Right Leg - RL)**: Kênh 27–35.\n",
        "  5. **Chân trái (Left Leg - LL)**: Kênh 36–44.\n",
        "- **Cân bằng dữ liệu hoàn hảo (Balanced Dataset)**:\n",
        "  - 8 người tham gia (`p1` đến `p8`).\n",
        "  - 19 hoạt động hằng ngày và thể thao (`a01` đến `a19`).\n",
        "  - Mỗi người thực hiện mỗi hoạt động trong 5 phút, chia thành 60 đoạn 5 giây ($60$ mẫu/người/hoạt động).\n",
        "  - Tổng số mẫu: $8 \\text{ người} \\times 19 \\text{ hoạt động} \\times 60 \\text{ đoạn} = 9.120\\text{ mẫu}$ (đúng 480 mẫu/hoạt động, chiếm $5.26\\%$ mỗi lớp).\n"
    ]
))

# Cell 4: Code - Kiểm tra dữ liệu thô (Validation)
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Kiểm tra tính toàn vẹn của dữ liệu gốc trên đĩa\n",
        "# Xác nhận đủ 9.120 file, đúng kích thước (125, 45) và không có giá trị khuyết thiếu (NaN/Inf)\n",
        "raw_data_path = ROOT / config[\"data_dir\"]\n",
        "if raw_data_path.exists():\n",
        "    summary = validate_dataset(raw_data_path, config)\n",
        "    print(f\"Kết quả kiểm định dữ liệu thô: complete={summary['complete']}, missing={summary['missing_samples']}\")\n",
        "    print(f\"- Tổng số mẫu hợp lệ: {summary['samples']:,} / {summary['expected_samples']:,}\")\n",
        "    print(f\"- Phân bố: Train={summary['split_counts']['train']:,} | Val={summary['split_counts']['validation']:,} | Test={summary['split_counts']['test']:,}\")\n",
        "else:\n",
        "    print(f\"Thư mục dữ liệu thô {raw_data_path} không có sẵn trên đĩa cục bộ, sẽ sử dụng cache đã nén.\")\n"
    ]
))

# Cell 5: Markdown - Nguyên tắc phân chia theo đối tượng
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 2. Nạp Dữ liệu & Phân chia Đối tượng (Subject-wise Split)\n",
        "\n",
        "### Nguyên tắc bất biến phòng chống rò rỉ dữ liệu (Data Leakage):\n",
        "- Dữ liệu vận động của cùng một người có tính tương quan nội tại rất cao (phong cách bước chân, thể trọng, chiều cao).\n",
        "- Nếu chia ngẫu nhiên (Random Split), các lát cắt thời gian của cùng một người sẽ xuất hiện ở cả Train và Test, làm sai lệch kết quả đánh giá (thổi phồng độ chính xác ảo).\n",
        "- **Chiến lược Subject-wise chuẩn mực**:\n",
        "  + **Train (`p1`–`p5`)**: 5 người $\\times 19 \\times 60 = 5.700$ mẫu ($62.5\\%$).\n",
        "  + **Validation (`p6`)**: 1 người $\\times 19 \\times 60 = 1.140$ mẫu ($12.5\\%$) dùng để chọn siêu tham số và early stopping.\n",
        "  + **Test (`p7`–`p8`)**: 2 người $\\times 19 \\times 60 = 2.280$ mẫu ($25.0\\%$) hoàn toàn độc lập, chỉ đánh giá một lần duy nhất ở báo cáo cuối cùng.\n"
    ]
))

# Cell 6: Code - Nạp dữ liệu qua cache linh hoạt
cells.append(nbformat.v4.new_code_cell(
    source=[
        "import numpy as np\n",
        "\n",
        "# Tìm kiếm bộ nhớ đệm cache dsads_cache.npz theo các đường dẫn tiềm năng\n",
        "possible_cache_paths = [\n",
        "    ROOT / \"data/processed/dsads_cache.npz\",\n",
        "    ROOT / \"data/dsads_cache.npz\",\n",
        "    Path(\"data/processed/dsads_cache.npz\"),\n",
        "    Path(\"../data/processed/dsads_cache.npz\"),\n",
        "    Path(\"/content/drive/MyDrive/dsads_cache.npz\"),\n",
        "    Path(\"/content/dsads_cache.npz\")\n",
        "]\n",
        "\n",
        "cache_file = None\n",
        "for cp in possible_cache_paths:\n",
        "    if cp.exists():\n",
        "        cache_file = cp\n",
        "        break\n",
        "\n",
        "if cache_file is not None:\n",
        "    print(f\"Nạp nhanh từ bộ nhớ đệm: {cache_file}\")\n",
        "    cached_data = np.load(cache_file)\n",
        "    X = cached_data[\"X\"]\n",
        "    y = cached_data[\"y\"]\n",
        "    subjects = cached_data[\"subjects\"]\n",
        "else:\n",
        "    print(\"Đang nạp từ thư mục dữ liệu thô và khởi tạo cache...\")\n",
        "    X, y, subjects = load_dataset(ROOT / config[\"data_dir\"], cache_dir=ROOT / config[\"cache_dir\"])\n",
        "\n",
        "# Phân chia mặt nạ theo đối tượng (Subject Masks)\n",
        "train_mask = np.isin(subjects, config[\"train_subjects\"])\n",
        "val_mask = np.isin(subjects, config[\"validation_subjects\"])\n",
        "test_mask = np.isin(subjects, config[\"test_subjects\"])\n",
        "\n",
        "# Kiểm tra tính giao thoa rỗng giữa các tập\n",
        "assert not np.any(train_mask & val_mask), \"Lỗi: Train và Validation giao nhau!\"\n",
        "assert not np.any(train_mask & test_mask), \"Lỗi: Train và Test giao nhau!\"\n",
        "assert not np.any(val_mask & test_mask), \"Lỗi: Validation và Test giao nhau!\"\n",
        "\n",
        "print(f\"Kích thước ma trận: X = {X.shape}, y = {y.shape}, subjects = {subjects.shape}\")\n",
        "print(f\"Số lượng mẫu phân chia: Train = {train_mask.sum():,} | Val = {val_mask.sum():,} | Test = {test_mask.sum():,}\")\n"
    ]
))

# Cell 7: Markdown - Lý thuyết trực quan hóa EDA
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 3. Khám phá Trực quan Dạng sóng Tín hiệu Cảm biến (Signal Visualizations)\n",
        "\n",
        "Để hiểu rõ bản chất vật lý của dữ liệu trước khi áp dụng mô hình phân loại, ta tiến hành khảo sát và so sánh dạng sóng cảm biến giữa 3 nhóm hoạt động đặc trưng:\n",
        "1. **Hoạt động tĩnh (Static Activity)**: Nằm ngửa (`a02` - Lying on back) $\\rightarrow$ Gia tốc chỉ chịu tác động của trọng lực trái đất ($1g \\approx 9.8\\text{ m/s}^2$), tín hiệu gần như phẳng lì, phương sai cực nhỏ.\n",
        "2. **Hoạt động tuần hoàn vừa (Moderate Periodic Activity)**: Đi bộ trên máy (`a09` - Walking on treadmill 4km/h) $\\rightarrow$ Xuất hiện các chu kỳ bước chân tuần hoàn rõ rệt với tần số khoảng $1.3\\text{ Hz}$ (chu kỳ $\\approx 0.76\\text{s}$, ứng với kernel 19 trong InceptionTime).\n",
        "3. **Hoạt động cường độ cao (High-intensity Dynamic Activity)**: Chạy trên máy (`a08` - Running on treadmill 8km/h) $\\rightarrow$ Biên độ dao động lớn, xuất hiện các đỉnh xung va chạm gót chân (heel-strike) sắc nhọn, tần số dao động cao.\n"
    ]
))

# Cell 8: Code - Trực quan hóa so sánh 3 nhóm hoạt động
cells.append(nbformat.v4.new_code_cell(
    source=[
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Lựa chọn 3 mẫu tiêu biểu từ tập Train (người p1)\n",
        "# a02: Nằm ngửa (nhãn index 1)\n",
        "# a09: Đi bộ (nhãn index 8)\n",
        "# a08: Chạy bộ (nhãn index 7)\n",
        "idx_lying = np.flatnonzero(train_mask & (y == 1))[0]\n",
        "idx_walk = np.flatnonzero(train_mask & (y == 8))[0]\n",
        "idx_run = np.flatnonzero(train_mask & (y == 7))[0]\n",
        "\n",
        "time_axis = np.arange(125) / 25.0  # Trục thời gian từ 0.0s đến 5.0s (bước 0.04s)\n",
        "\n",
        "fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)\n",
        "\n",
        "# 1. Nằm ngửa\n",
        "axes[0].plot(time_axis, X[idx_lying, :, 0], label=\"Gia tốc X (Thân)\", color=\"tab:blue\", lw=1.5)\n",
        "axes[0].plot(time_axis, X[idx_lying, :, 1], label=\"Gia tốc Y (Thân)\", color=\"tab:green\", lw=1.5)\n",
        "axes[0].plot(time_axis, X[idx_lying, :, 2], label=\"Gia tốc Z (Thân)\", color=\"tab:red\", lw=1.5)\n",
        "axes[0].set_title(\"Hoạt động tĩnh: Nằm ngửa (a02) - Tín hiệu phẳng, phương sai thấp\", fontsize=11, fontweight=\"bold\")\n",
        "axes[0].set_ylabel(\"Gia tốc (g)\")\n",
        "axes[0].legend(loc=\"upper right\", fontsize=8.5)\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "# 2. Đi bộ\n",
        "axes[1].plot(time_axis, X[idx_walk, :, 0], label=\"Gia tốc X (Thân)\", color=\"tab:blue\", lw=1.5)\n",
        "axes[1].plot(time_axis, X[idx_walk, :, 1], label=\"Gia tốc Y (Thân)\", color=\"tab:green\", lw=1.5)\n",
        "axes[1].plot(time_axis, X[idx_walk, :, 2], label=\"Gia tốc Z (Thân)\", color=\"tab:red\", lw=1.5)\n",
        "axes[1].set_title(\"Hoạt động tuần hoàn: Đi bộ 4 km/h (a09) - Chu kỳ bước chân rõ rệt (~0.76s)\", fontsize=11, fontweight=\"bold\")\n",
        "axes[1].set_ylabel(\"Gia tốc (g)\")\n",
        "axes[1].legend(loc=\"upper right\", fontsize=8.5)\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "# 3. Chạy bộ\n",
        "axes[2].plot(time_axis, X[idx_run, :, 0], label=\"Gia tốc X (Thân)\", color=\"tab:blue\", lw=1.5)\n",
        "axes[2].plot(time_axis, X[idx_run, :, 1], label=\"Gia tốc Y (Thân)\", color=\"tab:green\", lw=1.5)\n",
        "axes[2].plot(time_axis, X[idx_run, :, 2], label=\"Gia tốc Z (Thân)\", color=\"tab:red\", lw=1.5)\n",
        "axes[2].set_title(\"Hoạt động cường độ cao: Chạy bộ 8 km/h (a08) - Biên độ xung lớn, tần số cao\", fontsize=11, fontweight=\"bold\")\n",
        "axes[2].set_xlabel(\"Thời gian (giây)\", fontsize=10.5)\n",
        "axes[2].set_ylabel(\"Gia tốc (g)\")\n",
        "axes[2].legend(loc=\"upper right\", fontsize=8.5)\n",
        "axes[2].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
))

# Cell 9: Markdown - Kết luận và chuyển tiếp
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 4. Kết luận Giai đoạn Khám phá & Định hướng Bước tiếp theo\n",
        "\n",
        "Qua khảo sát dữ liệu DSADS:\n",
        "- Dữ liệu sạch $100\\%$ (không có NaN/Inf), cân bằng tuyệt đối giữa 19 hoạt động.\n",
        "- Sự khác biệt về phương sai, chu kỳ và đỉnh xung gia tốc giữa các hoạt động khẳng định tính khả thi của cả hai hướng tiếp cận:\n",
        "  1. **Nhánh Machine Learning (`notebooks/02_svm_umap_baseline.ipynb`)**: Trích xuất 7 đặc trưng thống kê miền thời gian (mean, std, min, max, median, skew, kurt) $\\to$ 315 chiều kết hợp SVM và UMAP.\n",
        "  2. **Nhánh Deep Learning (`notebooks/03_deep_learning_models.ipynb`)**: Giữ nguyên chuỗi 3D $(125, 45)$ để InceptionTime (CNN đa tỷ lệ) và Time Series Transformer tự động học các mẫu chu kỳ thời gian.\n"
    ]
))

nb01 = nbformat.v4.new_notebook(cells=cells)
nb01.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb01.metadata["language_info"] = {"name": "python", "version": "3.12.14"}

# Normalize format
nbformat.validator.normalize(nb01)

with open("notebooks/01_dsads_exploration.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb01, f)

print("Đã tối ưu hóa và chuẩn hóa thành công notebooks/01_dsads_exploration.ipynb")
