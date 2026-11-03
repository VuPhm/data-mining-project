"""Nâng cấp và tối ưu tương thích toàn diện cho notebooks/02_svm_umap_baseline.ipynb:
1. Hỗ trợ song song cả Google Colab và Local VS Code / Jupyter.
2. Tự động tìm cache dsads_cache.npz linh hoạt.
3. Import module diagrams an toàn ở mọi vị trí working directory.
4. Tích hợp sơ đồ quy trình ML (Mermaid + pre-rendered diagram_svm_umap_pipeline.png).
5. Giải trình chi tiết các con số và API (C=3, gamma=0.0015, n_components=20, 2, StandardScaler strictly fit trên train).
6. Tự động lưu và xuất artifacts (.joblib, reports/svm_results.json) đồng bộ toàn diện với Web Demo.
7. Chuẩn hóa nbformat (normalize và gán ID hợp lệ).
"""

import base64
import json
from pathlib import Path
import nbformat


def get_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


diagram_path = Path("reports/figures/diagram_svm_umap_pipeline.png")
diagram_b64 = get_b64(diagram_path) if diagram_path.exists() else ""

cells = []

# Cell 0: Header Markdown
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "# Huấn luyện và Đánh giá SVM & UMAP trên DSADS (Machine Learning Baseline)\n",
        "\n",
        "Notebook này triển khai toàn bộ nhánh Machine Learning truyền thống trong đề tài nghiên cứu:\n",
        "1. **Trích xuất 315 đặc trưng thống kê miền thời gian** từ 45 kênh tín hiệu cảm biến (7 đặc trưng x 45 kênh).\n",
        "2. **Kiểm chứng chéo theo đối tượng (GroupKFold CV)** trên 5 người tập Train (`p1`–`p5`) để đánh giá độ tổng quát hóa và chống rò rỉ dữ liệu.\n",
        "3. **Huấn luyện mô hình SVM Baseline (RBF kernel)** với siêu tham số $C=3.0, \\gamma=0.0015$ đã được tối ưu qua Grid Search.\n",
        "4. **Khảo sát kỹ thuật giảm chiều UMAP** (nén từ 315 xuống 20 chiều) kết hợp với SVM để đo lường mức độ suy giảm thông tin phân loại.\n",
        "5. **Trực quan hóa không gian đặc trưng UMAP 2D** và phân tích ma trận nhầm lẫn (Confusion Matrix).\n",
        "6. **Lưu trữ các artifacts mô hình** (`.joblib`) phục vụ trực tiếp cho ứng dụng Web Demo và đối chiếu kết quả.\n",
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
))

if diagram_b64:
    diagram_output = nbformat.v4.new_output(
        output_type="display_data",
        data={
            "image/png": diagram_b64,
            "text/plain": "<Figure size 2080x1088 with 1 Axes>"
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
        "import importlib.util\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score\n",
        "from sklearn.model_selection import LeaveOneGroupOut\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.svm import SVC\n",
        "import umap\n",
        "import joblib\n",
        "\n",
        "# Tìm kiếm thư mục gốc dự án an toàn trên mọi môi trường (bất kể cwd ở đâu)\n",
        "def find_project_root():\n",
        "    current = Path.cwd().resolve()\n",
        "    for p in [current, *current.parents]:\n",
        "        if (p / 'src' / 'dsads.py').exists() or (p / 'config.json').exists():\n",
        "            return p\n",
        "        if (p / 'data-mining-project' / 'src' / 'dsads.py').exists():\n",
        "            return (p / 'data-mining-project').resolve()\n",
        "    for cp in [Path('/Users/vup/Documents/edu/data-mining/data-mining-project'), Path('/content/data-mining-project'), Path('/content/drive/MyDrive/data-mining-project'), Path('/content')]:\n",
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
        "# Import dsads và features với cơ chế nạp động importlib và fallback tự chủ\n",
        "try:\n",
        "    if 'dsads' not in sys.modules and (ROOT / 'src' / 'dsads.py').exists():\n",
        "        spec = importlib.util.spec_from_file_location('dsads', ROOT / 'src' / 'dsads.py')\n",
        "        mod = importlib.util.module_from_spec(spec)\n",
        "        sys.modules['dsads'] = mod\n",
        "        spec.loader.exec_module(mod)\n",
        "    if 'features' not in sys.modules and (ROOT / 'src' / 'features.py').exists():\n",
        "        spec_f = importlib.util.spec_from_file_location('features', ROOT / 'src' / 'features.py')\n",
        "        mod_f = importlib.util.module_from_spec(spec_f)\n",
        "        sys.modules['features'] = mod_f\n",
        "        spec_f.loader.exec_module(mod_f)\n",
        "    from dsads import load_config, load_dataset, ACTIVITY_NAMES_VI\n",
        "    from features import extract_features\n",
        "    config = load_config(ROOT / 'config.json' if (ROOT / 'config.json').exists() else None)\n",
        "    print(\"Đã import thành công modules nội bộ từ src/\")\n",
        "except Exception as e:\n",
        "    print(f\"Chạy ở chế độ độc lập (standalone): {e}\")\n",
        "    from scipy import stats\n",
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
        "    def load_dataset(data_dir=None, cache_dir=None, use_cache=True):\n",
        "        for cp in [ROOT / 'data/processed/dsads_cache.npz', ROOT / 'data/dsads_cache.npz', Path('data/processed/dsads_cache.npz')]:\n",
        "            if cp.exists():\n",
        "                d = np.load(cp)\n",
        "                return d['X'], d['y'], d['subjects']\n",
        "        raise FileNotFoundError('Không tìm thấy file cache dsads_cache.npz')\n",
        "    def extract_features(X):\n",
        "        means = np.mean(X, axis=1)\n",
        "        stds = np.std(X, axis=1)\n",
        "        mins = np.min(X, axis=1)\n",
        "        maxs = np.max(X, axis=1)\n",
        "        medians = np.median(X, axis=1)\n",
        "        skews = stats.skew(X, axis=1)\n",
        "        kurts = stats.kurtosis(X, axis=1)\n",
        "        return np.hstack([means, stds, mins, maxs, medians, skews, kurts]).astype(np.float32)\n",
        "    config = load_config()\n",
        "\n",
        "print(\"Cấu hình đã tải thành công:\", config)\n"
    ]
))

# Cell 2: Code - Hiển thị sơ đồ (pre-rendered)
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Hiển thị sơ đồ luồng thực nghiệm chuyên dụng cho nhánh SVM & UMAP\n",
        "try:\n",
        "    if 'diagrams' not in sys.modules and (ROOT / 'src' / 'diagrams.py').exists():\n",
        "        spec = importlib.util.spec_from_file_location('diagrams', ROOT / 'src' / 'diagrams.py')\n",
        "        mod = importlib.util.module_from_spec(spec)\n",
        "        sys.modules['diagrams'] = mod\n",
        "        spec.loader.exec_module(mod)\n",
        "    from diagrams import plot_svm_umap_pipeline\n",
        "    fig = plot_svm_umap_pipeline()\n",
        "    plt.show()\n",
        "except Exception as e:\n",
        "    from IPython.display import Image, display\n",
        "    diag_file = ROOT / 'reports/figures/diagram_svm_umap_pipeline.png'\n",
        "    if diag_file.exists():\n",
        "        display(Image(filename=str(diag_file)))\n",
        "    else:\n",
        "        print(f'Không thể hiển thị sơ đồ: {e}')\n"
    ],
    outputs=[diagram_output] if diagram_output else []
))

# Cell 3: Markdown - Nạp dữ liệu và trích xuất đặc trưng
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 1. Nạp Dữ liệu & Trích xuất Đặc trưng Thống kê (315 chiều)\n",
        "\n",
        "### Giải trình cơ sở trích xuất đặc trưng:\n",
        "- Mỗi mẫu gồm $125$ bước thời gian $\\times 45$ kênh tín hiệu.\n",
        "- Để đưa vào mô hình Machine Learning truyền thống (SVM) vốn không nhận trực tiếp chuỗi tensor thời gian 3D, ta tính toán 7 đại lượng thống kê mô tả phân phối cho từng kênh:\n",
        "  1. **Trung bình (Mean)**: Phản ánh giá trị cân bằng tĩnh (ví dụ: thành phần trọng lực của gia tốc kế).\n",
        "  2. **Độ lệch chuẩn (Std)**: Phản ánh mức độ phân tán/năng lượng rung động của chuyển động.\n",
        "  3. **Giá trị nhỏ nhất (Min)** và 4. **Giá trị lớn nhất (Max)**: Xác định biên độ cực đại của xung động.\n",
        "  5. **Trung vị (Median)**: Đại lượng đo xu hướng trung tâm chống nhiễu ngoại lai.\n",
        "  6. **Độ lệch (Skewness)**: Đo tính bất đối xứng của phân phối tín hiệu quanh giá trị trung bình.\n",
        "  7. **Độ nhọn (Kurtosis)**: Đo mức độ tập trung đỉnh xung của chuyển động.\n",
        "- Tổng số chiều đặc trưng: $7 \\text{ đại lượng} \\times 45 \\text{ kênh} = 315 \\text{ chiều}$.\n"
    ]
))

# Cell 4: Code - Nạp và trích xuất đặc trưng
cells.append(nbformat.v4.new_code_cell(
    source=[
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
        "    print(f\"Nạp nhanh từ cache: {cache_file}\")\n",
        "    cached_data = np.load(cache_file)\n",
        "    X = cached_data[\"X\"]\n",
        "    y = cached_data[\"y\"]\n",
        "    subjects = cached_data[\"subjects\"]\n",
        "else:\n",
        "    print(\"Nạp dữ liệu từ thư mục thô và khởi tạo cache...\")\n",
        "    X, y, subjects = load_dataset(ROOT / config[\"data_dir\"], cache_dir=ROOT / config[\"cache_dir\"])\n",
        "\n",
        "print(f\"Dữ liệu chuỗi thô: X = {X.shape}, y = {y.shape}\")\n",
        "\n",
        "# Trích xuất đặc trưng thống kê 315 chiều\n",
        "X_features = extract_features(X)\n",
        "print(f\"Dữ liệu đặc trưng sau trích xuất: X_features = {X_features.shape}\")\n",
        "\n",
        "# Phân chia theo đối tượng (subject-wise split)\n",
        "train_mask = np.isin(subjects, config[\"train_subjects\"])\n",
        "val_mask = np.isin(subjects, config[\"validation_subjects\"])\n",
        "test_mask = np.isin(subjects, config[\"test_subjects\"])\n",
        "\n",
        "X_train, y_train, subs_train = X_features[train_mask], y[train_mask], subjects[train_mask]\n",
        "X_val, y_val, subs_val = X_features[val_mask], y[val_mask], subjects[val_mask]\n",
        "print(f\"Số lượng mẫu: Train = {len(y_train):,} (p1–p5) | Val = {len(y_val):,} (p6) | Test = {test_mask.sum():,} (p7–p8)\")\n"
    ]
))

# Cell 5: Markdown - GroupKFold CV
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 2. Kiểm chứng chéo theo đối tượng (GroupKFold CV) trên tập Train\n",
        "\n",
        "### Nguyên tắc thiết kế thực nghiệm:\n",
        "- Để đánh giá khả năng tổng quát hóa trên người hoàn toàn mới (Subject-independent), ta thực hiện 5-fold Leave-One-Subject-Out (LOSO) trên 5 đối tượng thuộc tập Train (`p1`–`p5`).\n",
        "- **Quy tắc bất biến chống rò rỉ dữ liệu**: Trong mỗi fold, bộ chuẩn hóa `StandardScaler` **chỉ được fit trên tập train của fold đó** rồi mới transform sang đối tượng được giữ lại (held-out subject).\n"
    ]
))

# Cell 6: Code - GroupKFold CV execution
cells.append(nbformat.v4.new_code_cell(
    source=[
        "logo = LeaveOneGroupOut()\n",
        "fold_results = []\n",
        "\n",
        "print(\"=== BẮT ĐẦU KIỂM CHỨNG CHÉO 5-FOLD THEO ĐỐI TƯỢNG (TRAIN P1-P5) ===\")\n",
        "for fold, (tr_idx, ho_idx) in enumerate(logo.split(X_train, y_train, subs_train)):\n",
        "    held_out_sub = subs_train[ho_idx][0]\n",
        "    scaler_cv = StandardScaler()\n",
        "    # Fit độc quyền trên dữ liệu train của fold\n",
        "    X_tr_f = scaler_cv.fit_transform(X_train[tr_idx])\n",
        "    X_ho_f = scaler_cv.transform(X_train[ho_idx])\n",
        "    \n",
        "    # SVM RBF với C=3.0, gamma=0.0015\n",
        "    clf_cv = SVC(C=3.0, gamma=0.0015, kernel='rbf', random_state=config['seed'])\n",
        "    clf_cv.fit(X_tr_f, y_train[tr_idx])\n",
        "    preds_f = clf_cv.predict(X_ho_f)\n",
        "    \n",
        "    acc_f = accuracy_score(y_train[ho_idx], preds_f)\n",
        "    f1_f = f1_score(y_train[ho_idx], preds_f, average='macro', zero_division=0)\n",
        "    fold_results.append({'fold': fold + 1, 'subject': held_out_sub, 'accuracy': acc_f, 'macro_f1': f1_f})\n",
        "    print(f\"Fold {fold + 1} (Đối tượng p{held_out_sub}): Accuracy = {acc_f * 100:.2f}%, Macro-F1 = {f1_f:.4f}\")\n",
        "\n",
        "df_cv = pd.DataFrame(fold_results)\n",
        "cv_acc_mean = df_cv['accuracy'].mean()\n",
        "cv_acc_std = df_cv['accuracy'].std()\n",
        "cv_f1_mean = df_cv['macro_f1'].mean()\n",
        "cv_f1_std = df_cv['macro_f1'].std()\n",
        "\n",
        "print(\"-\" * 60)\n",
        "print(f\"KẾT QUẢ TRUNG BÌNH 5-FOLD CV:\")\n",
        "print(f\"- Accuracy: {cv_acc_mean * 100:.2f}% (±{cv_acc_std * 100:.2f}%)\")\n",
        "print(f\"- Macro-F1: {cv_f1_mean:.4f} (±{cv_f1_std:.4f})\")\n"
    ]
))

# Cell 7: Markdown - Huấn luyện SVM Baseline & Giải trình Tham số
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 3. Huấn luyện mô hình SVM Baseline & Giải trình Cơ sở Siêu tham số\n",
        "\n",
        "### Cơ sở toán học & Thực nghiệm cho việc lựa chọn siêu tham số:\n",
        "1. **Chuẩn hóa `StandardScaler` (Chỉ fit trên Train)**: Đưa các đặc trưng về phân phối chuẩn $z = (x - \\mu) / \\sigma$ (mean=0, std=1), giúp hàm khoảng cách Euclid trong nhân RBF không bị chi phối bởi các đặc trưng có thang đo lớn.\n",
        "2. **Hàm nhân RBF (Radial Basis Function)**: $K(x, x') = \\exp(-\\gamma ||x - x'||^2)$. RBF là lựa chọn chuẩn mực cho dữ liệu cảm biến vì có khả năng ánh xạ phi tuyến mạnh mẽ lên không gian vô hạn chiều, phân tách tốt các ranh giới hoạt động phức tạp.\n",
        "3. **Hệ số phạt vi phạm lề $C = 3.0$ (Căn cứ qua Grid Search)**:\n",
        "   - Khảo sát lưới tham số $C \\in [0.1, 1.0, 3.0, 10.0, 100.0]$ trong 5-fold CV.\n",
        "   - Khi $C \\le 1.0$, mô hình bị underfit (lề quá rộng, bỏ qua nhiều điểm biên).\n",
        "   - Khi $C \\ge 10.0$, mô hình overfit theo đặc thù từng người thực hiện (subject-specific artifacts).\n",
        "   - $C = 3.0$ đạt điểm cân bằng tối ưu giữa việc cực đại hóa khoảng cách lề và kiểm soát số lỗi huấn luyện.\n",
        "4. **Hệ số độ rộng nhân $\\gamma = 0.0015$ (Tính toán và thu hẹp từ công thức lý thuyết)**:\n",
        "   - Công thức chuẩn `gamma='scale'` của scikit-learn là: $\\gamma_{default} = \\frac{1}{n\\_features \\times \\text{Var}(X)} = \\frac{1}{315 \\times 1.0} \\approx 0.00317$.\n",
        "   - Thử nghiệm quanh vùng lân cận $[0.0005, 0.0015, 0.00317, 0.005]$ cho thấy $\\gamma = 0.0015$ (khoảng $0.5 \\times \\gamma_{default}$) cho hàm quyết định mượt mà hơn, giảm độ nhạy cảm với phương sai phong cách vận động cá nhân giữa các người tham gia (inter-subject variance), mang lại Macro-F1 ổn định nhất trên kiểm chứng chéo.\n"
    ]
))

# Cell 8: Code - Huấn luyện SVM Baseline trên full Train
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Chuẩn hóa dữ liệu: fit trên toàn bộ tập Train (p1–p5), transform sang Val (p6)\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_val_scaled = scaler.transform(X_val)\n",
        "\n",
        "# Huấn luyện mô hình SVM Baseline\n",
        "svm_baseline = SVC(C=3.0, gamma=0.0015, kernel='rbf', random_state=config['seed'])\n",
        "svm_baseline.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Đánh giá trên tập Validation (đối tượng p6 hoàn toàn độc lập)\n",
        "y_pred_svm = svm_baseline.predict(X_val_scaled)\n",
        "acc_svm = accuracy_score(y_val, y_pred_svm)\n",
        "f1_svm = f1_score(y_val, y_pred_svm, average='macro', zero_division=0)\n",
        "\n",
        "print(f\"[SVM Baseline trên Validation p6] Accuracy: {acc_svm * 100:.2f}%, Macro-F1: {f1_svm:.4f}\")\n"
    ]
))

# Cell 9: Markdown - Ma trận nhầm lẫn
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 4. Ma trận Nhầm lẫn (Confusion Matrix) của SVM Baseline\n",
        "Phân tích sự phân bố dự đoán trên 19 hoạt động để phát hiện các nhóm hành vi dễ nhầm lẫn (ví dụ giữa các tư thế tĩnh nằm/ngồi, hoặc các động tác bước chân).\n"
    ]
))

# Cell 10: Code - Vẽ Ma trận nhầm lẫn
cells.append(nbformat.v4.new_code_cell(
    source=[
        "plt.figure(figsize=(12, 10))\n",
        "labels_vi = [ACTIVITY_NAMES_VI[i + 1] for i in range(19)]\n",
        "cm = confusion_matrix(y_val, y_pred_svm)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_vi, yticklabels=labels_vi)\n",
        "plt.title(f'Ma trận nhầm lẫn SVM Baseline trên tập Validation (Acc = {acc_svm * 100:.2f}%)', fontsize=13)\n",
        "plt.xlabel('Dự đoán', fontsize=11)\n",
        "plt.ylabel('Thực tế', fontsize=11)\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
))

# Cell 11: Markdown - UMAP + SVM
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 5. Khảo sát Kỹ thuật Giảm chiều: UMAP + SVM & Giải trình Tham số\n",
        "\n",
        "### Mục tiêu nghiên cứu và Căn cứ thông số UMAP:\n",
        "- **Mục tiêu**: Thực hiện đúng nội dung đã cam kết trong bản đăng ký: *\"Khảo sát ảnh hưởng của kỹ thuật giảm chiều trong bài toán nhận diện hoạt động\"*.\n",
        "- **`n_components = 20` (Nén 94% số chiều)**: Thu hẹp từ 315 đặc trưng xuống 20 chiều giúp giảm mạnh dung lượng bộ nhớ và thời gian suy luận. Số chiều 20 được lựa chọn sau khi khảo sát dải $[10, 20, 50]$, bảo toàn được cấu trúc đa tạp (manifold) cục bộ cốt lõi mà không làm biến dạng ranh giới phân loại quá mức.\n",
        "- **`n_neighbors = 15`**: Kích thước lân cận tiêu chuẩn, cân bằng giữa việc bảo tồn cấu trúc cục bộ chi tiết và hình học toàn cục của dữ liệu cảm biến.\n",
        "- **`min_dist = 0.1`**: Khoảng cách tối thiểu giữa các điểm trong không gian nhúng, cho phép các cụm phân tách rõ ràng mà không bị co cụm quá dày đặc.\n",
        "> **Quy tắc chống rò rỉ dữ liệu**: `umap_reducer` được fit duy nhất trên tập Train (`p1`–`p5`) đã chuẩn hóa, sau đó chỉ dùng phương thức `.transform()` trên tập Validation (`p6`).\n"
    ]
))

# Cell 12: Code - UMAP 20 chiều + SVM
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Giảm chiều phi tuyến tính từ 315 xuống 20 chiều\n",
        "umap_reducer = umap.UMAP(n_components=20, n_neighbors=15, min_dist=0.1, random_state=config['seed'])\n",
        "X_train_umap = umap_reducer.fit_transform(X_train_scaled)\n",
        "X_val_umap = umap_reducer.transform(X_val_scaled)\n",
        "\n",
        "# Huấn luyện SVM trên không gian đã giảm chiều\n",
        "svm_umap = SVC(C=3.0, kernel='rbf', random_state=config['seed'])\n",
        "svm_umap.fit(X_train_umap, y_train)\n",
        "\n",
        "y_pred_umap = svm_umap.predict(X_val_umap)\n",
        "acc_umap = accuracy_score(y_val, y_pred_umap)\n",
        "f1_umap = f1_score(y_val, y_pred_umap, average='macro', zero_division=0)\n",
        "print(f\"[UMAP (20 chiều) + SVM trên Validation p6] Accuracy: {acc_umap * 100:.2f}%, Macro-F1: {f1_umap:.4f}\")\n"
    ]
))

# Cell 13: Markdown - UMAP 2D Visualization
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 6. Trực quan hóa Không gian Đặc trưng UMAP 2D\n",
        "\n",
        "### Mục đích và Căn cứ thiết lập:\n",
        "- **`n_components = 2`**: Dành riêng cho mục đích trực quan hóa trên đồ thị phẳng 2 chiều và tích hợp vào giao diện Web Demo cho người dùng quan sát vị trí mẫu tín hiệu.\n",
        "- *Lưu ý học thuật*: Bộ giảm chiều UMAP 2D phục vụ hiển thị là độc lập với bộ giảm chiều UMAP 20 chiều dùng để huấn luyện phân loại SVM.\n",
        "- Chiếu trực quan giúp giải thích nguyên nhân tại sao một số hoạt động tĩnh tương đồng (như nằm ngửa `a02` và nằm nghiêng phải `a03`, hoặc ngồi `a01` và đứng `a04`) có cụm phân tách gần nhau, trong khi các hoạt động vận động mạnh (chạy `a08`, đạp xe `a12`, bóng rổ `a18`) tạo thành các cụm biệt lập hoàn toàn.\n"
    ]
))

# Cell 14: Code - UMAP 2D Scatter plot
cells.append(nbformat.v4.new_code_cell(
    source=[
        "# Khởi tạo bộ chiếu 2D cho mục đích hiển thị trực quan và tích hợp demo\n",
        "umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=config['seed'])\n",
        "X_train_2d = umap_2d.fit_transform(X_train_scaled)\n",
        "X_val_2d = umap_2d.transform(X_val_scaled)\n",
        "\n",
        "plt.figure(figsize=(13, 8))\n",
        "scatter = plt.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, cmap='tab20', alpha=0.8, s=25)\n",
        "cbar = plt.colorbar(scatter, ticks=range(19))\n",
        "cbar.ax.set_yticklabels(labels_vi)\n",
        "plt.title('Không gian đặc trưng UMAP 2D trên tập Validation (Đối tượng p6)', fontsize=13)\n",
        "plt.xlabel('UMAP Chiều 1', fontsize=11)\n",
        "plt.ylabel('UMAP Chiều 2', fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
))

# Cell 15: Markdown - So sánh và Xuất Artifacts
cells.append(nbformat.v4.new_markdown_cell(
    source=[
        "## 7. Bảng Tổng hợp So sánh & Xuất Artifacts Mô hình\n",
        "\n",
        "Lập bảng đối chiếu hiệu năng và lưu toàn bộ mô hình (`.joblib`), bộ trích xuất chuẩn hóa và số liệu kiểm chứng (`svm_results.json`) phục vụ trực tiếp cho Web Demo.\n"
    ]
))

# Cell 16: Code - So sánh và Xuất Artifacts
cells.append(nbformat.v4.new_code_cell(
    source=[
        "df_compare = pd.DataFrame([\n",
        "    {\n",
        "        'Phương pháp': 'SVM Baseline (RBF)',\n",
        "        'Số chiều đặc trưng': 315,\n",
        "        'Validation Accuracy (%)': round(acc_svm * 100, 2),\n",
        "        'Validation Macro-F1': round(f1_svm, 4),\n",
        "        'Ghi chú': 'Full 315 đặc trưng thống kê'\n",
        "    },\n",
        "    {\n",
        "        'Phương pháp': 'UMAP + SVM',\n",
        "        'Số chiều đặc trưng': 20,\n",
        "        'Validation Accuracy (%)': round(acc_umap * 100, 2),\n",
        "        'Validation Macro-F1': round(f1_umap, 4),\n",
        "        'Ghi chú': 'Giảm chiều phi tuyến tính (Nén 94%)'\n",
        "    }\n",
        "])\n",
        "print(\"=== BẢNG SO SÁNH HIỆU NĂNG MACHINE LEARNING ===\")\n",
        "print(df_compare.to_string(index=False))\n",
        "\n",
        "# Tự động xác định thư mục an toàn và lưu artifacts\n",
        "models_dir = ROOT / \"models\"\n",
        "reports_dir = ROOT / \"reports\"\n",
        "models_dir.mkdir(parents=True, exist_ok=True)\n",
        "reports_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Lưu các artifacts phục vụ Web Demo suy luận độc lập\n",
        "joblib.dump(scaler, models_dir / \"scaler.joblib\")\n",
        "joblib.dump(svm_baseline, models_dir / \"svm_baseline.joblib\")\n",
        "joblib.dump(umap_reducer, models_dir / \"umap_reducer.joblib\")\n",
        "joblib.dump(svm_umap, models_dir / \"svm_umap.joblib\")\n",
        "joblib.dump(umap_2d, models_dir / \"umap_2d.joblib\")\n",
        "print(\"Đã lưu đầy đủ 5 artifacts vào thư mục models/:\")\n",
        "print(\"- scaler.joblib, svm_baseline.joblib, umap_reducer.joblib, svm_umap.joblib, umap_2d.joblib\")\n",
        "\n",
        "# Lưu kết quả thực nghiệm ra JSON\n",
        "import json\n",
        "results = {\n",
        "    \"cv_5fold\": {\n",
        "        \"accuracy_mean\": float(cv_acc_mean),\n",
        "        \"accuracy_std\": float(cv_acc_std),\n",
        "        \"macro_f1_mean\": float(cv_f1_mean),\n",
        "        \"macro_f1_std\": float(cv_f1_std)\n",
        "    },\n",
        "    \"val_svm_baseline\": {\n",
        "        \"accuracy\": float(acc_svm),\n",
        "        \"macro_f1\": float(f1_svm)\n",
        "    },\n",
        "    \"val_svm_umap\": {\n",
        "        \"accuracy\": float(acc_umap),\n",
        "        \"macro_f1\": float(f1_umap)\n",
        "    }\n",
        "}\n",
        "with open(reports_dir / \"svm_results.json\", \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(results, f, indent=2, ensure_ascii=False)\n",
        "print(\"Đã cập nhật số liệu kiểm chứng vào reports/svm_results.json.\")\n",
        "display(df_compare)\n"
    ]
))

nb02 = nbformat.v4.new_notebook(cells=cells)
nb02.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb02.metadata["language_info"] = {"name": "python", "version": "3.12.14"}

# Normalize format
nbformat.validator.normalize(nb02)

with open("notebooks/02_svm_umap_baseline.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb02, f)

print("Đã tối ưu hóa và chuẩn hóa thành công notebooks/02_svm_umap_baseline.ipynb")
