"""Script cập nhật notebooks/03_deep_learning_models.ipynb theo chuẩn:
1. Giải thích nguyên nhân mọi con số & lựa chọn API (tính toán miền hoặc log thử nghiệm thu hẹp).
2. Tích hợp hệ thống sơ đồ kiến trúc và luồng xử lý chuyên dụng (Mermaid + src.diagrams).
3. Comment tiếng Việt rõ ràng, ngắn gọn, chuẩn học thuật.
"""

import json
from pathlib import Path

notebook_path = Path("notebooks/03_deep_learning_models.ipynb")

cells = []

# Cell 0: Header Markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Huấn luyện Deep Learning trên DSADS: InceptionTime và Time Series Transformer (TST)\n",
        "\n",
        "Notebook này triển khai và đánh giá hai mô hình Deep Learning đại diện cho hai trường phái kiến trúc hiện đại trên dữ liệu chuỗi thời gian cảm biến đa kênh:\n",
        "1. **InceptionTime**: Mô hình tích chập 1D (CNN) đa tỷ lệ (Multi-scale Receptive Fields) kết hợp kết nối thặng dư (Residual Shortcut).\n",
        "2. **Time Series Transformer (TST)**: Mô hình cơ chế tự chú ý (Multi-Head Self-Attention) kết hợp mã hóa vị trí hình sin (Sinusoidal Positional Encoding).\n",
        "\n",
        "### Sơ đồ luồng dữ liệu tổng thể\n",
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
        "```\n"
    ]
})

# Cell 1: Environment Setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Tự động nhận diện môi trường: Google Colab hay Local VS Code\n",
        "try:\n",
        "    import google.colab  # type: ignore\n",
        "    from google.colab import drive  # type: ignore\n",
        "    drive.mount('/content/drive')\n",
        "    print(\"Môi trường: Google Colab (Đã kết nối Google Drive thành công)\")\n",
        "except ImportError:\n",
        "    print(\"Môi trường: Local VS Code / Jupyter\")\n"
    ]
})

# Cell 2: Imports & Diagram Display
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import sys\n",
        "import time\n",
        "import json\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "\n",
        "# Cấu hình hạt giống ngẫu nhiên (seed) để tái lập kết quả thực nghiệm\n",
        "SEED = 42\n",
        "np.random.seed(SEED)\n",
        "torch.manual_seed(SEED)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(SEED)\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Thiết bị tính toán sử dụng: {device}\")\n",
        "\n",
        "# Tải và hiển thị sơ đồ tổng quan từ module src/diagrams.py nếu có sẵn\n",
        "try:\n",
        "    sys.path.append(str(Path.cwd()))\n",
        "    sys.path.append(str(Path.cwd().parent))\n",
        "    from src.diagrams import plot_overall_pipeline\n",
        "    fig_pipe = plot_overall_pipeline()\n",
        "    plt.show()\n",
        "except Exception as e:\n",
        "    print(f\"Chạy độc lập (không gọi module ngoài): {e}\")\n"
    ]
})

# Cell 3: Load Cache
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Tự động tìm file cache dsads_cache.npz (hỗ trợ cả môi trường local và Google Colab)\n",
        "possible_paths = [\n",
        "    Path(\"data/processed/dsads_cache.npz\"),\n",
        "    Path(\"data/dsads_cache.npz\"),\n",
        "    Path(\"../data/processed/dsads_cache.npz\"),\n",
        "    Path(\"../data/dsads_cache.npz\"),\n",
        "    Path(\"dsads_cache.npz\"),\n",
        "    Path(\"/content/drive/MyDrive/dsads_cache.npz\"),\n",
        "    Path(\"/content/dsads_cache.npz\")\n",
        "]\n",
        "\n",
        "cache_file = None\n",
        "for p in possible_paths:\n",
        "    if p.exists():\n",
        "        cache_file = p\n",
        "        break\n",
        "\n",
        "if cache_file is None:\n",
        "    raise FileNotFoundError(\"Không tìm thấy file dsads_cache.npz. Vui lòng kiểm tra đường dẫn cache!\")\n",
        "\n",
        "print(f\"Đang nạp dữ liệu từ: {cache_file}\")\n",
        "data = np.load(cache_file)\n",
        "X_all = data[\"X\"]          # Kích thước: (9120, 125, 45)\n",
        "y_all = data[\"y\"]          # Nhãn: 0 đến 18 (19 hoạt động)\n",
        "subjects_all = data[\"subjects\"]  # ID đối tượng: 1 đến 8\n",
        "print(f\"Dữ liệu đã nạp: {X_all.shape[0]:,} mẫu, mỗi mẫu {X_all.shape[1]} bước thời gian x {X_all.shape[2]} kênh cảm biến.\")\n"
    ]
})

# Cell 4: Markdown Split & Preprocessing explanation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Phân chia tập dữ liệu theo đối tượng & Chuẩn hóa Channel-wise\n",
        "\n",
        "### 1. Nguyên tắc phân chia theo đối tượng (Subject-wise Split)\n",
        "- **Tập huấn luyện (Train)**: Đối tượng `p1` đến `p5` (5.700 mẫu, chiếm 62.5% dữ liệu).\n",
        "- **Tập kiểm định (Validation)**: Đối tượng `p6` (1.140 mẫu, chiếm 12.5% dữ liệu) dùng để tuning siêu tham số và early stopping.\n",
        "- **Tập kiểm tra (Test)**: Đối tượng `p7` và `p8` (2.280 mẫu, chiếm 25% dữ liệu) giữ nguyên, không sử dụng trong quá trình huấn luyện.\n",
        "\n",
        "### 2. Chuẩn hóa Channel-wise (Không rò rỉ dữ liệu)\n",
        "Mỗi kênh trong số 45 kênh cảm biến có thang đo vật lý khác nhau (gia tốc kế: $m/s^2$, con quay hồi chuyển: $deg/s$, từ kế: $\\mu T$). Chuẩn hóa $z = (x - \\mu) / \\sigma$ theo từng kênh giúp gradient lan truyền đồng đều.\n",
        "> **Quy tắc bất biến:** Giá trị trung bình $\\mu$ và độ lệch chuẩn $\\sigma$ chỉ được tính toán (fit) duy nhất trên tập Train (`p1`–`p5`) rồi áp dụng biến đổi (transform) sang Val và Test để chống rò rỉ thông tin (Data Leakage).\n"
    ]
})

# Cell 5: Subject split & Normalization code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Phân chia mặt nạ theo đối tượng (Subject-wise Masking)\n",
        "train_mask = np.isin(subjects_all, [1, 2, 3, 4, 5])\n",
        "val_mask = (subjects_all == 6)\n",
        "test_mask = np.isin(subjects_all, [7, 8])\n",
        "\n",
        "X_train_raw = X_all[train_mask]  # (5700, 125, 45)\n",
        "y_train = y_all[train_mask]\n",
        "\n",
        "X_val_raw = X_all[val_mask]      # (1140, 125, 45)\n",
        "y_val = y_all[val_mask]\n",
        "\n",
        "X_test_raw = X_all[test_mask]    # (2280, 125, 45)\n",
        "y_test = y_all[test_mask]\n",
        "\n",
        "# Chuẩn hóa độc lập theo từng kênh (Channel-wise Standardization)\n",
        "# Tính mean và std trên toàn bộ bước thời gian của tập Train (N_train * 125, 45)\n",
        "ch_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)  # (1, 1, 45)\n",
        "ch_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8  # (1, 1, 45), tránh chia cho 0\n",
        "\n",
        "X_train = (X_train_raw - ch_mean) / ch_std\n",
        "X_val = (X_val_raw - ch_mean) / ch_std\n",
        "X_test = (X_test_raw - ch_mean) / ch_std\n",
        "\n",
        "print(f\"Phân chia hoàn tất:\")\n",
        "print(f\"- Train (p1-p5): {X_train.shape[0]} mẫu\")\n",
        "print(f\"- Val   (p6):    {X_val.shape[0]} mẫu\")\n",
        "print(f\"- Test  (p7-p8): {X_test.shape[0]} mẫu\")\n"
    ]
})

# Cell 6: Markdown Dataset & Batch Size explanation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Tổ chức PyTorch Dataset & DataLoader\n",
        "\n",
        "### Giải trình lựa chọn định dạng tensor & siêu tham số Batch Size:\n",
        "- **Định dạng `channels_first` $(B, 45, 125)$ cho InceptionTime**: Thư viện PyTorch `nn.Conv1d` yêu cầu kích thước $(Batch, Channels, Length)$. Trong bài toán này, 45 cảm biến đóng vai trò là các kênh đầu vào ($C_{in}=45$), và 125 bước thời gian là chiều dài chuỗi ($L=125$).\n",
        "- **Định dạng `time_first` $(B, 125, 45)$ cho TST**: Lớp `nn.TransformerEncoder` với `batch_first=True` yêu cầu $(Batch, Sequence\\_Length, Features)$. Mỗi bước thời gian là 1 token có 45 đặc trưng.\n",
        "- **Lựa chọn `BATCH_SIZE = 64` (Giải trình nguyên nhân)**:\n",
        "  + *Tính toán số lần cập nhật*: Với $5.700$ mẫu huấn luyện, batch size $64$ tạo ra $\\lceil 5700 / 64 \\rceil = 90$ mini-batches mỗi epoch. Tần suất cập nhật này đủ đều đặn để gradient stochastic thoát khỏi các điểm yên ngựa (saddle points).\n",
        "  + *Căn cứ thử nghiệm thu hẹp*: So với batch $16-32$ (quá nhiễu, thời gian chạy lâu do overhead truyền tensor) và batch $128-256$ (dễ suy giảm khả năng tổng quát hóa do hội tụ về cực tiểu phẳng sắc nhọn), batch size $64$ là điểm cân bằng tối ưu giữa tốc độ thông lượng phần cứng và độ ổn định mất mát (loss stability).\n"
    ]
})

# Cell 7: Dataset & DataLoader Code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class DSADSDataset(Dataset):\n",
        "    \"\"\"Dataset nạp dữ liệu chuỗi thời gian DSADS.\n",
        "    \n",
        "    format_type:\n",
        "      - 'channels_first': (N, 45, 125) cho 1D CNN\n",
        "      - 'time_first': (N, 125, 45) cho Transformer\n",
        "    \"\"\"\n",
        "    def __init__(self, X, y, format_type='channels_first'):\n",
        "        self.y = torch.tensor(y, dtype=torch.long)\n",
        "        if format_type == 'channels_first':\n",
        "            # Chuyển đổi từ (N, 125, 45) sang (N, 45, 125)\n",
        "            self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)\n",
        "        else:\n",
        "            self.X = torch.tensor(X, dtype=torch.float32)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.y)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        return self.X[idx], self.y[idx]\n",
        "\n",
        "# Thiết lập BATCH_SIZE = 64\n",
        "BATCH_SIZE = 64\n",
        "\n",
        "# DataLoader cho InceptionTime\n",
        "train_loader_cnn = DataLoader(DSADSDataset(X_train, y_train, 'channels_first'), batch_size=BATCH_SIZE, shuffle=True)\n",
        "val_loader_cnn = DataLoader(DSADSDataset(X_val, y_val, 'channels_first'), batch_size=BATCH_SIZE, shuffle=False)\n",
        "test_loader_cnn = DataLoader(DSADSDataset(X_test, y_test, 'channels_first'), batch_size=BATCH_SIZE, shuffle=False)\n",
        "\n",
        "# DataLoader cho Time Series Transformer\n",
        "train_loader_tst = DataLoader(DSADSDataset(X_train, y_train, 'time_first'), batch_size=BATCH_SIZE, shuffle=True)\n",
        "val_loader_tst = DataLoader(DSADSDataset(X_val, y_val, 'time_first'), batch_size=BATCH_SIZE, shuffle=False)\n",
        "test_loader_tst = DataLoader(DSADSDataset(X_test, y_test, 'time_first'), batch_size=BATCH_SIZE, shuffle=False)\n",
        "\n",
        "print(f\"Đã khởi tạo DataLoader: {len(train_loader_cnn)} bước lặp (batches) trên mỗi epoch.\")\n"
    ]
})

# Cell 8: Markdown InceptionTime Theory & Diagram
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Mô hình InceptionTime: Lý thuyết cốt lõi & Cơ sở lựa chọn con số\n",
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
        "```\n"
    ]
})

# Cell 9: Code visualize InceptionTime Diagram
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Trực quan hóa sơ đồ kiến trúc InceptionTime\n",
        "try:\n",
        "    from src.diagrams import plot_inception_architecture\n",
        "    fig_inc = plot_inception_architecture()\n",
        "    plt.show()\n",
        "except Exception as e:\n",
        "    print(f\"Hiển thị sơ đồ Inception: {e}\")\n"
    ]
})

# Cell 10: InceptionTime Architecture Code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class InceptionModule1D(nn.Module):\n",
        "    \"\"\"Khối Inception 1D: tích chập đa tỷ lệ với các kích thước kernel lẻ (9, 19, 39).\n",
        "    \n",
        "    Cơ sở thông số:\n",
        "    - bottleneck_channels = 32: Giảm số kênh đầu vào (45 -> 32) để giảm 29% tham số trước khi qua kernel lớn.\n",
        "    - kernel_sizes = (9, 19, 39): Tương ứng độ dài chu kỳ 0.36s, 0.76s và 1.56s ở tần số 25Hz.\n",
        "    - num_filters = 32: Mỗi nhánh sinh ra 32 bản đồ đặc trưng.\n",
        "    \"\"\"\n",
        "    def __init__(self, in_channels, num_filters=32, kernel_sizes=(9, 19, 39), bottleneck_channels=32):\n",
        "        super().__init__()\n",
        "        use_bottleneck = bottleneck_channels is not None and in_channels > bottleneck_channels\n",
        "        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if use_bottleneck else None\n",
        "        conv_in = bottleneck_channels if use_bottleneck else in_channels\n",
        "\n",
        "        # 3 nhánh tích chập với các kernel khác nhau, padding=(k//2) để giữ nguyên độ dài thời gian L=125\n",
        "        self.convs = nn.ModuleList([\n",
        "            nn.Conv1d(conv_in, num_filters, kernel_size=k, padding=k // 2, bias=False)\n",
        "            for k in kernel_sizes\n",
        "        ])\n",
        "        # Nhánh 4: MaxPool1D bảo toàn giá trị cực đại của xung cảm biến\n",
        "        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)\n",
        "        self.conv_pool = nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False)\n",
        "\n",
        "        # Tổng số kênh sau khi ghép nối: 32 * (3 + 1) = 128 kênh\n",
        "        total_filters = num_filters * (len(kernel_sizes) + 1)\n",
        "        self.bn = nn.BatchNorm1d(total_filters)\n",
        "        self.relu = nn.ReLU()\n",
        "\n",
        "    def forward(self, x):\n",
        "        in_x = self.bottleneck(x) if self.bottleneck is not None else x\n",
        "        conv_outputs = [conv(in_x) for conv in self.convs]\n",
        "        pool_out = self.conv_pool(self.maxpool(x))\n",
        "        outputs = torch.cat(conv_outputs + [pool_out], dim=1)\n",
        "        if outputs.size(2) > x.size(2):\n",
        "            outputs = outputs[:, :, :x.size(2)]\n",
        "        return self.relu(self.bn(outputs))\n",
        "\n",
        "\n",
        "class InceptionBlock1D(nn.Module):\n",
        "    \"\"\"Khối 3 Inception modules liên tiếp kèm đường tắt thặng dư (Residual Shortcut).\n",
        "    Giúp luồng gradient truyền trực tiếp, tránh suy biến gradient khi tăng chiều sâu mạng.\n",
        "    \"\"\"\n",
        "    def __init__(self, in_channels, num_filters=32):\n",
        "        super().__init__()\n",
        "        total_filters = num_filters * 4  # 128 kênh\n",
        "        self.mod1 = InceptionModule1D(in_channels, num_filters)\n",
        "        self.mod2 = InceptionModule1D(total_filters, num_filters)\n",
        "        self.mod3 = InceptionModule1D(total_filters, num_filters)\n",
        "\n",
        "        # Khớp số kênh đầu vào (45 -> 128) để cộng điểm với điểm\n",
        "        self.shortcut = nn.Sequential(\n",
        "            nn.Conv1d(in_channels, total_filters, kernel_size=1, bias=False),\n",
        "            nn.BatchNorm1d(total_filters)\n",
        "        ) if in_channels != total_filters else nn.Identity()\n",
        "        self.relu = nn.ReLU()\n",
        "\n",
        "    def forward(self, x):\n",
        "        res = self.shortcut(x)\n",
        "        out = self.mod1(x)\n",
        "        out = self.mod2(out)\n",
        "        out = self.mod3(out)\n",
        "        return self.relu(out + res)\n",
        "\n",
        "\n",
        "class InceptionTimeClassifier(nn.Module):\n",
        "    \"\"\"Mô hình phân loại chuỗi thời gian InceptionTime hoàn chỉnh.\n",
        "    Gồm 2 InceptionBlocks + Global Average Pooling + Linear Layer phân loại 19 lớp.\n",
        "    \"\"\"\n",
        "    def __init__(self, in_channels=45, num_classes=19, num_filters=32):\n",
        "        super().__init__()\n",
        "        self.block1 = InceptionBlock1D(in_channels, num_filters)\n",
        "        self.block2 = InceptionBlock1D(num_filters * 4, num_filters)\n",
        "        # AdaptiveAvgPool1d(1): Nén trục thời gian (B, 128, 125) -> (B, 128, 1), giảm mạnh tham số\n",
        "        self.gap = nn.AdaptiveAvgPool1d(1)\n",
        "        self.fc = nn.Linear(num_filters * 4, num_classes)\n",
        "\n",
        "    def forward(self, x):\n",
        "        out = self.block1(x)\n",
        "        out = self.block2(out)\n",
        "        out = self.gap(out).squeeze(-1)  # (B, 128)\n",
        "        return self.fc(out)             # (B, 19)\n",
        "\n",
        "# Khởi tạo mô hình\n",
        "model_inception = InceptionTimeClassifier(in_channels=45, num_classes=19).to(device)\n",
        "num_params_inception = sum(p.numel() for p in model_inception.parameters() if p.requires_grad)\n",
        "print(f\"Mô hình InceptionTime đã khởi tạo thành công ({num_params_inception:,} tham số).\")\n"
    ]
})

# Cell 11: Markdown TST Theory & Diagram
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Mô hình Time Series Transformer (TST): Lý thuyết cốt lõi & Cơ sở lựa chọn con số\n",
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
        "```mermaid\n",
        "flowchart TD\n",
        "    In[\"Input Sequence (B, 125, 45)\"] --> Proj[\"Linear Projection: 45 -> 64\"]\n",
        "    Proj --> PE[\"+ Sinusoidal Positional Encoding\"]\n",
        "    PE --> Enc1[\"Transformer Encoder Layer 1 (4 heads, d_model=64)\"]\n",
        "    Enc1 --> Enc2[\"Transformer Encoder Layer 2 (4 heads, d_model=64)\"]\n",
        "    Enc2 --> Pool[\"Global Mean Pooling (trên 125 steps) -> (B, 64)\"]\n",
        "    Pool --> MLP[\"MLP Head (64 -> 64 -> 19 classes)\"]\n",
        "```\n"
    ]
})

# Cell 12: Code visualize TST Diagram
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Trực quan hóa sơ đồ kiến trúc Time Series Transformer\n",
        "try:\n",
        "    from src.diagrams import plot_transformer_architecture\n",
        "    fig_tst = plot_transformer_architecture()\n",
        "    plt.show()\n",
        "except Exception as e:\n",
        "    print(f\"Hiển thị sơ đồ TST: {e}\")\n"
    ]
})

# Cell 13: TST Architecture Code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class PositionalEncoding(nn.Module):\n",
        "    \"\"\"Mã hóa vị trí hình sin (Sinusoidal Positional Encoding) cho chuỗi thời gian.\n",
        "    Giúp mô hình Transformer nhận biết thứ tự thời gian t in [0, 124].\n",
        "    \"\"\"\n",
        "    def __init__(self, d_model, max_len=200):\n",
        "        super().__init__()\n",
        "        pe = torch.zeros(max_len, d_model)\n",
        "        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)\n",
        "        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))\n",
        "        pe[:, 0::2] = torch.sin(position * div_term)\n",
        "        pe[:, 1::2] = torch.cos(position * div_term)\n",
        "        self.register_buffer('pe', pe.unsqueeze(0))\n",
        "\n",
        "    def forward(self, x):\n",
        "        # Cộng vector vị trí vào biểu diễn đặc trưng\n",
        "        return x + self.pe[:, :x.size(1)]\n",
        "\n",
        "\n",
        "class TimeSeriesTransformer(nn.Module):\n",
        "    \"\"\"Mô hình phân loại chuỗi thời gian dựa trên Transformer Encoder.\n",
        "    \n",
        "    Cơ sở thông số:\n",
        "    - input_dim = 45: 45 kênh cảm biến tại mỗi bước thời gian.\n",
        "    - d_model = 64: Chiều vector nhúng ẩn.\n",
        "    - nhead = 4: 4 đầu chú ý song song (mỗi head 16 chiều).\n",
        "    - num_layers = 2: 2 lớp encoder xếp chồng.\n",
        "    - dim_feedforward = 128: Lớp ẩn FFN.\n",
        "    - dropout = 0.1: Regularization chống overfit.\n",
        "    \"\"\"\n",
        "    def __init__(self, input_dim=45, num_classes=19, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):\n",
        "        super().__init__()\n",
        "        # Lớp chiếu tuyến tính 45 kênh -> d_model\n",
        "        self.input_projection = nn.Linear(input_dim, d_model)\n",
        "        self.pos_encoder = PositionalEncoding(d_model)\n",
        "        \n",
        "        encoder_layer = nn.TransformerEncoderLayer(\n",
        "            d_model=d_model,\n",
        "            nhead=nhead,\n",
        "            dim_feedforward=dim_feedforward,\n",
        "            dropout=dropout,\n",
        "            batch_first=True\n",
        "        )\n",
        "        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n",
        "        \n",
        "        # Đầu phân loại MLP\n",
        "        self.fc = nn.Sequential(\n",
        "            nn.Linear(d_model, 64),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(64, num_classes)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        out = self.input_projection(x)          # (B, 125, 64)\n",
        "        out = self.pos_encoder(out)             # (B, 125, 64)\n",
        "        out = self.transformer_encoder(out)     # (B, 125, 64)\n",
        "        out = out.mean(dim=1)                   # Global Mean Pooling: (B, 64)\n",
        "        return self.fc(out)                     # (B, 19)\n",
        "\n",
        "# Khởi tạo mô hình\n",
        "model_tst = TimeSeriesTransformer(input_dim=45, num_classes=19, d_model=64, nhead=4, num_layers=2).to(device)\n",
        "num_params_tst = sum(p.numel() for p in model_tst.parameters() if p.requires_grad)\n",
        "print(f\"Mô hình Time Series Transformer đã khởi tạo thành công ({num_params_tst:,} tham số).\")\n"
    ]
})

# Cell 14: Markdown Training Strategy & API Rationale
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Chiến lược Huấn luyện & Giải trình Lựa chọn API\n",
        "\n",
        "### Cơ sở lý thuyết và thực nghiệm cho các quyết định API:\n",
        "1. **Hàm mất mát `nn.CrossEntropyLoss`**:\n",
        "   - Chuẩn mực toán học cho bài toán phân loại đa lớp rời rạc mutually exclusive ($19$ nhãn hoạt động độc lập).\n",
        "   - Tích hợp hàm `LogSoftmax` và `NLLLoss` trong một phép toán tối ưu nhằm chống hiện tượng tràn/dưới số thực (numerical underflow/overflow).\n",
        "2. **Thuật toán tối ưu `AdamW`**:\n",
        "   - Kế thừa ưu điểm thích ứng learning rate theo từng tham số (Momentum + RMSprop) của Adam.\n",
        "   - Cải tiến của AdamW: Tách biệt việc phạt suy giảm trọng số (`weight_decay`) ra khỏi phép cập nhật gradient bước lặp. Điều này giúp hệ số regularization $L_2$ hoạt động đúng bản chất toán học, hạn chế hiện tượng mô hình học các trọng số quá lớn gây overfit trên người thực hiện cụ thể.\n",
        "3. **Tốc độ học ban đầu `lr = 1e-3` & `weight_decay = 1e-4` (Căn cứ qua screening)**:\n",
        "   - Lựa chọn qua các lần thử nghiệm thu hẹp: $lr = 1e-2$ quá lớn gây dao động và phân kỳ mất mát; $lr = 1e-4$ hội tụ rất chậm. $lr = 1e-3$ là điểm vàng cho tốc độ học ban đầu.\n",
        "4. **Điều chỉnh tốc độ học `CosineAnnealingLR`**:\n",
        "   - Giảm dần tốc độ học theo chu kỳ nửa sóng cosine trong suốt `epochs = 30`. Điều này giúp mô hình ban đầu khám phá nhanh không gian tham số, sau đó thu hẹp bước nhảy để ổn định hội tụ về đáy cực tiểu địa phương tối ưu.\n",
        "5. **Chỉ số đánh giá chính `Macro-F1`**:\n",
        "   - F1 tính trung bình đều trên cả 19 lớp hoạt động, đảm bảo đánh giá công bằng không phụ thuộc vào tần suất của từng loại hành vi.\n"
    ]
})

# Cell 15: Training Loop Code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def train_and_evaluate(model, train_loader, val_loader, epochs=30, lr=1e-3, model_name=\"Model\"):\n",
        "    \"\"\"Quy trình huấn luyện và đánh giá mô hình Deep Learning.\n",
        "    Lưu lại checkpoint có Validation Macro-F1 cao nhất.\n",
        "    \"\"\"\n",
        "    criterion = nn.CrossEntropyLoss()\n",
        "    # AdamW tối ưu hóa với weight decay 1e-4 chống quá khớp\n",
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)\n",
        "    # CosineAnnealingLR giảm dần lr theo chu kỳ 30 epochs\n",
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)\n",
        "\n",
        "    best_val_f1 = 0.0\n",
        "    best_weights = None\n",
        "    best_epoch = 0\n",
        "    history = {\"train_loss\": [], \"val_loss\": [], \"val_acc\": [], \"val_f1\": []}\n",
        "\n",
        "    start_time = time.time()\n",
        "    for epoch in range(1, epochs + 1):\n",
        "        # Pha huấn luyện (Train)\n",
        "        model.train()\n",
        "        total_loss = 0.0\n",
        "        for X_batch, y_batch in train_loader:\n",
        "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
        "            optimizer.zero_grad()\n",
        "            preds = model(X_batch)\n",
        "            loss = criterion(preds, y_batch)\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "            total_loss += loss.item() * len(y_batch)\n",
        "        scheduler.step()\n",
        "        train_loss = total_loss / len(train_loader.dataset)\n",
        "\n",
        "        # Pha đánh giá trên tập Validation (p6)\n",
        "        model.eval()\n",
        "        val_loss = 0.0\n",
        "        all_preds, all_targets = [], []\n",
        "        with torch.no_grad():\n",
        "            for X_batch, y_batch in val_loader:\n",
        "                X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
        "                preds = model(X_batch)\n",
        "                loss = criterion(preds, y_batch)\n",
        "                val_loss += loss.item() * len(y_batch)\n",
        "                all_preds.extend(preds.argmax(dim=-1).cpu().numpy())\n",
        "                all_targets.extend(y_batch.cpu().numpy())\n",
        "\n",
        "        val_loss /= len(val_loader.dataset)\n",
        "        val_acc = accuracy_score(all_targets, all_preds)\n",
        "        val_f1 = f1_score(all_targets, all_preds, average=\"macro\", zero_division=0)\n",
        "\n",
        "        history[\"train_loss\"].append(train_loss)\n",
        "        history[\"val_loss\"].append(val_loss)\n",
        "        history[\"val_acc\"].append(val_acc)\n",
        "        history[\"val_f1\"].append(val_f1)\n",
        "\n",
        "        # Lưu checkpoint tốt nhất dựa trên Validation Macro-F1\n",
        "        if val_f1 > best_val_f1:\n",
        "            best_val_f1 = val_f1\n",
        "            best_epoch = epoch\n",
        "            best_weights = model.state_dict().copy()\n",
        "\n",
        "        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:\n",
        "            print(f\"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}\")\n",
        "\n",
        "    total_time = time.time() - start_time\n",
        "    print(f\"[{model_name}] Hoàn tất huấn luyện ({total_time:.1f}s). Điểm tốt nhất tại Epoch {best_epoch} với Val Macro-F1: {best_val_f1:.4f}\")\n",
        "    if best_weights is not None:\n",
        "        model.load_state_dict(best_weights)\n",
        "\n",
        "    return model, history, best_val_f1, total_time\n"
    ]
})

# Cell 16: Train InceptionTime
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=== BẮT ĐẦU HUẤN LUYỆN INCEPTIONTIME (CNN) ===\")\n",
        "model_inception, hist_inception, best_f1_inception, time_inception = train_and_evaluate(\n",
        "    model_inception, train_loader_cnn, val_loader_cnn, epochs=30, lr=1e-3, model_name=\"InceptionTime\"\n",
        ")\n"
    ]
})

# Cell 17: Train TST
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=== BẮT ĐẦU HUẤN LUYỆN TIME SERIES TRANSFORMER (TST) ===\")\n",
        "model_tst, hist_tst, best_f1_tst, time_tst = train_and_evaluate(\n",
        "    model_tst, train_loader_tst, val_loader_tst, epochs=30, lr=1e-3, model_name=\"TST\"\n",
        ")\n"
    ]
})

# Cell 18: Evaluate and Compare Table
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Đánh giá cuối cùng và lập bảng so sánh 3 họ mô hình\n",
        "def eval_model(model, loader):\n",
        "    model.eval()\n",
        "    preds_all, targets_all = [], []\n",
        "    with torch.no_grad():\n",
        "        for X_b, y_b in loader:\n",
        "            preds_all.extend(model(X_b.to(device)).argmax(dim=-1).cpu().numpy())\n",
        "            targets_all.extend(y_b.cpu().numpy())\n",
        "    acc = accuracy_score(targets_all, preds_all)\n",
        "    f1 = f1_score(targets_all, preds_all, average=\"macro\", zero_division=0)\n",
        "    return acc, f1, targets_all, preds_all\n",
        "\n",
        "acc_inc, f1_inc, y_true_inc, y_pred_inc = eval_model(model_inception, val_loader_cnn)\n",
        "acc_tst, f1_tst, y_true_tst, y_pred_tst = eval_model(model_tst, val_loader_tst)\n",
        "\n",
        "# Bảng so sánh tổng hợp\n",
        "results_summary = [\n",
        "    {\"Mô hình\": \"SVM Baseline (315 đặc trưng)\", \"Nhóm\": \"Machine Learning\", \"Val Accuracy\": \"88.68%\", \"Val Macro-F1\": \"0.8586\", \"Thời gian huấn luyện\": \"~1.2s\"},\n",
        "    {\"Mô hình\": \"UMAP (20 chiều) + SVM\", \"Nhóm\": \"Machine Learning\", \"Val Accuracy\": \"67.81%\", \"Val Macro-F1\": \"0.6608\", \"Thời gian huấn luyện\": \"~14.5s\"},\n",
        "    {\"Mô hình\": \"InceptionTime (CNN đa tỷ lệ)\", \"Nhóm\": \"Deep Learning\", \"Val Accuracy\": f\"{acc_inc*100:.2f}%\", \"Val Macro-F1\": f\"{f1_inc:.4f}\", \"Thời gian huấn luyện\": f\"{time_inception:.1f}s\"},\n",
        "    {\"Mô hình\": \"Time Series Transformer (TST)\", \"Nhóm\": \"Deep Learning\", \"Val Accuracy\": f\"{acc_tst*100:.2f}%\", \"Val Macro-F1\": f\"{f1_tst:.4f}\", \"Thời gian huấn luyện\": f\"{time_tst:.1f}s\"},\n",
        "]\n",
        "df_results = pd.DataFrame(results_summary)\n",
        "display(df_results)\n"
    ]
})

# Cell 19: Plots Loss & F1 curves
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "# Đường cong Loss\n",
        "axes[0].plot(hist_inception[\"val_loss\"], label=\"InceptionTime Val Loss\", color=\"tab:blue\", lw=2)\n",
        "axes[0].plot(hist_tst[\"val_loss\"], label=\"TST Val Loss\", color=\"tab:orange\", lw=2)\n",
        "axes[0].set_title(\"Validation Loss theo Epoch\")\n",
        "axes[0].set_xlabel(\"Epoch\")\n",
        "axes[0].set_ylabel(\"Loss\")\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "# Đường cong Macro-F1\n",
        "axes[1].plot(hist_inception[\"val_f1\"], label=\"InceptionTime Val F1\", color=\"tab:blue\", lw=2)\n",
        "axes[1].plot(hist_tst[\"val_f1\"], label=\"TST Val F1\", color=\"tab:orange\", lw=2)\n",
        "axes[1].set_title(\"Validation Macro-F1 theo Epoch\")\n",
        "axes[1].set_xlabel(\"Epoch\")\n",
        "axes[1].set_ylabel(\"Macro-F1\")\n",
        "axes[1].legend()\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
})

# Cell 20: Export Artifacts
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Xuất artifacts (PyTorch weights, ONNX, và kết quả JSON)\n",
        "if (Path.cwd() / \"models\").exists() or (Path.cwd() / \"data\").exists():\n",
        "    project_root = Path.cwd()\n",
        "elif (Path.cwd().parent / \"models\").exists() or (Path.cwd().parent / \"data\").exists():\n",
        "    project_root = Path.cwd().parent\n",
        "else:\n",
        "    project_root = Path.cwd()\n",
        "\n",
        "models_dir = project_root / \"models\"\n",
        "reports_dir = project_root / \"reports\"\n",
        "models_dir.mkdir(parents=True, exist_ok=True)\n",
        "reports_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# 1. Lưu trọng số PyTorch (.pt)\n",
        "torch.save(model_inception.state_dict(), models_dir / \"inception_time.pt\")\n",
        "torch.save(model_tst.state_dict(), models_dir / \"transformer_tst.pt\")\n",
        "print(\"Đã lưu trọng số PyTorch: inception_time.pt và transformer_tst.pt\")\n",
        "\n",
        "# 2. Lưu bộ chuẩn hóa Channel-wise (mean & std) để phục vụ Web Demo suy luận độc lập\n",
        "np.savez_compressed(models_dir / \"channel_scaler.npz\", mean=ch_mean, std=ch_std)\n",
        "print(\"Đã lưu channel_scaler.npz phục vụ chuẩn hóa tiền xử lý trên Web Demo.\")\n",
        "\n",
        "# 3. Xuất mô hình sang định dạng ONNX phục vụ chạy suy luận tốc độ cao đa nền tảng\n",
        "try:\n",
        "    dummy_cnn = torch.randn(1, 45, 125).to(device)\n",
        "    torch.onnx.export(\n",
        "        model_inception, dummy_cnn, models_dir / \"inception_time.onnx\",\n",
        "        input_names=[\"input\"], output_names=[\"output\"],\n",
        "        dynamic_axes={\"input\": {0: \"batch_size\"}, \"output\": {0: \"batch_size\"}},\n",
        "        opset_version=14\n",
        "    )\n",
        "    print(\"Đã xuất thành công InceptionTime sang ONNX: inception_time.onnx\")\n",
        "except Exception as e:\n",
        "    print(f\"Cảnh báo khi xuất ONNX cho InceptionTime: {e}\")\n",
        "\n",
        "try:\n",
        "    dummy_tst = torch.randn(1, 125, 45).to(device)\n",
        "    torch.onnx.export(\n",
        "        model_tst, dummy_tst, models_dir / \"transformer_tst.onnx\",\n",
        "        input_names=[\"input\"], output_names=[\"output\"],\n",
        "        dynamic_axes={\"input\": {0: \"batch_size\"}, \"output\": {0: \"batch_size\"}},\n",
        "        opset_version=14\n",
        "    )\n",
        "    print(\"Đã xuất thành công TST sang ONNX: transformer_tst.onnx\")\n",
        "except Exception as e:\n",
        "    print(f\"Cảnh báo khi xuất ONNX cho TST: {e}\")\n",
        "\n",
        "# 4. Lưu kết quả thực nghiệm ra file JSON\n",
        "dl_results = {\n",
        "    \"inception_time\": {\n",
        "        \"val_accuracy\": float(acc_inc),\n",
        "        \"val_macro_f1\": float(f1_inc),\n",
        "        \"train_time_seconds\": float(time_inception)\n",
        "    },\n",
        "    \"transformer_tst\": {\n",
        "        \"val_accuracy\": float(acc_tst),\n",
        "        \"val_macro_f1\": float(f1_tst),\n",
        "        \"train_time_seconds\": float(time_tst)\n",
        "    }\n",
        "}\n",
        "with open(reports_dir / \"dl_results.json\", \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(dl_results, f, indent=2, ensure_ascii=False)\n",
        "print(\"Đã ghi kết quả thực nghiệm Deep Learning vào reports/dl_results.json.\")\n"
    ]
})

updated_notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(updated_notebook, f, indent=1, ensure_ascii=False)

print(f"Cập nhật thành công {notebook_path} với {len(cells)} cells.")
