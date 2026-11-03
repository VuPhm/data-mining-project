"""Tối ưu tương thích import diagrams cho notebooks/03_deep_learning_models.ipynb"""
import json
from pathlib import Path

nb_path = Path("notebooks/03_deep_learning_models.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 2: Tối ưu path setup
nb["cells"][2]["source"] = [
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
    "# Thiết lập đường dẫn thư mục an toàn cho cả Local và Colab\n",
    "for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / 'src', Path.cwd().parent / 'src']:\n",
    "    if str(p) not in sys.path:\n",
    "        sys.path.insert(0, str(p))\n",
    "\n",
    "# Cấu hình hạt giống ngẫu nhiên (seed) để tái lập kết quả thực nghiệm\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.manual_seed_all(SEED)\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Thiết bị tính toán sử dụng: {device}')\n",
    "\n",
    "# Hiển thị sơ đồ luồng tổng thể\n",
    "try:\n",
    "    from diagrams import plot_overall_pipeline\n",
    "except ImportError:\n",
    "    from src.diagrams import plot_overall_pipeline\n",
    "\n",
    "fig_pipe = plot_overall_pipeline()\n",
    "plt.show()\n"
]

# Cell 9: InceptionTime diagram
nb["cells"][9]["source"] = [
    "# Trực quan hóa sơ đồ kiến trúc InceptionTime\n",
    "try:\n",
    "    from diagrams import plot_inception_architecture\n",
    "except ImportError:\n",
    "    from src.diagrams import plot_inception_architecture\n",
    "\n",
    "fig_inc = plot_inception_architecture()\n",
    "plt.show()\n"
]

# Cell 12: TST diagram
nb["cells"][12]["source"] = [
    "# Trực quan hóa sơ đồ kiến trúc Time Series Transformer\n",
    "try:\n",
    "    from diagrams import plot_transformer_architecture\n",
    "except ImportError:\n",
    "    from src.diagrams import plot_transformer_architecture\n",
    "\n",
    "fig_tst = plot_transformer_architecture()\n",
    "plt.show()\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Đã cập nhật notebooks/03_deep_learning_models.ipynb với path setup an toàn.")

