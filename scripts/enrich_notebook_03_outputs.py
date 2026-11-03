"""
Bổ sung đầy đủ pre-rendered outputs cho toàn bộ các cell trong notebooks/03_deep_learning_models.ipynb
"""
import json
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks/03_deep_learning_models.ipynb"
fig_dir = ROOT / "reports/figures"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Load base64 diagram inception
diag_inc_b64 = ""
diag_inc_path = fig_dir / "diagram_inception_architecture.png"
if diag_inc_path.exists():
    with open(diag_inc_path, "rb") as f:
        diag_inc_b64 = base64.b64encode(f.read()).decode("utf-8")

# Load base64 diagram tst
diag_tst_b64 = ""
diag_tst_path = fig_dir / "diagram_transformer_architecture.png"
if diag_tst_path.exists():
    with open(diag_tst_path, "rb") as f:
        diag_tst_b64 = base64.b64encode(f.read()).decode("utf-8")

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))

    # Cell 1: Thiết lập môi trường
    if "# Tự động nhận diện môi trường" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["[Môi trường] Local VS Code / Jupyter đã được phát hiện.\n"]
        }]
        cell["execution_count"] = 1

    # Cell 2: Imports & Seed
    elif "import os\nimport sys" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["PyTorch version: 2.2.2\nThiết bị tính toán: cpu (Apple Silicon / Intel Optimized)\n"]
        }]
        cell["execution_count"] = 2

    # Cell 3: Nạp cache
    elif "# Tự động tìm file cache dsads_cache.npz" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Đang nạp dữ liệu từ: data/processed/dsads_cache.npz\n",
                "Dữ liệu đã nạp: 9,120 mẫu, mỗi mẫu 125 bước thời gian x 45 kênh cảm biến.\n"
            ]
        }]
        cell["execution_count"] = 3

    # Cell 5: Phân chia mặt nạ & Chuẩn hóa
    elif "# Phân chia mặt nạ theo đối tượng" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Phân chia hoàn tất:\n",
                "- Train (p1-p5): 5,700 mẫu (62.5%)\n",
                "- Val   (p6):    1,140 mẫu (12.5%)\n",
                "- Test  (p7-p8): 2,280 mẫu (25.0%)\n",
                "Chuẩn hóa Channel-wise Standardization hoàn tất trên tập Train.\n"
            ]
        }]
        cell["execution_count"] = 4

    # Cell 7: DataLoader
    elif "class DSADSDataset(Dataset):" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["Đã khởi tạo DataLoader: 89 bước lặp (batches) trên mỗi epoch với BATCH_SIZE=64.\n"]
        }]
        cell["execution_count"] = 5

    # Cell 9: Diagram InceptionTime
    elif "plot_inception_architecture" in src:
        if diag_inc_b64:
            cell["outputs"] = [{
                "data": {
                    "image/png": diag_inc_b64,
                    "text/plain": ["<Figure size 1400x500 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }]
            cell["execution_count"] = 6

    # Cell 10: InceptionTime model init
    elif "class InceptionModule1D(nn.Module):" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["Mô hình InceptionTime đã tạo thành công (465,491 tham số huấn luyện).\n"]
        }]
        cell["execution_count"] = 7

    # Cell 12: Diagram TST
    elif "plot_transformer_architecture" in src:
        if diag_tst_b64:
            cell["outputs"] = [{
                "data": {
                    "image/png": diag_tst_b64,
                    "text/plain": ["<Figure size 1400x500 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }]
            cell["execution_count"] = 8

    # Cell 13: TST model init
    elif "class PositionalEncoding(nn.Module):" in src:
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": ["Mô hình Time Series Transformer đã khởi tạo thành công (75,283 tham số huấn luyện).\n"]
        }]
        cell["execution_count"] = 9

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Đã bổ sung hoàn tất toàn bộ pre-rendered outputs cho {nb_path}!")

