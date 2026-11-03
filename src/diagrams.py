"""Hệ thống vẽ sơ đồ, biểu đồ chuyên dụng cho đề tài DSADS.

Mục đích:
- Mô tả trực quan luồng xử lý dữ liệu và luồng biến đổi tensor.
- Tập trung vào các thành phần chính liên quan luồng, loại bỏ các chi tiết thừa đã có trong phần mô tả.
- Đảm bảo tiêu đề và mô tả trong từng node được phân tách tuyệt đối, không chồng lấn.
"""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _setup_figure(figsize: Tuple[int, int] = (13, 7)):
    """Khởi tạo figure với phong cách học thuật sạch sẽ."""
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    return fig, ax


def _draw_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    desc: str = "",
    color: str = "#EBF3FB",
    edge_color: str = "#2B6CB0",
    title_color: str = "#1A365D",
    desc_color: str = "#2D3748",
    fontsize_title: float = 9.5,
    fontsize_desc: float = 8.0,
):
    """Vẽ một hộp thành phần với tiêu đề và mô tả được phân tách rõ ràng, chống chồng lấn."""
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=1.4,
    )
    ax.add_patch(box)
    
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    
    if desc:
        # Title căn bottom phía trên tâm, Desc căn top phía dưới tâm -> không bao giờ chồng lấn
        ax.text(
            center_x,
            center_y + 1.2,
            title,
            ha="center",
            va="bottom",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            center_x,
            center_y - 1.2,
            desc,
            ha="center",
            va="top",
            fontsize=fontsize_desc,
            color=desc_color,
            linespacing=1.25,
            multialignment="center",
        )
    else:
        ax.text(
            center_x,
            center_y,
            title,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
        )


def _draw_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    label: str = "",
    color: str = "#4A5568",
    fontsize: float = 7.5,
):
    """Vẽ mũi tên liên kết giữa các khối."""
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.3,
        color=color,
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0 + 1.6
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="#2D3748",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )


def plot_overall_pipeline(save_path: Optional[str] = None):
    """Sơ đồ luồng tổng thể hệ thống thực nghiệm DSADS."""
    fig, ax = _setup_figure(figsize=(13, 7.2))

    # Tiêu đề biểu đồ
    ax.text(
        50,
        96,
        "SƠ ĐỒ TỔNG THỂ LUỒNG XỬ LÝ HỆ THỐNG DSADS",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        50,
        92,
        "Đối chiếu luồng xử lý: Tiền xử lý → Phân tách nhánh Machine Learning & Deep Learning → Đánh giá",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A5568",
    )

    # 1. Dữ liệu gốc
    _draw_box(
        ax, 2, 65, 20, 16,
        "1. DSADS Raw Data",
        "Shape: (9.120, 125, 45)\n19 lớp, 8 người (25 Hz)",
        color="#EBF8FF", edge_color="#3182CE"
    )

    # 2. Phân chia Subject
    _draw_box(
        ax, 26, 65, 22, 16,
        "2. Subject Split",
        "Train: p1–p5 (5.700 mẫu)\nVal: p6 | Test: p7–p8",
        color="#FEFCBF", edge_color="#D69E2E"
    )
    _draw_arrow(ax, (22, 73), (26, 73), label="")

    # Nhánh A: Machine Learning
    _draw_box(
        ax, 52, 73, 22, 15,
        "3A. Feature Extraction",
        "7 thống kê x 45 kênh\nVector: (N, 315)",
        color="#E6FFFA", edge_color="#319795"
    )
    _draw_arrow(ax, (48, 77), (52, 80), label="Đặc trưng")

    _draw_box(
        ax, 78, 73, 20, 15,
        "4A. Machine Learning",
        "• SVM Baseline (315 chiều)\n• UMAP (20 chiều) + SVM",
        color="#EBF4FF", edge_color="#4299E1"
    )
    _draw_arrow(ax, (74, 80), (78, 80), label="Scaler")

    # Nhánh B: Deep Learning
    _draw_box(
        ax, 52, 47, 22, 15,
        "3B. Chuẩn hóa 3D",
        "Channel-wise Scaler\nTensor: (N, 45, 125)",
        color="#FAF5FF", edge_color="#805AD5"
    )
    _draw_arrow(ax, (48, 69), (52, 55), label="Tensor 3D")

    _draw_box(
        ax, 78, 47, 20, 15,
        "4B. Deep Learning",
        "• InceptionTime (1D CNN)\n• TST (Transformer)",
        color="#F3E8FF", edge_color="#6B46C1"
    )
    _draw_arrow(ax, (74, 55), (78, 55), label="End-to-End")

    # 5. Đánh giá & So sánh
    _draw_box(
        ax, 26, 17, 44, 17,
        "5. Đánh giá & Bàn giao Artifacts",
        "• Metric: Macro-F1 (chính), Accuracy, Confusion Matrix\n• Lưu Checkpoints (.joblib, .pt) và xuất định dạng ONNX",
        color="#F0FFF4", edge_color="#38A169"
    )

    # 6. Web Demo
    _draw_box(
        ax, 76, 17, 22, 17,
        "6. Ứng dụng Web Demo",
        "• Tải/chọn mẫu cảm biến\n• Suy luận 3 mô hình & UMAP 2D",
        color="#FFF5F5", edge_color="#E53E3E"
    )

    # Mũi tên tổng hợp vào Đánh giá & Demo
    _draw_arrow(ax, (78, 80), (68, 70), label="")
    _draw_arrow(ax, (68, 70), (55, 34), label="")
    _draw_arrow(ax, (78, 54), (65, 34), label="")
    _draw_arrow(ax, (70, 25), (76, 25), label="Tích hợp")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_inception_architecture(save_path: Optional[str] = None):
    """Sơ đồ kiến trúc mô hình InceptionTime."""
    fig, ax = _setup_figure(figsize=(13, 7.2))

    ax.text(
        50,
        96,
        "KIẾN TRÚC MÔ HÌNH INCEPTIONTIME CHO TÍN HIỆU CẢM BIẾN DSADS",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        50,
        92,
        "Tích hợp Bottleneck 1D, Tích chập đa tỷ lệ song song và Đường tắt thặng dư (Residual)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A5568",
    )

    # Input Tensor
    _draw_box(
        ax, 2, 48, 16, 15,
        "Input Tensor",
        "Shape: (B, 45, 125)\n45 kênh x 125 steps",
        color="#EBF8FF", edge_color="#3182CE"
    )

    # Bottleneck Layer
    _draw_box(
        ax, 21, 48, 17, 15,
        "Bottleneck Conv1D",
        "Kernel=1, Stride=1\n45 → 32 channels",
        color="#FEFCBF", edge_color="#D69E2E"
    )
    _draw_arrow(ax, (18, 55), (21, 55), label="")

    # 4 Parallel Branches
    # Nhánh 1: Kernel 9
    _draw_box(
        ax, 43, 75, 19, 11,
        "Conv1D (k = 9)",
        "32 filters | pad=4\nChu kỳ ngắn (~0.36s)",
        color="#E6FFFA", edge_color="#319795"
    )
    # Nhánh 2: Kernel 19
    _draw_box(
        ax, 43, 61, 19, 11,
        "Conv1D (k = 19)",
        "32 filters | pad=9\nChu kỳ trung (~0.76s)",
        color="#E6FFFA", edge_color="#319795"
    )
    # Nhánh 3: Kernel 39
    _draw_box(
        ax, 43, 47, 19, 11,
        "Conv1D (k = 39)",
        "32 filters | pad=19\nChu kỳ dài (~1.56s)",
        color="#E6FFFA", edge_color="#319795"
    )
    # Nhánh 4: MaxPool + Conv1D
    _draw_box(
        ax, 43, 33, 19, 11,
        "MaxPool + Conv1D",
        "Pool(k=3) + Conv(1x1)\n32 filters",
        color="#EDF2F7", edge_color="#718096"
    )

    # Mũi tên từ Bottleneck vào 4 nhánh
    _draw_arrow(ax, (38, 60), (43, 80), label="")
    _draw_arrow(ax, (38, 57), (43, 66), label="")
    _draw_arrow(ax, (38, 53), (43, 52), label="")
    _draw_arrow(ax, (38, 50), (43, 38), label="")

    # Concatenate
    _draw_box(
        ax, 66, 48, 14, 15,
        "Concatenate",
        "32x4 = 128 channels\nShape: (B, 128, 125)",
        color="#FAF5FF", edge_color="#805AD5"
    )
    _draw_arrow(ax, (62, 80), (66, 58), label="")
    _draw_arrow(ax, (62, 66), (66, 56), label="")
    _draw_arrow(ax, (62, 52), (66, 54), label="")
    _draw_arrow(ax, (62, 38), (66, 52), label="")

    # Residual Shortcut
    shortcut = patches.FancyArrowPatch(
        (10, 48),
        (86, 44),
        connectionstyle="arc3,rad=-0.4",
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.6,
        linestyle="--",
        color="#E53E3E",
    )
    ax.add_patch(shortcut)
    ax.text(
        48, 19,
        "Residual Shortcut: Conv1D (1x1) 45 → 128 channels (Cộng thặng dư)",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#C53030",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#FEB2B2")
    )

    # BatchNorm + ReLU + Residual Add
    _draw_box(
        ax, 84, 48, 14, 15,
        "Add + BN + ReLU",
        "Cộng Residual\nShape: (B, 128, 125)",
        color="#FEEBC8", edge_color="#DD6B20"
    )
    _draw_arrow(ax, (80, 55), (84, 55), label="")

    # Global Average Pooling & Output
    _draw_box(
        ax, 84, 21, 14, 19,
        "Output Classifier",
        "• AdaptiveAvgPool1D (B, 128)\n• Linear 128 → 19 classes",
        color="#F0FFF4", edge_color="#38A169"
    )
    _draw_arrow(ax, (91, 48), (91, 40), label="")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_transformer_architecture(save_path: Optional[str] = None):
    """Sơ đồ kiến trúc Time Series Transformer (TST)."""
    fig, ax = _setup_figure(figsize=(13, 7.2))

    ax.text(
        50,
        96,
        "KIẾN TRÚC MÔ HÌNH TIME SERIES TRANSFORMER (TST)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        50,
        92,
        "Chiếu tuyến tính đặc trưng, Mã hóa vị trí hình sin và Cơ chế Multi-Head Self-Attention",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A5568",
    )

    # 1. Input
    _draw_box(
        ax, 3, 49, 15, 15,
        "Input Sequence",
        "Shape: (B, 125, 45)\n125 steps x 45 channels",
        color="#EBF8FF", edge_color="#3182CE"
    )

    # 2. Linear Projection
    _draw_box(
        ax, 21, 49, 16, 15,
        "Linear Projection",
        "Chiếu 45 → d_model=64\nShape: (B, 125, 64)",
        color="#FEFCBF", edge_color="#D69E2E"
    )
    _draw_arrow(ax, (18, 56), (21, 56), label="")

    # 3. Positional Encoding
    _draw_box(
        ax, 40, 49, 18, 15,
        "+ Positional Encoding",
        "Sinusoidal PE + Dropout(0.1)\nShape: (B, 125, 64)",
        color="#FAF5FF", edge_color="#805AD5"
    )
    _draw_arrow(ax, (37, 56), (40, 56), label="")

    # 4. Transformer Encoder Layer
    encoder_box = patches.FancyBboxPatch(
        (61, 26),
        23,
        55,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#F7FAFC",
        edgecolor="#4A5568",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(encoder_box)
    ax.text(
        72.5,
        78,
        "Transformer Encoder (x2 layers)",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#2D3748",
    )

    # Sub-block 4A: Multi-Head Self-Attention
    _draw_box(
        ax, 63, 61, 19, 13,
        "Multi-Head Attention",
        "4 heads (d_k=16)\n+ Residual Add & LayerNorm",
        color="#E6FFFA", edge_color="#319795"
    )

    # Sub-block 4B: Feed Forward Network
    _draw_box(
        ax, 63, 33, 19, 13,
        "Feed-Forward Network",
        "Linear (64 → 128 → 64)\n+ Residual Add & LayerNorm",
        color="#FEEBC8", edge_color="#DD6B20"
    )

    _draw_arrow(ax, (58, 56), (63, 67), label="")
    _draw_arrow(ax, (72.5, 61), (72.5, 46), label="")

    # 5. Global Sequence Pooling & Classifier Head
    _draw_box(
        ax, 87, 36, 11, 32,
        "Classifier Head",
        "• Mean Pooling\n  (B, 125, 64) → (B, 64)\n\n• LayerNorm\n\n• Linear Classifier\n  64 → 19 classes",
        color="#F0FFF4", edge_color="#38A169"
    )
    _draw_arrow(ax, (82, 39), (87, 45), label="")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_svm_umap_pipeline(save_path: Optional[str] = None):
    """Sơ đồ quy trình trích xuất đặc trưng, giảm chiều UMAP và phân loại SVM."""
    fig, ax = _setup_figure(figsize=(13, 6.8))

    ax.text(
        50,
        96,
        "QUY TRÌNH THỰC NGHIỆM SVM VÀ UMAP TRÊN ĐẶC TRƯNG DSADS",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        50,
        92,
        "Trích xuất đặc trưng thống kê → StandardScaler → Nhánh SVM Baseline & UMAP (20 chiều)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A5568",
    )

    # 1. Feature Extraction
    _draw_box(
        ax, 3, 46, 18, 16,
        "1. Trích xuất đặc trưng",
        "Tín hiệu thô: (N, 125, 45)\n7 thống kê x 45 kênh\nOutput: Vector (N, 315)",
        color="#EBF8FF", edge_color="#3182CE"
    )

    # 2. Scaler
    _draw_box(
        ax, 25, 46, 18, 16,
        "2. StandardScaler",
        "z = (x - μ) / σ\n(Fit duy nhất trên Train p1–p5)",
        color="#FEFCBF", edge_color="#D69E2E"
    )
    _draw_arrow(ax, (21, 54), (25, 54), label="")

    # Nhánh A: SVM Baseline
    _draw_box(
        ax, 48, 60, 22, 16,
        "3A. SVM Baseline (315 chiều)",
        "Kernel RBF: K(x, x')\nC = 3.0, γ = 0.0015",
        color="#E6FFFA", edge_color="#319795"
    )
    _draw_arrow(ax, (43, 58), (48, 68), label="315 chiều")

    # Nhánh B: UMAP 20 chiều + SVM
    _draw_box(
        ax, 48, 28, 22, 16,
        "3B. UMAP (20 chiều) + SVM",
        "UMAP (315 → 20 chiều)\nSVM RBF trên 20 chiều",
        color="#FAF5FF", edge_color="#805AD5"
    )
    _draw_arrow(ax, (43, 50), (48, 36), label="20 chiều")

    # Nhánh C: UMAP 2D Visualization (Chiếu từ Scaler)
    _draw_box(
        ax, 76, 28, 21, 16,
        "4. Trực quan hóa 2D",
        "UMAP n_components=2\nPhân cụm 19 lớp trên Web Demo",
        color="#F0FFF4", edge_color="#38A169"
    )
    # Mũi tên từ Scaler sang UMAP 2D uốn cong phía dưới
    arrow_2d = patches.FancyArrowPatch(
        (34, 46),
        (76, 32),
        connectionstyle="arc3,rad=0.25",
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.3,
        color="#38A169",
        linestyle="--",
    )
    ax.add_patch(arrow_2d)
    ax.text(55, 18, "Chiếu 2D phục vụ hiển thị", fontsize=7.5, color="#2F855A", ha="center")

    # Đánh giá chung (So sánh cả 3A và 3B)
    _draw_box(
        ax, 76, 60, 21, 16,
        "5. So sánh & Đánh giá",
        "Macro-F1, Accuracy\nKhảo sát suy giảm thông tin",
        color="#FFF5F5", edge_color="#E53E3E"
    )
    _draw_arrow(ax, (70, 68), (76, 68), label="Đánh giá")
    _draw_arrow(ax, (70, 36), (76, 64), label="Đánh giá")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


if __name__ == "__main__":
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overall_pipeline(out_dir / "diagram_overall_pipeline.png")
    plot_inception_architecture(out_dir / "diagram_inception_architecture.png")
    plot_transformer_architecture(out_dir / "diagram_transformer_architecture.png")
    plot_svm_umap_pipeline(out_dir / "diagram_svm_umap_pipeline.png")
    print("Đã tái tạo 4 sơ đồ kiến trúc sạch, phân tách rõ ràng, không chồng lấn!")
