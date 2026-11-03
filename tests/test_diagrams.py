"""Kiểm thử tự động cho module sinh sơ đồ hệ thống diagrams.py."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần màn hình
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagrams import (
    plot_overall_pipeline,
    plot_inception_architecture,
    plot_transformer_architecture,
    plot_svm_umap_pipeline,
)


def test_diagram_generation(tmp_path):
    """Kiểm tra việc khởi tạo figure và xuất file ảnh cho cả 4 sơ đồ."""
    fig1 = plot_overall_pipeline(tmp_path / "test_pipeline.png")
    assert isinstance(fig1, plt.Figure)
    assert (tmp_path / "test_pipeline.png").exists()

    fig2 = plot_inception_architecture(tmp_path / "test_inception.png")
    assert isinstance(fig2, plt.Figure)
    assert (tmp_path / "test_inception.png").exists()

    fig3 = plot_transformer_architecture(tmp_path / "test_tst.png")
    assert isinstance(fig3, plt.Figure)
    assert (tmp_path / "test_tst.png").exists()

    fig4 = plot_svm_umap_pipeline(tmp_path / "test_svm.png")
    assert isinstance(fig4, plt.Figure)
    assert (tmp_path / "test_svm.png").exists()

    plt.close("all")
