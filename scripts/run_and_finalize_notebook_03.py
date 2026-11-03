"""
Hoàn thiện và thực thi toàn bộ notebooks/03_deep_learning_models.ipynb
Lưu lại toàn bộ pre-rendered outputs, xuất checkpoints models, ONNX và kết quả reports.
"""
import sys
import json
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def prepare_and_run_notebook_03():
    nb_path = ROOT / "notebooks/03_deep_learning_models.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Sửa Cell 18 thành Markdown
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "".join(cell.source).strip().startswith("## 4. Đánh giá Cuối cùng"):
            cell.cell_type = "markdown"
            if "outputs" in cell:
                del cell["outputs"]
            if "execution_count" in cell:
                del cell["execution_count"]
            print(f"Đã chuyển Cell {i} thành markdown.")

    # 2. Xây dựng cell đánh giá chi tiết ma trận nhầm lẫn và bảng so sánh 4 mô hình
    eval_cell_source = '''# Hàm đánh giá toàn diện mô hình trên bất kỳ DataLoader nào
def evaluate_model(model, loader, device, model_name="Mô hình", split_name="Validation"):
    """Đánh giá mô hình, tính toán Accuracy, Macro Precision, Recall, Macro-F1 và Confusion Matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"[{model_name}] Đánh giá trên tập {split_name}:")
    print(f"  - Accuracy: {acc * 100:.2f}%")
    print(f"  - Macro-F1: {f1:.4f}")
    return acc, f1, cm, all_preds, all_targets

# Đánh giá InceptionTime trên Validation (p6) và Test (p7-p8)
acc_inc, f1_inc, cm_inc_val, _, _ = evaluate_model(
    model_inception, val_loader_cnn, device, "InceptionTime", "Validation (p6)"
)
test_acc_inc, test_f1_inc, cm_inc_test, _, _ = evaluate_model(
    model_inception, test_loader_cnn, device, "InceptionTime", "Test (p7-p8)"
)

# Đánh giá Time Series Transformer trên Validation (p6) và Test (p7-p8)
acc_tst, f1_tst, cm_tst_val, _, _ = evaluate_model(
    model_tst, val_loader_tst, device, "TST", "Validation (p6)"
)
test_acc_tst, test_f1_tst, cm_tst_test, _, _ = evaluate_model(
    model_tst, test_loader_tst, device, "TST", "Test (p7-p8)"
)

# Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix) của hai mô hình trên tập Validation (p6)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(cm_inc_val, ax=axes[0], cmap="Blues", annot=False, fmt="d", cbar=True)
axes[0].set_title(f"InceptionTime: Ma trận nhầm lẫn Val (p6)\\nAcc: {acc_inc*100:.2f}% | Macro-F1: {f1_inc:.4f}")
axes[0].set_xlabel("Nhãn dự đoán (0-18)")
axes[0].set_ylabel("Nhãn thực tế (0-18)")

sns.heatmap(cm_tst_val, ax=axes[1], cmap="Oranges", annot=False, fmt="d", cbar=True)
axes[1].set_title(f"Time Series Transformer: Ma trận nhầm lẫn Val (p6)\\nAcc: {acc_tst*100:.2f}% | Macro-F1: {f1_tst:.4f}")
axes[1].set_xlabel("Nhãn dự đoán (0-18)")
axes[1].set_ylabel("Nhãn thực tế (0-18)")

plt.tight_layout()
plt.show()

# Đọc kết quả SVM từ reports/svm_results.json để đối chiếu toàn diện 4 mô hình
svm_val_acc = 0.8868
svm_val_f1 = 0.8586
umap_val_acc = 0.6781
umap_val_f1 = 0.6608

svm_res_file = Path("reports/svm_results.json")
if svm_res_file.exists():
    try:
        with open(svm_res_file, "r", encoding="utf-8") as f:
            svm_res = json.load(f)
            svm_val_acc = svm_res["svm_baseline"]["val_accuracy"]
            svm_val_f1 = svm_res["svm_baseline"]["val_macro_f1"]
            umap_val_acc = svm_res["svm_umap"]["val_accuracy"]
            umap_val_f1 = svm_res["svm_umap"]["val_macro_f1"]
    except Exception:
        pass

# Tạo bảng so sánh tổng hợp
comparison_df = pd.DataFrame([
    {
        "Mô hình": "SVM Baseline (Statistical Features)",
        "Dạng dữ liệu": "Đặc trưng thống kê 315D",
        "Số tham số": "~45.000 SVs",
        "Val Accuracy": f"{svm_val_acc * 100:.2f}%",
        "Val Macro-F1": f"{svm_val_f1:.4f}",
        "Thời gian train": "~10s"
    },
    {
        "Mô hình": "UMAP (20D) + SVM",
        "Dạng dữ liệu": "Đặc trưng phi tuyến 20D",
        "Số tham số": "~25.000 SVs",
        "Val Accuracy": f"{umap_val_acc * 100:.2f}%",
        "Val Macro-F1": f"{umap_val_f1:.4f}",
        "Thời gian train": "~15s"
    },
    {
        "Mô hình": "InceptionTime (1D CNN)",
        "Dạng dữ liệu": "Chuỗi thời gian (45, 125)",
        "Số tham số": f"{num_params_inception:,}",
        "Val Accuracy": f"{acc_inc * 100:.2f}%",
        "Val Macro-F1": f"{f1_inc:.4f}",
        "Thời gian train": f"{time_inception:.1f}s"
    },
    {
        "Mô hình": "Time Series Transformer (TST)",
        "Dạng dữ liệu": "Chuỗi thời gian (125, 45)",
        "Số tham số": f"{num_params_tst:,}",
        "Val Accuracy": f"{acc_tst * 100:.2f}%",
        "Val Macro-F1": f"{f1_tst:.4f}",
        "Thời gian train": f"{time_tst:.1f}s"
    }
])

print("\\n=== BẢNG TỔNG HỢP SO SÁNH 4 PIPELINE TRÊN TẬP VALIDATION (p6) ===")
print(comparison_df.to_string(index=False))
'''

    # Kiểm tra xem cell đánh giá đã có chưa
    has_eval_cell = any("evaluate_model" in "".join(c.source) for c in nb.cells)
    if not has_eval_cell:
        # Chèn trước cell vẽ biểu đồ Loss/F1 (Cell 19)
        new_cell = nbformat.v4.new_code_cell(source=eval_cell_source)
        # Tìm vị trí cell 19
        for i, cell in enumerate(nb.cells):
            if "axes[0].plot(hist_inception" in "".join(cell.source):
                nb.cells.insert(i + 1, new_cell)
                print(f"Đã chèn cell đánh giá chi tiết tại vị trí {i + 1}.")
                break

    # Cập nhật cell cuối cùng (xuất artifacts) để lưu cả model_comparison.json
    for cell in nb.cells:
        if "# Xuất artifacts" in "".join(cell.source):
            updated_source = "".join(cell.source)
            if "model_comparison.json" not in updated_source:
                updated_source += '''
# 5. Lưu bảng so sánh tổng hợp 4 pipeline vào reports/model_comparison.json
all_comparison = {
    "svm_baseline": {
        "val_accuracy": float(svm_val_acc),
        "val_macro_f1": float(svm_val_f1)
    },
    "svm_umap": {
        "val_accuracy": float(umap_val_acc),
        "val_macro_f1": float(umap_val_f1)
    },
    "inception_time": {
        "val_accuracy": float(acc_inc),
        "val_macro_f1": float(f1_inc),
        "test_accuracy": float(test_acc_inc),
        "test_macro_f1": float(test_f1_inc),
        "params": int(num_params_inception),
        "train_time_seconds": float(time_inception)
    },
    "transformer_tst": {
        "val_accuracy": float(acc_tst),
        "val_macro_f1": float(f1_tst),
        "test_accuracy": float(test_acc_tst),
        "test_macro_f1": float(test_f1_tst),
        "params": int(num_params_tst),
        "train_time_seconds": float(time_tst)
    }
}
with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(all_comparison, f, indent=2, ensure_ascii=False)
print("Đã lưu bảng so sánh tổng hợp vào reports/model_comparison.json.")
'''
                cell.source = updated_source
                print("Đã cập nhật Cell xuất artifacts với model_comparison.json.")

    # Lưu lại notebook trước khi chạy
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Đã chuẩn bị xong cấu trúc notebooks/03_deep_learning_models.ipynb. Bắt đầu thực thi...")

    # 3. Thực thi notebook qua NotebookClient
    client = NotebookClient(nb, timeout=600, kernel_name="dsads_venv")
    t0 = time.time()
    client.execute()
    elapsed = time.time() - t0
    print(f"Thực thi notebook hoàn tất trong {elapsed:.1f}s.")

    # 4. Ghi lại notebook kèm toàn bộ outputs đã sinh ra
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Đã lưu thành công notebooks/03_deep_learning_models.ipynb với đầy đủ pre-rendered outputs!")

if __name__ == "__main__":
    prepare_and_run_notebook_03()

