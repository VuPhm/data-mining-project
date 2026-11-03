"""Tạo và thực thi notebook 02_svm_umap_baseline.ipynb."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# Huấn luyện và Đánh giá SVM & UMAP trên DSADS\n\n"
        "Notebook này thực hiện:\n"
        "1. Trích xuất đặc trưng thống kê (315 chiều) từ chuỗi thời gian cảm biến.\n"
        "2. Đánh giá tính tổng quát hóa qua 5-Fold GroupKFold Cross-Validation theo đối tượng (subject) trên tập huấn luyện (p1–p5).\n"
        "3. Huấn luyện mô hình SVM baseline (RBF kernel) trên toàn bộ tập huấn luyện và đánh giá trên tập kiểm định (p6).\n"
        "4. Khảo sát kỹ thuật giảm chiều UMAP kết hợp với SVM.\n"
        "5. Trực quan hóa không gian đặc trưng UMAP 2D và ma trận nhầm lẫn (confusion matrix)."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score\n"
        "from sklearn.model_selection import LeaveOneGroupOut\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.svm import SVC\n"
        "import umap\n"
        "import joblib\n\n"
        "# Thêm thư mục mã nguồn nội bộ\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / \"config.json\").exists():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT / \"src\"))\n"
        "from dsads import load_config, load_dataset, ACTIVITY_NAMES_VI\n"
        "from features import extract_features\n\n"
        "config = load_config()\n"
        "print(\"Cấu hình đã tải thành công:\", config)"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Nạp dữ liệu và trích xuất đặc trưng thống kê\n"
        "Mỗi mẫu là tín hiệu 125 bước thời gian $\\times$ 45 kênh cảm biến. Ta trích xuất 7 đặc trưng thống kê miền thời gian "
        "(trung bình, độ lệch chuẩn, giá trị nhỏ nhất, giá trị lớn nhất, trung vị, độ lệch skewness, độ nhọn kurtosis) "
        "cho mỗi kênh, tạo thành vector đặc trưng $7 \\times 45 = 315$ chiều."
    ),
    nbf.v4.new_code_cell(
        "# Đọc dữ liệu từ bộ nhớ đệm cache\n"
        "X, y, subjects = load_dataset(config[\"data_dir\"], cache_dir=config[\"cache_dir\"])\n"
        "print(\"Dữ liệu gốc X:\", X.shape, \"y:\", y.shape)\n\n"
        "# Trích xuất đặc trưng\n"
        "X_features = extract_features(X)\n"
        "print(\"Dữ liệu đặc trưng X_features:\", X_features.shape)\n\n"
        "# Phân chia theo đối tượng (subject)\n"
        "train_mask = np.isin(subjects, config[\"train_subjects\"])\n"
        "val_mask = np.isin(subjects, config[\"validation_subjects\"])\n"
        "test_mask = np.isin(subjects, config[\"test_subjects\"])\n\n"
        "X_train, y_train, subs_train = X_features[train_mask], y[train_mask], subjects[train_mask]\n"
        "X_val, y_val, subs_val = X_features[val_mask], y[val_mask], subjects[val_mask]\n"
        "print(f\"Số lượng mẫu: Train={len(y_train)} (p1–p5), Val={len(y_val)} (p6), Test={test_mask.sum()} (p7–p8)\")"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Kiểm chứng chéo theo đối tượng (GroupKFold CV) trên tập Train\n"
        "Để đánh giá khả năng nhận diện khi gặp một người hoàn toàn mới (Subject-independent), "
        "ta thực hiện 5-fold Leave-One-Subject-Out CV trên 5 người thuộc tập Train (p1–p5). "
        "Bộ chuẩn hóa `StandardScaler` chỉ được fit trên tập train của từng fold."
    ),
    nbf.v4.new_code_cell(
        "logo = LeaveOneGroupOut()\n"
        "fold_results = []\n\n"
        "for fold, (tr_idx, ho_idx) in enumerate(logo.split(X_train, y_train, subs_train)):\n"
        "    held_out_sub = subs_train[ho_idx][0]\n"
        "    scaler_cv = StandardScaler()\n"
        "    X_tr_f = scaler_cv.fit_transform(X_train[tr_idx])\n"
        "    X_ho_f = scaler_cv.transform(X_train[ho_idx])\n"
        "    \n"
        "    clf_cv = SVC(C=3.0, gamma=0.0015, kernel='rbf', random_state=config['seed'])\n"
        "    clf_cv.fit(X_tr_f, y_train[tr_idx])\n"
        "    preds_f = clf_cv.predict(X_ho_f)\n"
        "    \n"
        "    acc = accuracy_score(y_train[ho_idx], preds_f)\n"
        "    f1 = f1_score(y_train[ho_idx], preds_f, average='macro')\n"
        "    fold_results.append({\n"
        "        'Fold': fold + 1,\n"
        "        'Held-out Subject': f'p{held_out_sub}',\n"
        "        'Accuracy (%)': round(acc * 100, 2),\n"
        "        'Macro-F1': round(f1, 4)\n"
        "    })\n\n"
        "df_cv = pd.DataFrame(fold_results)\n"
        "print(df_cv.to_string(index=False))\n"
        "print(f\"\\nTrung bình CV: Accuracy = {df_cv['Accuracy (%)'].mean():.2f}% (±{df_cv['Accuracy (%)'].std():.2f}%), \"\n"
        "      f\"Macro-F1 = {df_cv['Macro-F1'].mean():.4f} (±{df_cv['Macro-F1'].std():.4f})\")"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Huấn luyện mô hình SVM Baseline\n"
        "Chuẩn hóa dữ liệu bằng `StandardScaler` (chỉ fit trên Train) rồi huấn luyện mô hình SVM kernel RBF "
        "với tham số tối ưu $C=3.0, \\gamma=0.0015$."
    ),
    nbf.v4.new_code_cell(
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_val_scaled = scaler.transform(X_val)\n\n"
        "svm_baseline = SVC(C=3.0, gamma=0.0015, kernel='rbf', random_state=config['seed'])\n"
        "svm_baseline.fit(X_train_scaled, y_train)\n\n"
        "y_pred_svm = svm_baseline.predict(X_val_scaled)\n"
        "acc_svm = accuracy_score(y_val, y_pred_svm)\n"
        "f1_svm = f1_score(y_val, y_pred_svm, average='macro')\n"
        "print(f\"[SVM Baseline trên Validation p6] Accuracy: {acc_svm * 100:.2f}%, Macro-F1: {f1_svm:.4f}\")"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Ma trận nhầm lẫn (Confusion Matrix) của SVM Baseline\n"
        "Quan sát sự phân loại giữa 19 hoạt động để phát hiện các hoạt động dễ bị nhầm lẫn."
    ),
    nbf.v4.new_code_cell(
        "plt.figure(figsize=(12, 10))\n"
        "labels_vi = [ACTIVITY_NAMES_VI[i + 1] for i in range(19)]\n"
        "cm = confusion_matrix(y_val, y_pred_svm)\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_vi, yticklabels=labels_vi)\n"
        "plt.title(f'Ma trận nhầm lẫn SVM Baseline trên tập Validation (Acc = {acc_svm * 100:.2f}%)', fontsize=13)\n"
        "plt.xlabel('Dự đoán', fontsize=11)\n"
        "plt.ylabel('Thực tế', fontsize=11)\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Khảo sát kỹ thuật giảm chiều: UMAP + SVM\n"
        "Áp dụng UMAP để giảm số chiều từ 315 xuống 20 chiều, sau đó huấn luyện mô hình SVM trên không gian đã nén. "
        "Bộ giảm chiều UMAP được fit độc quyền trên tập Train."
    ),
    nbf.v4.new_code_cell(
        "umap_reducer = umap.UMAP(n_components=20, random_state=config['seed'])\n"
        "X_train_umap = umap_reducer.fit_transform(X_train_scaled)\n"
        "X_val_umap = umap_reducer.transform(X_val_scaled)\n\n"
        "svm_umap = SVC(C=3.0, kernel='rbf', random_state=config['seed'])\n"
        "svm_umap.fit(X_train_umap, y_train)\n\n"
        "y_pred_umap = svm_umap.predict(X_val_umap)\n"
        "acc_umap = accuracy_score(y_val, y_pred_umap)\n"
        "f1_umap = f1_score(y_val, y_pred_umap, average='macro')\n"
        "print(f\"[UMAP (20 chiều) + SVM trên Validation p6] Accuracy: {acc_umap * 100:.2f}%, Macro-F1: {f1_umap:.4f}\")"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Trực quan hóa không gian đặc trưng UMAP 2D\n"
        "Chiếu các mẫu hoạt động từ 315 chiều xuống 2D bằng UMAP để trực quan hóa sự phân cụm tự nhiên của 19 hoạt động."
    ),
    nbf.v4.new_code_cell(
        "umap_2d = umap.UMAP(n_components=2, random_state=config['seed'])\n"
        "X_train_2d = umap_2d.fit_transform(X_train_scaled)\n"
        "X_val_2d = umap_2d.transform(X_val_scaled)\n\n"
        "plt.figure(figsize=(13, 8))\n"
        "scatter = plt.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, cmap='tab20', alpha=0.8, s=25)\n"
        "cbar = plt.colorbar(scatter, ticks=range(19))\n"
        "cbar.ax.set_yticklabels(labels_vi)\n"
        "plt.title('Không gian đặc trưng UMAP 2D trên tập Validation (Đối tượng p6)', fontsize=13)\n"
        "plt.xlabel('UMAP Chiều 1', fontsize=11)\n"
        "plt.ylabel('UMAP Chiều 2', fontsize=11)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 7. Bảng tổng hợp so sánh mô hình\n"
        "So sánh hiệu năng giữa SVM baseline không giảm chiều và nhánh UMAP + SVM trên cùng tập Validation."
    ),
    nbf.v4.new_code_cell(
        "df_compare = pd.DataFrame([\n"
        "    {\n"
        "        'Phương pháp': 'SVM Baseline (RBF)',\n"
        "        'Số chiều đặc trưng': 315,\n"
        "        'Validation Accuracy (%)': round(acc_svm * 100, 2),\n"
        "        'Validation Macro-F1': round(f1_svm, 4),\n"
        "        'Ghi chú': 'Full đặc trưng thống kê'\n"
        "    },\n"
        "    {\n"
        "        'Phương pháp': 'UMAP + SVM',\n"
        "        'Số chiều đặc trưng': 20,\n"
        "        'Validation Accuracy (%)': round(acc_umap * 100, 2),\n"
        "        'Validation Macro-F1': round(f1_umap, 4),\n"
        "        'Ghi chú': 'Giảm chiều phi tuyến tính'\n"
        "    }\n"
        "])\n"
        "print(df_compare.to_string(index=False))\n"
        "df_compare"
    )
]

nb.cells = cells
nb_path = ROOT / "notebooks" / "02_svm_umap_baseline.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Đã tạo notebook tại {nb_path}")
