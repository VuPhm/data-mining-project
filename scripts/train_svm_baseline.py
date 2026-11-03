"""Huấn luyện và đánh giá SVM baseline cùng nhánh UMAP + SVM trên DSADS.
Chỉ fit Scaler và UMAP trên tập Train hoặc trong từng fold cross-validation.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import umap

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsads import load_config, load_dataset, ACTIVITY_NAMES_VI
from features import extract_features


def run_experiments():
    config = load_config()
    print("1. Đọc dữ liệu...")
    t0 = time.time()
    X, y, subjects = load_dataset(config["data_dir"], cache_dir=config["cache_dir"])
    print(f"   Đã đọc {len(X)} mẫu ({time.time() - t0:.2f}s).")

    print("2. Trích xuất đặc trưng thống kê (315 chiều)...")
    t0 = time.time()
    X_feat = extract_features(X)
    print(f"   Shape đặc trưng: {X_feat.shape} ({time.time() - t0:.2f}s).")

    # Phân chia train / validation / test theo cấu hình
    train_mask = np.isin(subjects, config["train_subjects"])
    val_mask = np.isin(subjects, config["validation_subjects"])
    test_mask = np.isin(subjects, config["test_subjects"])

    X_train_raw = X_feat[train_mask]
    y_train = y[train_mask]
    subs_train = subjects[train_mask]

    X_val_raw = X_feat[val_mask]
    y_val = y[val_mask]

    print(f"   Số lượng mẫu: Train={len(y_train)}, Val={len(y_val)}, Test={test_mask.sum()}")

    # 3. Đánh giá GroupKFold CV trên 5 người tập Train (p1–p5)
    print("3. Chạy GroupKFold CV (5-fold) trên tập Train...")
    logo = LeaveOneGroupOut()
    cv_scores = {"accuracy": [], "macro_f1": []}

    for fold, (tr_idx, ho_idx) in enumerate(logo.split(X_train_raw, y_train, subs_train)):
        scaler_fold = StandardScaler()
        X_tr_f = scaler_fold.fit_transform(X_train_raw[tr_idx])
        X_ho_f = scaler_fold.transform(X_train_raw[ho_idx])

        clf_f = SVC(C=3.0, gamma=0.0015, kernel="rbf", random_state=config["seed"])
        clf_f.fit(X_tr_f, y_train[tr_idx])
        preds_f = clf_f.predict(X_ho_f)

        acc_f = accuracy_score(y_train[ho_idx], preds_f)
        f1_f = f1_score(y_train[ho_idx], preds_f, average="macro")
        cv_scores["accuracy"].append(acc_f)
        cv_scores["macro_f1"].append(f1_f)
        print(f"   Fold {fold + 1} (Người {subs_train[ho_idx][0]}): Acc={acc_f*100:.2f}%, F1={f1_f:.4f}")

    cv_mean_acc = float(np.mean(cv_scores["accuracy"]))
    cv_std_acc = float(np.std(cv_scores["accuracy"]))
    cv_mean_f1 = float(np.mean(cv_scores["macro_f1"]))
    cv_std_f1 = float(np.std(cv_scores["macro_f1"]))
    print(f"   Trung bình CV: Acc = {cv_mean_acc*100:.2f}% (±{cv_std_acc*100:.2f}%), Macro-F1 = {cv_mean_f1:.4f} (±{cv_std_f1:.4f})")

    # 4. Huấn luyện SVM baseline trên toàn bộ Train (p1–p5)
    print("4. Huấn luyện SVM baseline (RBF, C=3, gamma=0.0015) trên toàn bộ Train...")
    t_train_start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    svm_baseline = SVC(C=3.0, gamma=0.0015, kernel="rbf", probability=True, random_state=config["seed"])
    svm_baseline.fit(X_train_scaled, y_train)
    t_train_svm = time.time() - t_train_start

    t_infer_start = time.time()
    svm_val_preds = svm_baseline.predict(X_val_scaled)
    t_infer_svm = time.time() - t_infer_start

    svm_val_acc = accuracy_score(y_val, svm_val_preds)
    svm_val_prec = precision_score(y_val, svm_val_preds, average="macro", zero_division=0)
    svm_val_rec = recall_score(y_val, svm_val_preds, average="macro", zero_division=0)
    svm_val_f1 = f1_score(y_val, svm_val_preds, average="macro")
    print(f"   [SVM Baseline] Val Acc: {svm_val_acc*100:.2f}%, Macro-F1: {svm_val_f1:.4f} (Thời gian train: {t_train_svm:.2f}s)")

    # 5. Huấn luyện UMAP + SVM (khảo sát giảm chiều)
    print("5. Huấn luyện UMAP + SVM (giảm xuống 20 chiều)...")
    t_umap_start = time.time()
    umap_reducer = umap.UMAP(n_components=20, random_state=config["seed"])
    X_train_umap = umap_reducer.fit_transform(X_train_scaled)
    X_val_umap = umap_reducer.transform(X_val_scaled)

    svm_umap = SVC(C=3.0, kernel="rbf", probability=True, random_state=config["seed"])
    svm_umap.fit(X_train_umap, y_train)
    t_train_umap_svm = time.time() - t_umap_start

    t_infer_start = time.time()
    umap_val_preds = svm_umap.predict(X_val_umap)
    t_infer_umap_svm = time.time() - t_infer_start

    umap_val_acc = accuracy_score(y_val, umap_val_preds)
    umap_val_prec = precision_score(y_val, umap_val_preds, average="macro", zero_division=0)
    umap_val_rec = recall_score(y_val, umap_val_preds, average="macro", zero_division=0)
    umap_val_f1 = f1_score(y_val, umap_val_preds, average="macro")
    print(f"   [UMAP + SVM] Val Acc: {umap_val_acc*100:.2f}%, Macro-F1: {umap_val_f1:.4f} (Thời gian: {t_train_umap_svm:.2f}s)")

    # 6. Fit UMAP 2D phục vụ trực quan hóa (fit trên train)
    print("6. Fit UMAP 2D phục vụ trực quan hóa không gian đặc trưng...")
    umap_2d = umap.UMAP(n_components=2, random_state=config["seed"])
    X_train_2d = umap_2d.fit_transform(X_train_scaled)
    X_val_2d = umap_2d.transform(X_val_scaled)

    # 7. Lưu Artifacts cho Web Demo và Đánh giá
    print("7. Lưu artifacts...")
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(svm_baseline, models_dir / "svm_baseline.joblib")
    joblib.dump(umap_reducer, models_dir / "umap_reducer.joblib")
    joblib.dump(svm_umap, models_dir / "svm_umap.joblib")
    joblib.dump(umap_2d, models_dir / "umap_2d.joblib")

    # 8. Lưu kết quả ra file JSON
    results = {
        "svm_baseline": {
            "val_accuracy": float(svm_val_acc),
            "val_macro_precision": float(svm_val_prec),
            "val_macro_recall": float(svm_val_rec),
            "val_macro_f1": float(svm_val_f1),
            "train_time_seconds": float(t_train_svm),
            "inference_time_seconds": float(t_infer_svm),
        },
        "svm_umap": {
            "n_components": 20,
            "val_accuracy": float(umap_val_acc),
            "val_macro_precision": float(umap_val_prec),
            "val_macro_recall": float(umap_val_rec),
            "val_macro_f1": float(umap_val_f1),
            "train_time_seconds": float(t_train_umap_svm),
            "inference_time_seconds": float(t_infer_umap_svm),
        },
        "group_kfold_cv_train": {
            "mean_accuracy": cv_mean_acc,
            "std_accuracy": cv_std_acc,
            "mean_macro_f1": cv_mean_f1,
            "std_macro_f1": cv_std_f1,
            "fold_accuracies": [float(x) for x in cv_scores["accuracy"]],
            "fold_macro_f1s": [float(x) for x in cv_scores["macro_f1"]],
        },
    }

    report_path = ROOT / "reports" / "svm_results.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"   Đã lưu kết quả tại {report_path}")
    print("Hoàn tất!")


if __name__ == "__main__":
    run_experiments()
