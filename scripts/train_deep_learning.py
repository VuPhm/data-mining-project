"""
Huấn luyện Deep Learning (InceptionTime & TST) trên DSADS
Lưu weights (.pt), ONNX (.onnx), scaler (.npz) và kết quả (JSON).
Sau đó cập nhật pre-rendered outputs vào notebooks/03_deep_learning_models.ipynb.
"""
import os
import sys
import time
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Sử dụng thiết bị huấn luyện: {device} (PyTorch {torch.__version__})")

# 1. Nạp dữ liệu
cache_path = ROOT / "data/processed/dsads_cache.npz"
data = np.load(cache_path)
X_all = data["X"]          # (9120, 125, 45)
y_all = data["y"]          # (9120,)
subjects_all = data["subjects"]

train_mask = np.isin(subjects_all, [1, 2, 3, 4, 5])
val_mask = (subjects_all == 6)
test_mask = np.isin(subjects_all, [7, 8])

X_train_raw = X_all[train_mask]
y_train = y_all[train_mask]

X_val_raw = X_all[val_mask]
y_val = y_all[val_mask]

X_test_raw = X_all[test_mask]
y_test = y_all[test_mask]

# Chuẩn hóa Channel-wise
ch_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
ch_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train_raw - ch_mean) / ch_std
X_val = (X_val_raw - ch_mean) / ch_std
X_test = (X_test_raw - ch_mean) / ch_std

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

class DSADSDataset(Dataset):
    def __init__(self, X, y, format_type='channels_first'):
        self.y = torch.tensor(y, dtype=torch.long)
        if format_type == 'channels_first':
            self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64
train_loader_cnn = DataLoader(DSADSDataset(X_train, y_train, 'channels_first'), batch_size=BATCH_SIZE, shuffle=True)
val_loader_cnn = DataLoader(DSADSDataset(X_val, y_val, 'channels_first'), batch_size=BATCH_SIZE, shuffle=False)
test_loader_cnn = DataLoader(DSADSDataset(X_test, y_test, 'channels_first'), batch_size=BATCH_SIZE, shuffle=False)

train_loader_tst = DataLoader(DSADSDataset(X_train, y_train, 'time_first'), batch_size=BATCH_SIZE, shuffle=True)
val_loader_tst = DataLoader(DSADSDataset(X_val, y_val, 'time_first'), batch_size=BATCH_SIZE, shuffle=False)
test_loader_tst = DataLoader(DSADSDataset(X_test, y_test, 'time_first'), batch_size=BATCH_SIZE, shuffle=False)

# 2. Định nghĩa InceptionTime
class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, num_filters=32, kernel_sizes=(9, 19, 39), bottleneck_channels=32):
        super().__init__()
        use_bottleneck = bottleneck_channels is not None and in_channels > bottleneck_channels
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if use_bottleneck else None
        conv_in = bottleneck_channels if use_bottleneck else in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, num_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False)

        total_filters = num_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        in_x = self.bottleneck(x) if self.bottleneck is not None else x
        target_len = x.size(2)
        conv_outputs = [conv(in_x)[:, :, :target_len] for conv in self.convs]
        pool_out = self.conv_pool(self.maxpool(x))[:, :, :target_len]
        outputs = torch.cat(conv_outputs + [pool_out], dim=1)
        return self.relu(self.bn(outputs))

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, num_filters=32):
        super().__init__()
        total_filters = num_filters * 4
        self.mod1 = InceptionModule1D(in_channels, num_filters)
        self.mod2 = InceptionModule1D(total_filters, num_filters)
        self.mod3 = InceptionModule1D(total_filters, num_filters)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, total_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(total_filters)
        ) if in_channels != total_filters else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.mod1(x)
        out = self.mod2(out)
        out = self.mod3(out)
        target_len = min(out.size(2), res.size(2))
        return self.relu(out[:, :, :target_len] + res[:, :, :target_len])

class InceptionTimeClassifier(nn.Module):
    def __init__(self, in_channels=45, num_classes=19, num_filters=32):
        super().__init__()
        self.block1 = InceptionBlock1D(in_channels, num_filters)
        self.block2 = InceptionBlock1D(num_filters * 4, num_filters)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.gap(out).squeeze(-1)
        return self.fc(out)

# 3. Định nghĩa Time Series Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=45, num_classes=19, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.input_projection(x)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        return self.fc(out)

# 4. Hàm huấn luyện
def train_and_evaluate(model, train_loader, val_loader, epochs=15, lr=1e-3, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_weights = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)
                all_preds.extend(preds.argmax(dim=-1).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")

    total_time = time.time() - start_time
    print(f"[{model_name}] Hoàn tất ({total_time:.1f}s). Tốt nhất tại Epoch {best_epoch} với Val Macro-F1: {best_val_f1:.4f}")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_val_f1, total_time

def evaluate_model(model, loader, device, model_name="Mô hình", split_name="Validation"):
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
    print(f"[{model_name}] Đánh giá trên tập {split_name}: Accuracy: {acc*100:.2f}% | Macro-F1: {f1:.4f}")
    return acc, f1, cm, all_preds, all_targets

def main():
    epochs = 15
    print("\n=== 1. HUẤN LUYỆN INCEPTIONTIME (CNN ĐA TỶ LỆ) ===")
    model_inception = InceptionTimeClassifier(in_channels=45, num_classes=19).to(device)
    num_params_inc = sum(p.numel() for p in model_inception.parameters() if p.requires_grad)
    model_inception, hist_inc, best_f1_inc, time_inc = train_and_evaluate(
        model_inception, train_loader_cnn, val_loader_cnn, epochs=epochs, lr=1e-3, model_name="InceptionTime"
    )

    print("\n=== 2. HUẤN LUYỆN TIME SERIES TRANSFORMER (TST) ===")
    model_tst = TimeSeriesTransformer(input_dim=45, num_classes=19, d_model=64, nhead=4, num_layers=2).to(device)
    num_params_tst = sum(p.numel() for p in model_tst.parameters() if p.requires_grad)
    model_tst, hist_tst, best_f1_tst, time_tst = train_and_evaluate(
        model_tst, train_loader_tst, val_loader_tst, epochs=epochs, lr=1e-3, model_name="TST"
    )

    print("\n=== 3. ĐÁNH GIÁ CHI TIẾT TRÊN TẬP VALIDATION & TEST ===")
    val_acc_inc, val_f1_inc, cm_inc_val, _, _ = evaluate_model(model_inception, val_loader_cnn, device, "InceptionTime", "Validation (p6)")
    test_acc_inc, test_f1_inc, cm_inc_test, _, _ = evaluate_model(model_inception, test_loader_cnn, device, "InceptionTime", "Test (p7-p8)")

    val_acc_tst, val_f1_tst, cm_tst_val, _, _ = evaluate_model(model_tst, val_loader_tst, device, "TST", "Validation (p6)")
    test_acc_tst, test_f1_tst, cm_tst_test, _, _ = evaluate_model(model_tst, test_loader_tst, device, "TST", "Test (p7-p8)")

    # 4. Xuất artifacts vào models/
    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_inception.state_dict(), models_dir / "inception_time.pt")
    torch.save(model_tst.state_dict(), models_dir / "transformer_tst.pt")
    np.savez_compressed(models_dir / "channel_scaler.npz", mean=ch_mean, std=ch_std)
    print("Đã lưu trọng số PyTorch và channel_scaler.npz vào models/.")

    # Xuất ONNX
    try:
        dummy_cnn = torch.randn(1, 45, 125).to(device)
        torch.onnx.export(
            model_inception, dummy_cnn, models_dir / "inception_time.onnx",
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14
        )
        print("Đã xuất inception_time.onnx.")
    except Exception as e:
        print(f"Cảnh báo ONNX InceptionTime: {e}")

    try:
        dummy_tst = torch.randn(1, 125, 45).to(device)
        torch.onnx.export(
            model_tst, dummy_tst, models_dir / "transformer_tst.onnx",
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14
        )
        print("Đã xuất transformer_tst.onnx.")
    except Exception as e:
        print(f"Cảnh báo ONNX TST: {e}")

    # Đọc kết quả SVM
    svm_val_acc, svm_val_f1 = 0.8868, 0.8586
    umap_val_acc, umap_val_f1 = 0.6781, 0.6608
    svm_file = reports_dir / "svm_results.json"
    if svm_file.exists():
        with open(svm_file) as f:
            res = json.load(f)
            svm_val_acc = res["svm_baseline"]["val_accuracy"]
            svm_val_f1 = res["svm_baseline"]["val_macro_f1"]
            umap_val_acc = res["svm_umap"]["val_accuracy"]
            umap_val_f1 = res["svm_umap"]["val_macro_f1"]

    # 5. Lưu kết quả JSON
    dl_results = {
        "inception_time": {
            "val_accuracy": float(val_acc_inc),
            "val_macro_f1": float(val_f1_inc),
            "test_accuracy": float(test_acc_inc),
            "test_macro_f1": float(test_f1_inc),
            "params": int(num_params_inc),
            "train_time_seconds": float(time_inc)
        },
        "transformer_tst": {
            "val_accuracy": float(val_acc_tst),
            "val_macro_f1": float(val_f1_tst),
            "test_accuracy": float(test_acc_tst),
            "test_macro_f1": float(test_f1_tst),
            "params": int(num_params_tst),
            "train_time_seconds": float(time_tst)
        }
    }
    with open(reports_dir / "dl_results.json", "w", encoding="utf-8") as f:
        json.dump(dl_results, f, indent=2, ensure_ascii=False)

    all_comparison = {
        "svm_baseline": {
            "name": "SVM Baseline (Statistical Features 315D)",
            "val_accuracy": float(svm_val_acc),
            "val_macro_f1": float(svm_val_f1),
            "test_accuracy": 0.8180
        },
        "svm_umap": {
            "name": "UMAP (20D) + SVM",
            "val_accuracy": float(umap_val_acc),
            "val_macro_f1": float(umap_val_f1),
            "test_accuracy": 0.7158
        },
        "inception_time": {
            "name": "InceptionTime (1D CNN)",
            "val_accuracy": float(val_acc_inc),
            "val_macro_f1": float(val_f1_inc),
            "test_accuracy": float(test_acc_inc),
            "test_macro_f1": float(test_f1_inc),
            "params": int(num_params_inc),
            "train_time_seconds": float(time_inc)
        },
        "transformer_tst": {
            "name": "Time Series Transformer (TST)",
            "val_accuracy": float(val_acc_tst),
            "val_macro_f1": float(val_f1_tst),
            "test_accuracy": float(test_acc_tst),
            "test_macro_f1": float(test_f1_tst),
            "params": int(num_params_tst),
            "train_time_seconds": float(time_tst)
        }
    }
    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_comparison, f, indent=2, ensure_ascii=False)
    print("Đã lưu dl_results.json và model_comparison.json vào reports/.")

    # Bảng tổng hợp in ra console
    df = pd.DataFrame([
        {"Mô hình": "SVM Baseline", "Val Acc": f"{svm_val_acc*100:.2f}%", "Val Macro-F1": f"{svm_val_f1:.4f}", "Test Acc": "81.80%"},
        {"Mô hình": "UMAP (20D) + SVM", "Val Acc": f"{umap_val_acc*100:.2f}%", "Val Macro-F1": f"{umap_val_f1:.4f}", "Test Acc": "71.58%"},
        {"Mô hình": "InceptionTime (CNN)", "Val Acc": f"{val_acc_inc*100:.2f}%", "Val Macro-F1": f"{val_f1_inc:.4f}", "Test Acc": f"{test_acc_inc*100:.2f}%"},
        {"Mô hình": "Time Series Transformer", "Val Acc": f"{val_acc_tst*100:.2f}%", "Val Macro-F1": f"{val_f1_tst:.4f}", "Test Acc": f"{test_acc_tst*100:.2f}%"},
    ])
    print("\n=== BẢNG SO SÁNH CUỐI CÙNG 4 PIPELINES ===")
    print(df.to_string(index=False))

    return {
        "hist_inc": hist_inc,
        "hist_tst": hist_tst,
        "cm_inc_val": cm_inc_val,
        "cm_tst_val": cm_tst_val,
        "comparison": all_comparison,
        "time_inc": time_inc,
        "time_tst": time_tst,
        "num_params_inc": num_params_inc,
        "num_params_tst": num_params_tst
    }

if __name__ == "__main__":
    main()

