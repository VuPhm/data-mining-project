"""Trích xuất đặc trưng thống kê miền thời gian từ tín hiệu cảm biến DSADS."""
from typing import List, Tuple
import numpy as np
from scipy import stats

FEATURE_NAMES_PER_CHANNEL = ["mean", "std", "min", "max", "median", "skew", "kurt"]


def get_feature_names(num_channels: int = 45) -> List[str]:
    """Trả về danh sách tên cho tất cả 315 đặc trưng."""
    names = []
    for stat in FEATURE_NAMES_PER_CHANNEL:
        for ch in range(num_channels):
            names.append(f"ch{ch:02d}_{stat}")
    return names


def extract_features(X: np.ndarray) -> np.ndarray:
    """Trích xuất 7 đặc trưng thống kê cho mỗi kênh cảm biến.

    Input:
        X: mảng numpy (N, 125, 45)
    Output:
        Mảng numpy (N, 315) kiểu float32
    """
    if X.ndim != 3 or X.shape[1] != 125 or X.shape[2] != 45:
        raise ValueError(f"X phải có kích thước (N, 125, 45), nhận: {X.shape}")

    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    mins = np.min(X, axis=1)
    maxs = np.max(X, axis=1)
    medians = np.median(X, axis=1)
    skews = stats.skew(X, axis=1)
    kurts = stats.kurtosis(X, axis=1)

    features = np.hstack([means, stds, mins, maxs, medians, skews, kurts])
    return features.astype(np.float32)
