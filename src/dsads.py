"""Đọc và kiểm tra DSADS; giữ nhãn và subject cho từng mẫu."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SHAPE = (125, 45)

ACTIVITY_NAMES = {
    1: "sitting",
    2: "standing",
    3: "lying_back",
    4: "lying_right",
    5: "ascending_stairs",
    6: "descending_stairs",
    7: "standing_elevator",
    8: "moving_elevator",
    9: "walking_lot",
    10: "walking_treadmill_flat",
    11: "walking_treadmill_inclined",
    12: "running_treadmill",
    13: "exercising_stepper",
    14: "exercising_cross_trainer",
    15: "cycling_exercise_bike_horizontal",
    16: "cycling_exercise_bike_vertical",
    17: "rowing",
    18: "jumping",
    19: "playing_basketball",
}

ACTIVITY_NAMES_VI = {
    1: "Ngồi",
    2: "Đứng",
    3: "Nằm ngửa",
    4: "Nằm nghiêng phải",
    5: "Leo cầu thang",
    6: "Đi xuống cầu thang",
    7: "Đứng trong thang máy",
    8: "Di chuyển trong thang máy",
    9: "Đi bộ ở bãi đỗ xe",
    10: "Đi bộ máy chạy bộ (phẳng)",
    11: "Đi bộ máy chạy bộ (nghiêng 15 độ)",
    12: "Chạy máy chạy bộ (8 km/h)",
    13: "Tập máy bước (stepper)",
    14: "Tập máy cross trainer",
    15: "Đạp xe tại chỗ (tư thế ngang)",
    16: "Đạp xe tại chỗ (tư thế đứng)",
    17: "Chèo thuyền (rowing)",
    18: "Nhảy bật (jumping)",
    19: "Chơi bóng rổ",
}


def load_config(config_path=None):
    if config_path is None:
        cfg_file = ROOT / "config.json"
    else:
        cfg_file = Path(config_path)
    config = json.loads(cfg_file.read_text())
    base_dir = cfg_file.parent
    for name in ("data_dir", "cache_dir"):
        config[name] = (base_dir / config[name]).resolve()
    groups = [set(config[name]) for name in
              ("train_subjects", "validation_subjects", "test_subjects")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Các tập không được trùng subject.")
    if set.union(*groups) != set(range(1, 9)):
        raise ValueError("Cấu hình phải phân chia đủ subject 1–8.")
    return config


def find_samples(data_dir):
    """Yêu cầu cấu trúc a01/p1/s01.txt, đúng theo mô tả UCI."""
    data_dir = Path(data_dir)
    try:
        list(data_dir.iterdir())
    except PermissionError as error:
        raise PermissionError(f"Không có quyền đọc dữ liệu: {data_dir}") from error
    samples = sorted(data_dir.glob("a[0-9][0-9]/p[1-8]/s[0-9][0-9].txt"))
    if not samples:
        raise FileNotFoundError(
            f"Không tìm thấy mẫu tại {data_dir}. "
            "data_dir phải trỏ đến thư mục chứa a01 … a19."
        )
    return samples


def sample_ids(path):
    path = Path(path)
    activity = int(path.parent.parent.name[1:])
    subject = int(path.parent.name[1:])
    segment = int(path.stem[1:])
    if not (1 <= activity <= 19 and 1 <= subject <= 8 and 1 <= segment <= 60):
        raise ValueError(f"Nhãn hoặc mã mẫu ngoài phạm vi: {path}")
    return activity, subject, segment


def read_sample(path):
    with Path(path).open(newline="") as file:
        rows = [[float(value) for value in row] for row in csv.reader(file)]
    if len(rows) != 125 or any(len(row) != 45 for row in rows):
        raise ValueError(f"Mẫu phải có 125 dòng × 45 cột: {path}")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"Mẫu chứa NaN hoặc Inf: {path}")
    return rows


def split_name(subject, config):
    for name in ("train", "validation", "test"):
        if subject in config[name + "_subjects"]:
            return name
    raise ValueError(f"Subject chưa được phân tập: {subject}")


def validate_dataset(data_dir, config):
    """Đọc toàn bộ mẫu; không huấn luyện hoặc đánh giá trên test."""
    samples = find_samples(data_dir)
    seen = set()
    counts = {"train": 0, "validation": 0, "test": 0}
    for path in samples:
        activity, subject, segment = sample_ids(path)
        key = (activity, subject, segment)
        if key in seen:
            raise ValueError(f"Trùng mã mẫu: {path}")
        read_sample(path)
        seen.add(key)
        counts[split_name(subject, config)] += 1
    expected = {(a, p, s) for a in range(1, 20)
                for p in range(1, 9) for s in range(1, 61)}
    missing = expected - seen
    return {
        "data_dir": str(data_dir), "samples": len(seen),
        "expected_samples": len(expected), "missing_samples": len(missing),
        "complete": seen == expected, "sample_shape": list(SAMPLE_SHAPE),
        "split_counts": counts,
        "source": "https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities",
    }


def load_dataset(data_dir, cache_dir=None, use_cache=True):
    """Trả X: mẫu × thời gian × kênh; y: nhãn 0–18; subjects: 1–8."""
    import numpy as np

    if cache_dir and use_cache:
        cache_path = Path(cache_dir) / "dsads_cache.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return data["X"], data["y"], data["subjects"]

    samples = find_samples(data_dir)
    X = np.empty((len(samples), 125, 45), dtype=np.float32)
    y = np.empty(len(samples), dtype=np.int64)
    subjects = np.empty(len(samples), dtype=np.int64)
    for index, path in enumerate(samples):
        activity, subject, _ = sample_ids(path)
        X[index] = read_sample(path)
        y[index] = activity - 1
        subjects[index] = subject

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dsads_cache.npz"
        np.savez_compressed(cache_path, X=X, y=y, subjects=subjects)

    return X, y, subjects
