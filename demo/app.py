"""
Ứng dụng Web Demo nhận diện hoạt động con người (DSADS)
Sử dụng Streamlit phục vụ trực quan hóa tín hiệu, suy luận và so sánh 3 họ mô hình:
1. Machine Learning: SVM Baseline & UMAP + SVM
2. Deep Learning: InceptionTime (1D CNN đa tỷ lệ)
3. Deep Learning: Time Series Transformer (TST)
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Đảm bảo import được module từ src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

# Danh sách 19 hoạt động DSADS
ACTIVITY_NAMES = {
    0: "a01 - Ngồi (Sitting)",
    1: "a02 - Đứng (Standing)",
    2: "a03 - Nằm ngửa (Lying on back)",
    3: "a04 - Nằm nghiêng phải (Lying on right side)",
    4: "a05 - Đi thang máy lên (Ascending stairs / Elevator up)",
    5: "a06 - Đi thang máy xuống (Descending stairs / Elevator down)",
    6: "a07 - Đứng trong thang máy (Standing in elevator)",
    7: "a08 - Di chuyển trong thang máy (Moving in elevator)",
    8: "a09 - Đi bộ trong bãi đỗ xe (Walking in parking lot)",
    9: "a10 - Đi bộ máy chạy dốc 4 km/h (Walking on treadmill 4km/h flat)",
    10: "a11 - Đi bộ máy chạy dốc 15 độ 4 km/h (Walking 4km/h 15 deg incline)",
    11: "a12 - Chạy bộ máy chạy 8 km/h (Running on treadmill 8km/h)",
    12: "a13 - Tập thể dục trên máy chèo thuyền (Exercising on stepper)",
    13: "a14 - Tập thể dục trên máy đạp xe (Exercising on cross trainer)",
    14: "a15 - Đạp xe đạp tập nằm nghiêng (Cycling horizontal)",
    15: "a16 - Đạp xe đạp tập đứng (Cycling vertical)",
    16: "a17 - Chèo thuyền máy (Rowing)",
    17: "a18 - Nhảy bật cóc (Jumping)",
    18: "a19 - Chơi bóng rổ (Playing basketball)"
}

SENSOR_POSITIONS = {
    "Tất cả các vị trí": list(range(45)),
    "Thân người (Torso) [kênh 0-8]": list(range(0, 9)),
    "Cánh tay phải (Right Arm) [kênh 9-17]": list(range(9, 18)),
    "Cánh tay trái (Left Arm) [kênh 18-26]": list(range(18, 27)),
    "Chân phải (Right Leg) [kênh 27-35]": list(range(27, 36)),
    "Chân trái (Left Leg) [kênh 36-44]": list(range(36, 45))
}

def load_models_and_artifacts():
    """Nạp tất cả artifacts cần thiết cho suy luận."""
    models_dir = ROOT / "models"
    artifacts = {}

    # Nạp Scaler thống kê & SVM
    if joblib is not None:
        if (models_dir / "scaler.joblib").exists():
            artifacts["scaler"] = joblib.load(models_dir / "scaler.joblib")
        if (models_dir / "svm_baseline.joblib").exists():
            artifacts["svm_baseline"] = joblib.load(models_dir / "svm_baseline.joblib")
        if (models_dir / "svm_umap.joblib").exists():
            artifacts["svm_umap"] = joblib.load(models_dir / "svm_umap.joblib")
        if (models_dir / "umap_reducer.joblib").exists():
            artifacts["umap_reducer"] = joblib.load(models_dir / "umap_reducer.joblib")
        if (models_dir / "umap_2d.joblib").exists():
            artifacts["umap_2d"] = joblib.load(models_dir / "umap_2d.joblib")

    # Nạp Channel Scaler cho Deep Learning
    if (models_dir / "channel_scaler.npz").exists():
        cs = np.load(models_dir / "channel_scaler.npz")
        artifacts["channel_mean"] = cs["mean"]
        artifacts["channel_std"] = cs["std"]

    return artifacts

def extract_features_single_sample(sample_125x45):
    """Trích xuất 315 đặc trưng thống kê cho 1 mẫu (125, 45)."""
    # 7 thống kê: mean, std, min, max, median, skew, kurtosis
    from scipy.stats import skew, kurtosis
    f_mean = np.mean(sample_125x45, axis=0)
    f_std = np.std(sample_125x45, axis=0, ddof=1)
    f_min = np.min(sample_125x45, axis=0)
    f_max = np.max(sample_125x45, axis=0)
    f_median = np.median(sample_125x45, axis=0)
    f_skew = skew(sample_125x45, axis=0)
    f_kurt = kurtosis(sample_125x45, axis=0)
    feats = np.concatenate([f_mean, f_std, f_min, f_max, f_median, f_skew, f_kurt])
    return feats.reshape(1, -1)

def run_streamlit_app():
    if st is None:
        print("Vui lòng cài đặt Streamlit: pip install streamlit")
        return

    st.set_page_config(
        page_title="DSADS Activity Recognition Demo",
        page_icon="🏃",
        layout="wide"
    )

    st.title("🏃 Nhận diện Hoạt động Con người từ Cảm biến Quán tính (DSADS)")
    st.markdown("""
    **Đồ án Data Mining**: Khảo sát kỹ thuật giảm chiều UMAP và so sánh 3 họ mô hình: **Machine Learning (SVM)**, **CNN (InceptionTime)**, và **Transformer (TST)**.
    """)

    st.sidebar.header("⚙️ Cấu hình Thử nghiệm")
    artifacts = load_models_and_artifacts()

    # Tải mẫu từ cache hoặc upload
    data_source = st.sidebar.radio("Nguồn dữ liệu mẫu:", ["Mẫu có sẵn từ tập Test (p7-p8)", "Tải lên tệp cảm biến (.txt / .csv)"])

    sample_data = None
    true_label = None

    if data_source == "Mẫu có sẵn từ tập Test (p7-p8)":
        cache_path = ROOT / "data/processed/dsads_cache.npz"
        if cache_path.exists():
            cache = np.load(cache_path)
            test_mask = np.isin(cache["subjects"], [7, 8])
            test_indices = np.where(test_mask)[0]
            selected_idx = st.sidebar.selectbox(
                "Chọn số thứ tự mẫu trong tập Test:",
                options=test_indices[:100],
                format_func=lambda idx: f"Mẫu #{idx} - Hoạt động thực: {ACTIVITY_NAMES.get(int(cache['y'][idx]), 'Chưa rõ')}"
            )
            sample_data = cache["X"][selected_idx]
            true_label = int(cache["y"][selected_idx])
        else:
            st.error("Không tìm thấy tệp cache `data/processed/dsads_cache.npz`.")
    else:
        uploaded_file = st.sidebar.file_uploader("Tải lên tệp phân đoạn cảm biến (định dạng 125 dòng x 45 cột):", type=["txt", "csv"])
        if uploaded_file is not None:
            try:
                sample_data = np.loadtxt(uploaded_file, delimiter=",")
                if sample_data.shape != (125, 45):
                    st.error(f"Kích thước tệp không hợp lệ: {sample_data.shape}. Yêu cầu: (125, 45).")
                    sample_data = None
                else:
                    st.success("Tải tệp thành công! Kích thước chuẩn: (125 bước thời gian x 45 kênh).")
            except Exception as e:
                st.error(f"Lỗi khi đọc tệp: {e}")

    if sample_data is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📈 Dạng sóng tín hiệu Cảm biến Quán tính")
            sensor_group = st.selectbox("Chọn cụm cảm biến để trực quan hóa:", list(SENSOR_POSITIONS.keys()))
            channels_to_plot = SENSOR_POSITIONS[sensor_group]

            df_plot = pd.DataFrame(sample_data[:, channels_to_plot[:3]])
            df_plot.columns = [f"Kênh {c}" for c in channels_to_plot[:3]]
            st.line_chart(df_plot)
            st.caption(f"Hiển thị tín hiệu dao động qua 125 bước thời gian (5 giây tại 25 Hz).")

        with col2:
            st.subheader("🤖 Kết quả Dự đoán & Độ tin cậy")
            if true_label is not None:
                st.info(f"🎯 **Nhãn thực tế (Ground Truth)**: `{ACTIVITY_NAMES.get(true_label, 'N/A')}`")

            # Tính đặc trưng cho SVM
            feats = extract_features_single_sample(sample_data)
            
            # Dự đoán SVM Baseline
            svm_pred_name = "N/A"
            svm_conf = 0.0
            if "scaler" in artifacts and "svm_baseline" in artifacts:
                feats_scaled = artifacts["scaler"].transform(feats)
                probs = artifacts["svm_baseline"].predict_proba(feats_scaled)[0]
                pred_cls = int(np.argmax(probs))
                svm_conf = float(probs[pred_cls])
                svm_pred_name = ACTIVITY_NAMES.get(pred_cls, f"Lớp {pred_cls}")

            # Dự đoán UMAP + SVM
            umap_pred_name = "N/A"
            umap_conf = 0.0
            if "scaler" in artifacts and "umap_reducer" in artifacts and "svm_umap" in artifacts:
                feats_scaled = artifacts["scaler"].transform(feats)
                feats_umap = artifacts["umap_reducer"].transform(feats_scaled)
                probs_umap = artifacts["svm_umap"].predict_proba(feats_umap)[0]
                pred_cls_u = int(np.argmax(probs_umap))
                umap_conf = float(probs_umap[pred_cls_u])
                umap_pred_name = ACTIVITY_NAMES.get(pred_cls_u, f"Lớp {pred_cls_u}")

            res_df = pd.DataFrame([
                {"Mô hình": "SVM Baseline", "Dự đoán": svm_pred_name, "Độ tin cậy": f"{svm_conf*100:.1f}%"},
                {"Mô hình": "UMAP (20D) + SVM", "Dự đoán": umap_pred_name, "Độ tin cậy": f"{umap_conf*100:.1f}%"},
                {"Mô hình": "InceptionTime (CNN)", "Dự đoán": "Đang đồng bộ trọng số", "Độ tin cậy": "-"},
                {"Mô hình": "Time Series Transformer", "Dự đoán": "Đang đồng bộ trọng số", "Độ tin cậy": "-"}
            ])
            st.table(res_df)

        st.markdown("---")
        st.subheader("🗺️ Vị trí mẫu trên Không gian đặc trưng UMAP 2D")
        if "scaler" in artifacts and "umap_2d" in artifacts:
            feats_scaled = artifacts["scaler"].transform(feats)
            pos_2d = artifacts["umap_2d"].transform(feats_scaled)[0]
            st.write(f"Tọa độ chiếu UMAP 2D của mẫu đang xét: **X = {pos_2d[0]:.2f}, Y = {pos_2d[1]:.2f}**")
            st.caption("Mẫu thử nghiệm được chiếu trực tiếp từ không gian 315 chiều xuống 2D bằng mô hình UMAP học từ tập Train.")
        else:
            st.info("Chưa nạp được mô hình UMAP 2D từ `models/umap_2d.joblib`.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        print("DSADS Demo CLI mode sẵn sàng.")
    else:
        run_streamlit_app()

