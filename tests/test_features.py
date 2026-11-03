"""Kiểm tra trích xuất đặc trưng thống kê."""
import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features, get_feature_names


class FeatureChecks(unittest.TestCase):
    def test_feature_shape_and_names(self):
        names = get_feature_names()
        self.assertEqual(len(names), 315)
        self.assertEqual(names[0], "ch00_mean")

        dummy_X = np.zeros((4, 125, 45), dtype=np.float32)
        feats = extract_features(dummy_X)
        self.assertEqual(feats.shape, (4, 315))
        self.assertEqual(feats.dtype, np.float32)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            extract_features(np.zeros((4, 100, 45)))

    def test_feature_values(self):
        # Mẫu với giá trị tăng tuyến tính từ 0 đến 124 ở kênh 0
        X = np.zeros((1, 125, 45), dtype=np.float32)
        X[0, :, 0] = np.arange(125, dtype=np.float32)

        feats = extract_features(X)
        # mean kênh 0 = 62.0
        self.assertAlmostEqual(feats[0, 0], 62.0, places=4)
        # min kênh 0 = 0.0, max kênh 0 = 124.0
        # channel 0 min is at index 45*2 = 90
        self.assertAlmostEqual(feats[0, 45 * 2], 0.0, places=4)
        # channel 0 max is at index 45*3 = 135
        self.assertAlmostEqual(feats[0, 45 * 3], 124.0, places=4)


if __name__ == "__main__":
    unittest.main()
