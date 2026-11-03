"""Dữ liệu tự tạo chỉ dùng kiểm tra loader, không phải kết quả DSADS."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsads import (
    read_sample,
    sample_ids,
    split_name,
    load_config,
    load_dataset,
    ACTIVITY_NAMES,
    ACTIVITY_NAMES_VI,
)


class DatasetChecks(unittest.TestCase):
    def test_subject_split(self):
        config = load_config()
        self.assertEqual([split_name(s, config) for s in range(1, 9)],
                         ["train"] * 5 + ["validation", "test", "test"])

    def test_metadata(self):
        self.assertEqual(sample_ids(Path("a19/p8/s60.txt")), (19, 8, 60))
        with self.assertRaises(ValueError):
            sample_ids(Path("a20/p8/s60.txt"))

    def test_activity_names(self):
        self.assertEqual(len(ACTIVITY_NAMES), 19)
        self.assertEqual(len(ACTIVITY_NAMES_VI), 19)
        self.assertEqual(ACTIVITY_NAMES[1], "sitting")
        self.assertEqual(ACTIVITY_NAMES_VI[1], "Ngồi")

    def test_shape_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            with path.open("w", newline="") as file:
                csv.writer(file).writerows([[0.0] * 45 for _ in range(125)])
            self.assertEqual(len(read_sample(path)), 125)
            path.write_text("1,2\n")
            with self.assertRaises(ValueError):
                read_sample(path)
            with path.open("w", newline="") as file:
                csv.writer(file).writerows([[float("nan")] * 45 for _ in range(125)])
            with self.assertRaises(ValueError):
                read_sample(path)

    def test_load_dataset_caching(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            sample_file = Path(data_dir) / "a01" / "p1" / "s01.txt"
            sample_file.parent.mkdir(parents=True)
            with sample_file.open("w", newline="") as file:
                csv.writer(file).writerows([[1.0] * 45 for _ in range(125)])

            # First load without cache
            X1, y1, sub1 = load_dataset(data_dir, cache_dir=cache_dir, use_cache=True)
            self.assertEqual(X1.shape, (1, 125, 45))
            self.assertEqual(y1[0], 0)
            self.assertEqual(sub1[0], 1)
            self.assertTrue((Path(cache_dir) / "dsads_cache.npz").exists())

            # Second load from cache
            X2, y2, sub2 = load_dataset(data_dir, cache_dir=cache_dir, use_cache=True)
            self.assertEqual(X2.shape, (1, 125, 45))
            self.assertEqual(y2[0], 0)
            self.assertEqual(sub2[0], 1)


if __name__ == "__main__":
    unittest.main()
