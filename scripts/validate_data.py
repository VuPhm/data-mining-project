"""Chạy từ terminal: python scripts/validate_data.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsads import load_config, validate_dataset

if __name__ == "__main__":
    try:
        config = load_config()
        result = validate_dataset(config["data_dir"], config)
        output = ROOT / "reports" / "data_validation.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["complete"]:
            sys.exit(1)
    except (OSError, ValueError) as error:
        print(f"Không thể kiểm tra DSADS: {error}", file=sys.stderr)
        sys.exit(1)
