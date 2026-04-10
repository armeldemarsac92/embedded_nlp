from __future__ import annotations

import glob
import json
import numbers
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "legacy"


def load_latest_resources() -> tuple[dict, str, str, Path]:
    """
    Return the latest legacy Optuna model/config pair from `artifacts/legacy`.
    """
    json_files = sorted(Path(path) for path in glob.glob(str(ARTIFACT_DIR / "best_results_*.json")))
    model_files = sorted(Path(path) for path in glob.glob(str(ARTIFACT_DIR / "final_model_*.joblib")))
    if not json_files or not model_files:
        raise FileNotFoundError(f"No model/config pair found in: {ARTIFACT_DIR}")

    json_by_stamp = {path.stem.removeprefix("best_results_"): path for path in json_files}
    model_by_stamp = {path.stem.removeprefix("final_model_"): path for path in model_files}
    common_stamps = sorted(set(json_by_stamp) & set(model_by_stamp))
    if not common_stamps:
        raise FileNotFoundError(f"No matching model/config pair found in: {ARTIFACT_DIR}")

    latest_stamp = common_stamps[-1]
    latest_json = json_by_stamp[latest_stamp]
    latest_model = model_by_stamp[latest_stamp]

    with open(latest_json, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    return config_data, str(latest_model), str(latest_json), ARTIFACT_DIR


def decode_topic(label, categories: list[str]) -> str:
    """Map sklearn class ids back to topic names when possible."""
    if isinstance(label, numbers.Integral):
        idx = int(label)
        if 0 <= idx < len(categories):
            return categories[idx]

    label_str = str(label)
    if label_str.isdigit():
        idx = int(label_str)
        if 0 <= idx < len(categories):
            return categories[idx]

    return label_str
