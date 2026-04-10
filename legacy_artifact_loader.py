from __future__ import annotations

import glob
import json
import numbers
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIRS = [
    PROJECT_ROOT / "artifacts" / "recreated_best_commit",
    PROJECT_ROOT / "artifacts" / "legacy",
]


def load_latest_resources() -> tuple[dict, str, str, Path]:
    """
    Return the preferred model/config pair.

    Preference order:
    1. `artifacts/recreated_best_commit`
    2. `artifacts/legacy`
    """
    for artifact_dir in ARTIFACT_DIRS:
        json_files = sorted(glob.glob(str(artifact_dir / "best_results_*.json")))
        model_files = sorted(glob.glob(str(artifact_dir / "final_model_*.joblib")))
        if not json_files or not model_files:
            continue

        latest_json = json_files[-1]
        latest_model = model_files[-1]
        with open(latest_json, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        return config_data, latest_model, latest_json, artifact_dir

    search_roots = ", ".join(str(path) for path in ARTIFACT_DIRS)
    raise FileNotFoundError(f"No model/config pair found in: {search_roots}")


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
