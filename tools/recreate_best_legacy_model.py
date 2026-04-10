#!/usr/bin/env python3
"""
Recreate the best archived committed model from the legacy Optuna runs.

This script rebuilds the strongest committed checkpoint whose full model
artifact and parameters were previously tracked in git:

- timestamp: 20260204_194813
- hidden layers: (64, 64)
- input size: 8192
- mean recall: 91.82%
- min recall: 85.42%

Outputs are written under:
    artifacts/recreated_best_commit/
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_model_recreation import BEST_RUN, DATASET, OUT_DIR, build_pipeline

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    df = df.dropna(subset=["french_sentence", "topic"])

    le = LabelEncoder()
    y = le.fit_transform(df["topic"])
    categories = list(le.classes_)

    seed = BEST_RUN["random_seed"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df["french_sentence"],
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    pipeline = build_pipeline()

    print(f"Training recreated model on {len(X_trainval)} samples...")
    pipeline.fit(X_trainval, y_trainval)

    y_pred = pipeline.predict(X_test)
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
    mean_recall = float(np.mean(recalls))
    min_recall = float(np.min(recalls))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))

    model_path = OUT_DIR / "final_model_recreated_best_20260204_194813.joblib"
    json_path = OUT_DIR / "best_results_recreated_best_20260204_194813.json"

    joblib.dump(pipeline, model_path)

    payload = {
        "source_run": BEST_RUN["source_timestamp"],
        "dataset_file": str(DATASET),
        "random_seed": seed,
        "n_features": BEST_RUN["n_features"],
        "best_params": BEST_RUN["best_params"],
        "categories": categories,
        "test_metrics": {
            "mean_recall": mean_recall,
            "min_recall": min_recall,
            "balanced_accuracy": balanced_acc,
            "per_class_recall": {cat: float(rec) for cat, rec in zip(categories, recalls)},
        },
        "artifacts": {
            "model": str(model_path),
            "results": str(json_path),
        },
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved model to:   {model_path}")
    print(f"Saved metrics to: {json_path}")
    print(f"Mean recall:      {mean_recall:.4f}")
    print(f"Min recall:       {min_recall:.4f}")
    print(f"Balanced acc:     {balanced_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
