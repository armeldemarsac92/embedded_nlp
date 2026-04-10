#!/usr/bin/env python3
"""
Evaluate the latest legacy Optuna model against a curated sentence suite and an
INT8-quantized copy of the same classifier.

Outputs:
  - artifacts/evaluation/sentence_suite_quantization_report.md
  - artifacts/evaluation/sentence_suite_quantization_results.json
"""

from __future__ import annotations

import copy
import json
import string
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_exporter import compute_quantization_error, quantize_symmetric
from legacy_artifact_loader import load_latest_resources as load_latest_artifacts


SUITE_PATH = ROOT / "data" / "evaluation_sentence_suite.txt"
OUT_DIR = ROOT / "artifacts" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "sentence_suite_quantization_results.json"
MD_OUT = OUT_DIR / "sentence_suite_quantization_report.md"

STOP_WORDS = set()


@dataclass(frozen=True)
class SentenceCase:
    expected: str
    variant: str
    sentence: str


class CustomAnalyzer:
    """Pickle-compatible analyzer matching legacy/optunaModelTrainer.py."""

    def __init__(self, params):
        self.params = params
        self.punct_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

    @staticmethod
    def normalize_text(text):
        normalized = unicodedata.normalize("NFD", text)
        return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    def __call__(self, text):
        if not isinstance(text, str):
            return []

        text = self.normalize_text(text)
        text = text.lower().translate(self.punct_trans)
        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words:
            return []

        p = self.params
        tokens = []

        if p["W_CHAR"] > 0:
            for word in words:
                padded = f"<{word}>"
                padded_len = len(padded)
                for i in range(padded_len):
                    for n in range(p["CHAR_MIN"], p["CHAR_MAX"] + 1):
                        if i + n <= padded_len:
                            ngram = padded[i:i + n]
                            tokens.extend([f"C_{ngram}"] * p["W_CHAR"])

        if p["W_WORD"] > 0:
            tokens.extend([f"W_{w}" for w in words] * p["W_WORD"])

        if p["W_BI"] > 0 and len(words) > 1:
            tokens.extend([f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p["W_BI"])

        if p["W_TRI"] > 0 and len(words) > 2:
            tokens.extend([f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p["W_TRI"])

        if p["W_POS"] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p["W_POS"])

        return tokens


def load_suite(path: Path) -> list[SentenceCase]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence suite not found: {path}")

    rows: list[SentenceCase] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3:
            raise ValueError(f"Invalid suite line {lineno}: expected 'expected|variant|sentence'")

        expected, variant, sentence = parts
        rows.append(SentenceCase(expected=expected, variant=variant, sentence=sentence))

    if not rows:
        raise ValueError(f"Sentence suite is empty: {path}")

    return rows


def build_quantized_classifier(clf):
    qclf = copy.deepcopy(clf)
    qclf.coefs_ = []
    qclf.intercepts_ = []

    weight_error_report = {}
    bias_error_report = {}
    weight_scales = {}
    bias_scales = {}

    for idx, weights in enumerate(clf.coefs_, start=1):
        weights_q, scale = quantize_symmetric(weights)
        weights_deq = weights_q.astype(np.float64) * scale
        qclf.coefs_.append(weights_deq)
        weight_scales[f"W{idx}"] = scale
        weight_error_report[f"W{idx}"] = compute_quantization_error(weights, weights_q, scale)

    for idx, bias in enumerate(clf.intercepts_, start=1):
        bias_q, scale = quantize_symmetric(bias)
        bias_deq = bias_q.astype(np.float64) * scale
        qclf.intercepts_.append(bias_deq)
        bias_scales[f"b{idx}"] = scale
        bias_error_report[f"b{idx}"] = compute_quantization_error(bias, bias_q, scale)

    return qclf, {
        "weight_scales": weight_scales,
        "bias_scales": bias_scales,
        "weight_error": weight_error_report,
        "bias_error": bias_error_report,
    }


def top2(classes: np.ndarray, class_name_map: dict[int, str], probs: np.ndarray) -> list[dict]:
    order = np.argsort(probs)[::-1][:2]
    return [
        {
            "label": class_name_map[int(classes[idx])],
            "probability": float(probs[idx]),
        }
        for idx in order
    ]


def parameter_count(clf) -> int:
    total = 0
    for arr in clf.coefs_:
        total += int(arr.size)
    for arr in clf.intercepts_:
        total += int(arr.size)
    return total


def memory_stats(clf) -> dict:
    params = parameter_count(clf)
    float32_bytes = params * 4
    int8_bytes = params + (len(clf.coefs_) + len(clf.intercepts_)) * 4
    reduction = 1.0 - (int8_bytes / float32_bytes)
    return {
        "parameter_count": params,
        "float32_bytes": float32_bytes,
        "float32_mib": float32_bytes / (1024 * 1024),
        "int8_bytes": int8_bytes,
        "int8_mib": int8_bytes / (1024 * 1024),
        "reduction_fraction": reduction,
        "reduction_percent": reduction * 100.0,
    }


def summarize(results: list[dict], model_key: str) -> dict:
    total = len(results)
    clean = [r for r in results if r["variant"] == "clean"]
    typo = [r for r in results if r["variant"] == "typo"]

    def acc(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r[model_key]["predicted"] == r["expected"]) / len(rows)

    def avg_conf(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return float(np.mean([r[model_key]["confidence"] for r in rows]))

    return {
        "accuracy": acc(results),
        "clean_accuracy": acc(clean),
        "typo_accuracy": acc(typo),
        "avg_confidence": avg_conf(results),
        "clean_avg_confidence": avg_conf(clean),
        "typo_avg_confidence": avg_conf(typo),
        "count": total,
        "clean_count": len(clean),
        "typo_count": len(typo),
    }


def build_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Sentence Suite Quantization Report")
    lines.append("")
    lines.append("This is a handcrafted sanity suite, not a formal benchmark. Each category is represented by one clean sentence and one typo-heavy variant.")
    lines.append("")
    lines.append(f"Suite file: `{payload['suite_path']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases: `{payload['summary']['float']['count']}` (`{payload['summary']['float']['clean_count']}` clean, `{payload['summary']['float']['typo_count']}` typo)")
    lines.append(f"- Float expected-label accuracy: `{payload['summary']['float']['accuracy']:.2%}`")
    lines.append(f"- Quantized expected-label accuracy: `{payload['summary']['quantized']['accuracy']:.2%}`")
    lines.append(f"- Float/quantized prediction agreement: `{payload['summary']['agreement_rate']:.2%}`")
    lines.append(f"- Float avg confidence: `{payload['summary']['float']['avg_confidence']:.2%}`")
    lines.append(f"- Quantized avg confidence: `{payload['summary']['quantized']['avg_confidence']:.2%}`")
    lines.append(f"- Mean absolute confidence delta: `{payload['summary']['mean_abs_conf_delta']:.4f}`")
    lines.append("")
    lines.append("## Memory")
    lines.append("")
    mem = payload["memory"]
    lines.append(f"- Parameters: `{mem['parameter_count']}`")
    lines.append(f"- Float32 params: `{mem['float32_mib']:.3f} MiB`")
    lines.append(f"- INT8 params + scales: `{mem['int8_mib']:.3f} MiB`")
    lines.append(f"- Reduction: `{mem['reduction_percent']:.2f}%`")
    lines.append("")
    lines.append("## Quantization Error")
    lines.append("")
    for layer, stats in payload["quantization"]["weight_error"].items():
        lines.append(
            f"- `{layer}`: mean abs error `{stats['mean_error']:.6f}`, rmse `{stats['rmse']:.6f}`, relative `{stats['relative_error']:.2%}`"
        )
    lines.append("")
    lines.append("## Sentence Results")
    lines.append("")
    lines.append("| Expected | Variant | Sentence | Float | Quantized | Same? |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["results"]:
        float_cell = f"{row['float']['predicted']} ({row['float']['confidence']:.1%})"
        quant_cell = f"{row['quantized']['predicted']} ({row['quantized']['confidence']:.1%})"
        same = "yes" if row["float"]["predicted"] == row["quantized"]["predicted"] else "no"
        sentence = row["sentence"].replace("|", "\\|")
        lines.append(f"| {row['expected']} | {row['variant']} | {sentence} | {float_cell} | {quant_cell} | {same} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    metrics, model_path, metrics_path, _ = load_latest_artifacts()
    suite = load_suite(SUITE_PATH)

    pipeline = joblib.load(model_path)
    pipeline.named_steps["vectorizer"].analyzer = CustomAnalyzer(metrics["best_params"])
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    categories = metrics["categories"]

    qclf, quant_report = build_quantized_classifier(classifier)

    texts = [case.sentence for case in suite]
    X = vectorizer.transform(texts)

    float_probs = classifier.predict_proba(X)
    quant_probs = qclf.predict_proba(X)
    classes = classifier.classes_
    class_name_map = {int(class_id): categories[int(class_id)] for class_id in classes}
    class_names = [class_name_map[int(x)] for x in classes]

    results: list[dict] = []
    for case, f_probs, q_probs in zip(suite, float_probs, quant_probs):
        float_idx = int(np.argmax(f_probs))
        quant_idx = int(np.argmax(q_probs))
        results.append(
            {
                "expected": case.expected,
                "variant": case.variant,
                "sentence": case.sentence,
                "float": {
                    "predicted": class_name_map[int(classes[float_idx])],
                    "confidence": float(f_probs[float_idx]),
                    "top2": top2(classes, class_name_map, f_probs),
                },
                "quantized": {
                    "predicted": class_name_map[int(classes[quant_idx])],
                    "confidence": float(q_probs[quant_idx]),
                    "top2": top2(classes, class_name_map, q_probs),
                },
            }
        )

    float_summary = summarize(results, "float")
    quant_summary = summarize(results, "quantized")
    agreement_rate = sum(
        1 for row in results if row["float"]["predicted"] == row["quantized"]["predicted"]
    ) / len(results)
    mean_abs_conf_delta = float(np.mean([
        abs(row["float"]["confidence"] - row["quantized"]["confidence"]) for row in results
    ]))

    payload = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "suite_path": str(SUITE_PATH),
        "classes": class_names,
        "memory": memory_stats(classifier),
        "quantization": quant_report,
        "summary": {
            "float": float_summary,
            "quantized": quant_summary,
            "agreement_rate": agreement_rate,
            "mean_abs_conf_delta": mean_abs_conf_delta,
        },
        "results": results,
    }

    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_OUT.write_text(build_report(payload), encoding="utf-8")

    print(f"Saved JSON report: {JSON_OUT}")
    print(f"Saved Markdown report: {MD_OUT}")
    print(f"Float accuracy: {float_summary['accuracy']:.2%}")
    print(f"Quantized accuracy: {quant_summary['accuracy']:.2%}")
    print(f"Agreement: {agreement_rate:.2%}")
    print(f"Mean |Δ confidence|: {mean_abs_conf_delta:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
