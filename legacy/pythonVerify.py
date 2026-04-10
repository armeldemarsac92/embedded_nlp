import joblib
import numpy as np
import glob
import os
import string
import re
import sys
import unicodedata
from pathlib import Path

# ==========================================
# 🚨 CRITICAL: DEPENDENCIES
# Must match optunaModelTrainer.py EXACTLY
# ==========================================

STOP_WORDS = set()
_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from legacy_artifact_loader import decode_topic, load_latest_resources as load_latest_artifacts

_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "legacy"

class CustomAnalyzer:
    """Pickle-compatible analyzer matching legacy/optunaModelTrainer.py."""

    def __init__(self, params):
        self.params = params
        self.punct_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    @staticmethod
    def normalize_text(text):
        normalized = unicodedata.normalize('NFD', text)
        return "".join(c for c in normalized if unicodedata.category(c) != 'Mn')

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

        if p['W_CHAR'] > 0:
            for word in words:
                padded = f"<{word}>"
                padded_len = len(padded)
                for i in range(padded_len):
                    for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                        if i + n <= padded_len:
                            tokens.extend([f"C_{padded[i:i + n]}"] * p['W_CHAR'])

        if p['W_WORD'] > 0:
            tokens.extend([f"W_{w}" for w in words] * p['W_WORD'])

        if p['W_BI'] > 0 and len(words) > 1:
            tokens.extend([f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p['W_BI'])

        if p['W_TRI'] > 0 and len(words) > 2:
            tokens.extend([f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p['W_TRI'])

        if p['W_POS'] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p['W_POS'])

        return tokens

# ==========================================
# END DEPENDENCIES
# ==========================================

def load_latest_model():
    try:
        config, latest, _, artifact_dir = load_latest_artifacts()
    except FileNotFoundError:
        print("❌ No model files found!")
        return None
    print(f"📂 Loading: {latest}")
    print(f"📦 Artifact dir: {artifact_dir}")
    pipeline = joblib.load(latest)
    params = config.get("best_params", {})
    if params:
        pipeline.named_steps['vectorizer'].analyzer = CustomAnalyzer(params)
    return pipeline, latest, config.get("categories", [])

def find_latest_header():
    header_files = sorted(glob.glob(str(_ARTIFACT_DIR / "model_teensy_*.h")))
    if header_files:
        return header_files[-1]
    fallback = _ARTIFACT_DIR / "ModelWeights.h"
    if fallback.exists():
        return str(fallback)
    return None

def get_cpp_topics():
    """Extract topic list from exported header to map IDs back to names."""
    header_path = find_latest_header()
    if not header_path:
        return None

    topics = []
    parsing_topics = False

    with open(header_path, "r") as f:
        for line in f:
            line = line.strip()
            if "const char* const TOPICS[]" in line or "const char* CATEGORIES[]" in line:
                parsing_topics = True
                continue
            if parsing_topics and "};" in line:
                break
            if parsing_topics and '"' in line:
                # Extract content between quotes
                match = re.search(r'"([^"]*)"', line)
                if match:
                    topics.append(match.group(1))
    return topics

def verify_with_test_sentence(pipeline, topic_map=None):
    print("\n" + "=" * 60)
    print("TEST SENTENCE VERIFICATION")
    print("=" * 60)

    test = "La direction nous prend pour des jambons c'est abusé"

    probs = pipeline.predict_proba([test])[0]
    pred_idx = np.argmax(probs)
    classes = pipeline.named_steps['classifier'].classes_

    # Resolve class name
    pred_class_raw = classes[pred_idx]
    pred_class_name = decode_topic(pred_class_raw, topic_map or [])

    # If we have the map from C++, use it
    if topic_map and isinstance(pred_class_raw, (int, np.integer)) and pred_class_raw < len(topic_map):
        pred_class_name = f"{pred_class_raw} ({topic_map[pred_class_raw]})"

    print(f"\n📝 Test: \"{test}\"")
    print(f"\n🎯 Prediction: {pred_class_name} with {probs[pred_idx]:.2%} confidence")

    print(f"\n📊 All probabilities:")
    sorted_indices = np.argsort(probs)[::-1] # Sort descending

    for i in sorted_indices:
        cls_raw = classes[i]
        prob = probs[i]

        cls_str = decode_topic(cls_raw, topic_map or [])
        if topic_map and isinstance(cls_raw, (int, np.integer)) and cls_raw < len(topic_map):
            cls_str = f"{cls_raw} ({topic_map[cls_raw]})"

        bar = "█" * int(prob * 40)
        # Fixed formatting: Use str() and remove 's' type specifier
        print(f"   {cls_str:<20} {prob:6.2%} {bar}")

    return pred_class_raw, probs[pred_idx]

def check_exported_header():
    print("\n" + "=" * 60)
    print("EXPORTED C++ HEADER CHECK")
    print("=" * 60)

    header_path = find_latest_header()
    if not header_path:
        print("❌ Exported header not found!")
        return None

    params = {}
    with open(header_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#define") and not line.startswith("#define MODEL_WEIGHTS_H"):
                parts = line.split()
                if len(parts) >= 3:
                    params[parts[1]] = parts[2]

    if params:
        print("\n🔧 C++ Model Parameters:")
        for key, value in params.items():
            print(f"   {key:20s} = {value}")

    return params

def main():
    print("🔬 MODEL VERIFICATION TOOL")
    print("=" * 60)

    model_data = load_latest_model()
    if not model_data: return
    pipeline, _, categories = model_data

    # Get C++ topic map for readable output
    topic_map = get_cpp_topics()
    if topic_map:
        print(f"ℹ️  Loaded {len(topic_map)} topics from exported header")
    else:
        topic_map = categories

    # Run verification
    verify_with_test_sentence(pipeline, topic_map)

    # Check params
    cpp_params = check_exported_header()

    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
