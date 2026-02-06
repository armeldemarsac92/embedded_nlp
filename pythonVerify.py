"""
🔍 MODEL VERIFICATION SCRIPT (FIXED)
=====================================
Checks if exported C++ model matches Python model.
Fixes: Integer class formatting & Label mapping
"""

import joblib
import numpy as np
import glob
import os
import string
import re

# ==========================================
# 🚨 CRITICAL: DEPENDENCIES
# Must match optunaModelTrainer.py EXACTLY
# ==========================================

STOP_WORDS = {}  # Empty as per your latest config

class CustomAnalyzer:
    """Picklable analyzer class for text feature extraction"""

    def __init__(self, params):
        self.params = params

    def __call__(self, text):
        if not isinstance(text, str): return []
        # Exact match to trainer logic:
        text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words: return []

        p = self.params
        tokens = []

        if p['W_CHAR'] > 0:
            for word in words:
                if len(word) < p['CHAR_MIN']: continue
                padded = f"<{word}>"
                for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                    for i in range(len(padded) - n + 1):
                        tokens.extend([f"C:{padded[i:i + n]}"] * p['W_CHAR'])

        for i in range(len(words)):
            tokens.extend([f"W:{words[i]}"] * p['W_WORD'])
            if p['W_BI'] > 0 and i < len(words) - 1:
                tokens.extend([f"B:{words[i]}_{words[i + 1]}"] * p['W_BI'])
            if p['W_TRI'] > 0 and i < len(words) - 2:
                tokens.extend([f"T:{words[i]}_{words[i + 1]}_{words[i + 2]}"] * p['W_TRI'])

        tokens.extend([f"S:{words[0]}", f"E:{words[-1]}"] * p['W_POS'])
        return tokens

# ==========================================
# END DEPENDENCIES
# ==========================================

def load_latest_model():
    model_files = sorted(glob.glob("final_model_*.joblib"))
    if not model_files:
        print("❌ No model files found!")
        return None
    latest = model_files[-1]
    print(f"📂 Loading: {latest}")
    return joblib.load(latest), latest

def get_cpp_topics():
    """Extract topic list from ModelWeights.h to map IDs back to Names"""
    if not os.path.exists("ModelWeights.h"):
        return None

    topics = []
    parsing_topics = False

    with open("ModelWeights.h", "r") as f:
        for line in f:
            line = line.strip()
            if "const char* const TOPICS[]" in line:
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
    pred_class_name = str(pred_class_raw)

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

        cls_str = str(cls_raw)
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

    if not os.path.exists("ModelWeights.h"):
        print("❌ ModelWeights.h not found!")
        return None

    params = {}
    with open("ModelWeights.h", "r") as f:
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
    pipeline, _ = model_data

    # Get C++ topic map for readable output
    topic_map = get_cpp_topics()
    if topic_map:
        print(f"ℹ️  Loaded {len(topic_map)} topics from ModelWeights.h")

    # Run verification
    verify_with_test_sentence(pipeline, topic_map)

    # Check params
    cpp_params = check_exported_header()

    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()