import joblib
import numpy as np
import os
import string
import json
import sys
import glob
import unicodedata

# ==========================================
# 🚨 CRITICAL: GLOBAL CONFIG
# Must match trainer exactly for the Analyzer
# ==========================================
STOP_WORDS = set()


# ==========================================
# 🧩 PICKLE-COMPATIBLE ANALYZER
# (Exact copy from Trainer)
# ==========================================
class CustomAnalyzer:
    """Picklable analyzer class for text feature extraction - WITH PADDING"""

    def __init__(self, params):
        self.params = params
        # Pre-compile translation table for performance
        self.punct_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    @staticmethod
    def normalize_text(text):
        """Unicode normalization + accent removal"""
        # NFD normalization + accent removal
        normalized = unicodedata.normalize('NFD', text)
        result = "".join([c for c in normalized if unicodedata.category(c) != 'Mn'])
        return result

    def __call__(self, text):
        if not isinstance(text, str):
            return []

        # Normalization and cleaning
        text = self.normalize_text(text)
        text = text.lower().translate(self.punct_trans)

        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words:
            return []

        p = self.params
        tokens = []

        # ==========================================
        # CHARACTER N-GRAMS WITH PADDING <word>
        # ==========================================
        if p['W_CHAR'] > 0:
            for word in words:
                padded = f"<{word}>"  # ✅ PADDING ACTIVATED
                padded_len = len(padded)

                # ⚠️ CRITICAL: Position FIRST, then n-gram size
                for i in range(padded_len):
                    for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                        if i + n <= padded_len:
                            ngram = padded[i:i + n]
                            tokens.extend([f"C_{ngram}"] * p['W_CHAR'])

        # ==========================================
        # WORD UNIGRAMS
        # ==========================================
        if p['W_WORD'] > 0:
            tokens.extend([f"W_{w}" for w in words] * p['W_WORD'])

        # ==========================================
        # WORD BIGRAMS
        # ==========================================
        if p['W_BI'] > 0 and len(words) > 1:
            tokens.extend([f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p['W_BI'])

        # ==========================================
        # WORD TRIGRAMS
        # ==========================================
        if p['W_TRI'] > 0 and len(words) > 2:
            tokens.extend([f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p['W_TRI'])

        # ==========================================
        # POSITIONAL FEATURES
        # ==========================================
        if p['W_POS'] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p['W_POS'])

        return tokens


# ==========================================
# --- ⚙️ SMART RESOURCE LOADER ---
# ==========================================
def load_latest_resources():
    """Finds the most recent .json and .joblib files."""
    # Look for files matching the trainer's output format
    json_files = sorted(glob.glob("best_results_*.json"))
    model_files = sorted(glob.glob("final_model_*.joblib"))

    if not json_files or not model_files:
        print("❌ Error: Files (.json or .joblib) not found. Run the trainer first.")
        sys.exit(1)

    # Get the absolute last file (highest timestamp)
    latest_json = json_files[-1]
    latest_model = model_files[-1]

    with open(latest_json, 'r') as f:
        config_data = json.load(f)

    return config_data, latest_model


def load_resources():
    config, model_path = load_latest_resources()

    # Extract params and metrics
    params = config["best_params"]
    metrics = config.get("test_metrics", {})
    categories = config["categories"]

    print(f"📂 Loading Model  : \033[1m{model_path}\033[0m")
    print(
        f"📂 Loading Config : \033[1m{os.path.basename(model_path).replace('final_model_', 'best_results_').replace('.joblib', '.json')}\033[0m")

    # Load the binary model
    # Because CustomAnalyzer is defined identically above, this will work.
    pipeline = joblib.load(model_path)

    # OPTIONAL: Explicitly force the params just in case,
    # though pickle usually handles state restoration.
    # pipeline.named_steps['vectorizer'].analyzer.params = params

    return pipeline, categories, params, metrics


# ==========================================
# --- 💬 MAIN INTERACTIVE LOOP ---
# ==========================================
def main():
    try:
        pipeline, classes, PARAMS, METRICS = load_resources()
    except Exception as e:
        print(f"❌ Load Error: {e}")
        print("💡 Hint: Ensure 'CustomAnalyzer' in this script matches the Trainer exactly.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ SYNC SUCCESSFUL")
    print(f"🧠 Architecture : Hidden Layers ({PARAMS['hidden_1']}, {PARAMS['hidden_2']})")
    print(f"🧠 Activation   : {PARAMS['activation']}")
    print(f"🧠 Alpha (L2)   : {PARAMS['alpha']:.6f}")

    if METRICS:
        print("-" * 60)
        print(f"📊 Mean Recall  : {METRICS.get('mean_recall', 0):.1%}")
        print(f"⚠️  Min Recall   : {METRICS.get('min_recall', 0):.1%} (Worst Class)")
    print("=" * 60)
    print("📝 Type a phrase to test (or 'q' to quit).")

    while True:
        try:
            user_input = input("\n💬 Phrase > ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input:
                continue

            # Predict
            probs = pipeline.predict_proba([user_input])[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            topic = classes[idx]

            # Visual Confidence
            # Green > 85%, Yellow > 60%, Red < 60%
            color = "\033[92m" if conf > 0.85 else "\033[93m" if conf > 0.6 else "\033[91m"

            print(f"🔍 Topic    : {color}\033[1m{topic}\033[0m")
            print(f"📊 Confidence : {color}{conf:.2%}\033[0m")

            # Show Runner-up if close
            sorted_indices = np.argsort(probs)
            second_idx = sorted_indices[-2]
            if probs[second_idx] > 0.10:
                print(f"🤔 Alternative: {classes[second_idx]} ({probs[second_idx]:.1%})")

            # Debug: Show activated features (Optional)
            # analyzer = pipeline.named_steps['vectorizer'].analyzer
            # print(f"⚙️  Tokens: {analyzer(user_input)}")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()