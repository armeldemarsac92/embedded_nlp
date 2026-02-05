import joblib
import numpy as np
import sys
import os
import json
import glob
from collections import Counter
import string

# Configuration - We no longer use a static MODEL_FILE string
COLORS = {
    'CYBER': '\033[91m', 'INFRA': '\033[93m', 'TECH': '\033[96m',
    'LOVE': '\033[95m', 'MISC': '\033[90m', 'RESET': '\033[0m', 'BOLD': '\033[1m'
}

STOP_WORDS = {}


# ==========================================
# --- 🧩 PICKLE-COMPATIBLE ANALYZER ---
# ==========================================
class CustomAnalyzer:
    """Must be present for joblib to load the model correctly"""

    def __init__(self, params):
        self.params = params

    def __call__(self, text):
        if not isinstance(text, str): return []
        text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words: return []
        p = self.params
        tokens = []
        if p.get('W_CHAR', 0) > 0:
            for word in words:
                if len(word) < p['CHAR_MIN']: continue
                padded = f"<{word}>"
                for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                    for i in range(len(padded) - n + 1):
                        tokens.extend([f"C:{padded[i:i + n]}"] * p['W_CHAR'])
        for i in range(len(words)):
            tokens.extend([f"W:{words[i]}"] * p['W_WORD'])
            if p.get('W_BI', 0) > 0 and i < len(words) - 1:
                tokens.extend([f"B:{words[i]}_{words[i + 1]}"] * p['W_BI'])
            if p.get('W_TRI', 0) > 0 and i < len(words) - 2:
                tokens.extend([f"T:{words[i]}_{words[i + 1]}_{words[i + 2]}"] * p['W_TRI'])
        tokens.extend([f"S:{words[0]}", f"E:{words[-1]}"] * p['W_POS'])
        return tokens


# ==========================================
# --- ⚙️ SMART RESOURCE LOADER ---
# ==========================================
def load_latest_resources():
    """Finds the most recent .json and .joblib files."""
    json_files = sorted(glob.glob("best_results_*.json"))
    model_files = sorted(glob.glob("final_model_*.joblib"))

    if not json_files or not model_files:
        print("❌ Erreur : Fichiers (.json ou .joblib) introuvables. Lancez le tuner.")
        sys.exit(1)

    latest_json = json_files[-1]
    latest_model = model_files[-1]

    print(f"📂 Config : \033[1m{latest_json}\033[0m")
    print(f"📂 Modèle : \033[1m{latest_model}\033[0m")

    with open(latest_json, 'r') as f:
        config_data = json.load(f)

    return config_data, latest_model


# Global Load
CONFIG, MODEL_PATH = load_latest_resources()
WINNING_PARAMS = CONFIG["best_params"]
METRICS = CONFIG.get("test_metrics", {})
CATEGORIES = CONFIG["categories"]


def load_model():
    print(f"📂 Chargement du binaire via Joblib...")
    # Joblib will look for CustomAnalyzer in the __main__ scope
    pipeline = joblib.load(MODEL_PATH)

    # Re-inject the latest analyzer params in case the .joblib was saved with old ones
    pipeline.named_steps['vectorizer'].analyzer = CustomAnalyzer(WINNING_PARAMS)

    return pipeline, CATEGORIES


def get_color(topic):
    for key, code in COLORS.items():
        if key in topic: return code
    return '\033[92m'


def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"❌ Erreur : Le fichier '{filename}' n'existe pas.")
        return

    pipeline, classes = load_model()

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n', ' ')

    raw_phrases = content.split(',')
    phrases = [p.strip() for p in raw_phrases if p.strip()]

    p = WINNING_PARAMS
    print(f"🚀 Arch: NN({p['hidden_1']},{p['hidden_2']}) | Alpha: {p['alpha']}")
    if METRICS:
        print(f"📊 Test Recall: Mean {METRICS.get('mean_recall', 0):.2%} | Min {METRICS.get('min_recall', 0):.2%}")
    print(f"🚀 Processing {len(phrases)} phrases...")
    print("-" * 100)

    all_probs = pipeline.predict_proba(phrases)
    stats = []

    for phrase, probs in zip(phrases, all_probs):
        best_idx = np.argmax(probs)
        best_conf = probs[best_idx]
        topic = classes[best_idx]
        stats.append(topic)

        color = get_color(topic)
        bar_len = int(best_conf * 10)
        bar_visual = "█" * bar_len + "░" * (10 - bar_len)

        display_phrase = (phrase[:70] + '..') if len(phrase) > 70 else phrase
        print(f"{color}{topic:<15}{COLORS['RESET']} | {best_conf:.0%} {bar_visual} | {display_phrase}")

    total = len(stats)
    print("\n" + "=" * 60)
    print(f"{COLORS['BOLD']}📊 RÉSUMÉ STATISTIQUE{COLORS['RESET']}")
    print("=" * 60)
    for topic, count in Counter(stats).most_common():
        perc = (count / total) * 100
        print(f"{topic:<15} : {count:3d} ({perc:>5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testModelBatch.py <fichier_phrases.txt>")
    else:
        analyze_file(sys.argv[1])