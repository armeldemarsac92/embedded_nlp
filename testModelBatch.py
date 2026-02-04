import joblib
import numpy as np
import sys
import os
import json
import glob
from collections import Counter
import string

# Configuration
MODEL_FILE = 'topic_detection_model.pkl'

COLORS = {
    'CYBER': '\033[91m', 'INFRA': '\033[93m', 'TECH': '\033[96m',
    'LOVE': '\033[95m', 'MISC': '\033[90m', 'RESET': '\033[0m', 'BOLD': '\033[1m'
}

STOP_WORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en'}


# ==========================================
# --- ⚙️ SMART CONFIG LOADER (LATEST) ---
# ==========================================
def load_latest_config():
    """Finds the most recent best_results_*.json file."""
    search_pattern = "best_results_*.json"
    files = glob.glob(search_pattern)

    if not files:
        print(f"❌ Erreur : Aucun fichier de config ({search_pattern}) trouvé. Lancez le tuner.")
        sys.exit(1)

    # Sort files by name (since timestamp is in the name, latest is last)
    latest_file = sorted(files)[-1]
    print(f"📂 Chargement de la config : \033[1m{latest_file}\033[0m")

    with open(latest_file, 'r') as f:
        data = json.load(f)
        # Adapt to your new nested structure
        return data["best_params"], data.get("test_metrics", {}), latest_file


# Load global configuration
WINNING_PARAMS, METRICS, FILE_NAME = load_latest_config()


def optimized_multi_word_analyzer(text):
    if not isinstance(text, str): return []
    tokens = []
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = [w for w in text.split()[:25] if w not in STOP_WORDS]
    if not words: return []

    p = WINNING_PARAMS
    # Layer 1: Chars
    if p.get('W_CHAR', 0) > 0:
        for word in words:
            if len(word) < p['CHAR_MIN']: continue
            padded = f"<{word}>"
            for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                for i in range(len(padded) - n + 1):
                    tokens.extend([f"C:{padded[i:i + n]}"] * p['W_CHAR'])
    # Layer 2: Words/N-grams
    for i in range(len(words)):
        tokens.extend([f"W:{words[i]}"] * p['W_WORD'])
        if p.get('W_BI', 0) > 0 and i < len(words) - 1:
            tokens.extend([f"B:{words[i]}_{words[i + 1]}"] * p['W_BI'])
        if p.get('W_TRI', 0) > 0 and i < len(words) - 2:
            tokens.extend([f"T:{words[i]}_{words[i + 1]}_{words[i + 2]}"] * p['W_TRI'])
    # Layer 3: Position
    tokens.extend([f"S:{words[0]}", f"E:{words[-1]}"] * p['W_POS'])
    return tokens


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : '{MODEL_FILE}' introuvable.")
        sys.exit(1)

    print(f"📂 Chargement du modèle via Joblib...")
    data = joblib.load(MODEL_FILE)

    # We re-inject the analyzer to ensure the pipeline uses the LATEST JSON weights
    if isinstance(data, dict) and 'pipeline' in data:
        pipeline = data['pipeline']
        pipeline.named_steps['vectorizer'].analyzer = optimized_multi_word_analyzer
        return pipeline, data['classes']
    return data, data.classes_


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

    # Architecture display
    p = WINNING_PARAMS
    print(f"🚀 Model: NN ({p['hidden_1']},{p['hidden_2']}) | Alpha: {p['alpha']}")
    if METRICS:
        print(
            f"📊 Tuner Performance: Mean Recall {METRICS.get('mean_recall', 0):.2%} | Min Recall {METRICS.get('min_recall', 0):.2%}")
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
    print(f"{COLORS['BOLD']}📊 RÉSUMÉ STATISTIQUE (Sync: {FILE_NAME}){COLORS['RESET']}")
    print("=" * 60)
    for topic, count in Counter(stats).most_common():
        perc = (count / total) * 100
        print(f"{topic:<15} : {count:3d} ({perc:>5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testModelBatch.py <fichier_phrases.txt>")
    else:
        analyze_file(sys.argv[1])