import joblib
import numpy as np
import sys
import os
import json
import glob
import unicodedata
from collections import Counter
import string
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from legacy_artifact_loader import decode_topic, load_latest_resources as load_latest_artifacts

# Configuration
COLORS = {
    'CYBER': '\033[91m', 'INFRA': '\033[93m', 'TECH': '\033[96m',
    'LOVE': '\033[95m', 'MISC': '\033[90m', 'ACCOUNTING': '\033[94m',
    'BANKING': '\033[92m', 'BUSINESS': '\033[96m', 'GOSSIP': '\033[95m',
    'HR_COMPLAINT': '\033[91m', 'HR_HIRING': '\033[93m',
    'RESET': '\033[0m', 'BOLD': '\033[1m', 'DIM': '\033[2m'
}

STOP_WORDS = set()


# ==========================================
# --- 🧩 PICKLE-COMPATIBLE ANALYZER (OPTIMIZED) ---
# ==========================================
class CustomAnalyzer:
    """Picklable analyzer class for text feature extraction - ISO TRAINER"""

    def __init__(self, params):
        self.params = params
        self.punct_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

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
                            ngram = padded[i:i + n]
                            tokens.extend([f"C_{ngram}"] * p['W_CHAR'])

        if p['W_WORD'] > 0:
            tokens.extend([f"W_{word}" for word in words] * p['W_WORD'])

        if p['W_BI'] > 0 and len(words) > 1:
            tokens.extend([f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p['W_BI'])

        if p['W_TRI'] > 0 and len(words) > 2:
            tokens.extend([f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p['W_TRI'])

        if p['W_POS'] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p['W_POS'])

        return tokens

    @staticmethod
    def normalize_text(text):
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


# ==========================================
# --- ⚙️ SMART RESOURCE LOADER ---
# ==========================================
def load_preferred_resources():
    """Finds the most recent .json and .joblib files with validation"""
    try:
        config_data, latest_model, latest_json, artifact_dir = load_latest_artifacts()
    except FileNotFoundError:
        print("❌ Erreur : Fichiers (.json ou .joblib) introuvables.")
        print("💡 Conseil : Lancez d'abord legacy/optunaModelTrainer.py")
        sys.exit(1)

    # Extract timestamps to verify matching
    json_timestamp = latest_json.split('_')[-1].replace('.json', '')
    model_timestamp = latest_model.split('_')[-1].replace('.joblib', '')

    print(f"\n{COLORS['BOLD']}📂 CHARGEMENT DES RESSOURCES{COLORS['RESET']}")
    print("=" * 70)
    print(f"📋 Config : {COLORS['BOLD']}{latest_json}{COLORS['RESET']}")
    print(f"🧠 Modèle : {COLORS['BOLD']}{latest_model}{COLORS['RESET']}")
    print(f"📦 Dossier : {COLORS['BOLD']}{artifact_dir}{COLORS['RESET']}")

    if json_timestamp != model_timestamp:
        print(f"{COLORS['DIM']}⚠️  Avertissement : Timestamps différents "
              f"(json: {json_timestamp} vs model: {model_timestamp}){COLORS['RESET']}")

    # Validation du contenu
    required_keys = ['best_params', 'categories']
    missing_keys = [k for k in required_keys if k not in config_data]
    if missing_keys:
        print(f"❌ Erreur : Clés manquantes dans {latest_json}: {missing_keys}")
        sys.exit(1)

    return config_data, latest_model


# Global Load
CONFIG, MODEL_PATH = load_preferred_resources()
WINNING_PARAMS = CONFIG["best_params"]
METRICS = CONFIG.get("test_metrics", {})
CATEGORIES = CONFIG["categories"]


def load_model():
    """Load model with progress indication"""
    print(f"\n🔄 Chargement du pipeline sklearn...")

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("💡 Le fichier .joblib pourrait être corrompu. Relancez le training.")
        sys.exit(1)

    # Re-inject the latest analyzer params (ensures consistency)
    pipeline.named_steps['vectorizer'].analyzer = CustomAnalyzer(WINNING_PARAMS)

    print(f"✅ Pipeline chargé avec succès")
    return pipeline, CATEGORIES


def get_color(topic):
    """Get color for topic with fallback"""
    return COLORS.get(topic.upper(), '\033[92m')  # Default green


def display_model_info():
    """Display detailed model architecture and performance"""
    p = WINNING_PARAMS

    print(f"\n{COLORS['BOLD']}🧠 ARCHITECTURE DU MODÈLE{COLORS['RESET']}")
    print("=" * 70)
    print(f"   Couches cachées : [{p['hidden_1']}, {p['hidden_2']}]")
    print(f"   Activation      : {p.get('activation', 'relu')}")
    print(f"   Alpha (L2)      : {p['alpha']:.2e}")
    print(f"   Learning rate   : {p.get('learning_rate_init', 0.001):.2e}")

    print(f"\n{COLORS['BOLD']}🔍 FEATURE ENGINEERING{COLORS['RESET']}")
    print("=" * 70)
    print(f"   Char n-grams    : {p['CHAR_MIN']}-{p['CHAR_MAX']} (weight: {p['W_CHAR']})")
    print(f"   Word unigrams   : weight {p['W_WORD']}")
    print(f"   Word bigrams    : weight {p['W_BI']}")
    print(f"   Word trigrams   : weight {p['W_TRI']}")
    print(f"   Positional      : weight {p['W_POS']}")

    if METRICS:
        print(f"\n{COLORS['BOLD']}📊 PERFORMANCES (Test Set){COLORS['RESET']}")
        print("=" * 70)
        print(f"   Mean Recall     : {METRICS.get('mean_recall', 0):.2%}")
        print(f"   Min Recall      : {METRICS.get('min_recall', 0):.2%}")
        print(f"   Balanced Acc    : {METRICS.get('balanced_accuracy', 0):.2%}")

        # Per-class recall if available
        per_class = METRICS.get('per_class_recall', {})
        if per_class:
            print(f"\n   {COLORS['DIM']}Per-class Recall:{COLORS['RESET']}")
            for cat in sorted(CATEGORIES):
                recall = per_class.get(cat, 0)
                bar_len = int(recall * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"   {cat:<15} : {recall:.1%} {bar}")


def analyze_file(filename, verbose=True):
    """Analyze text file with improved visualization"""
    if not os.path.exists(filename):
        print(f"❌ Erreur : Le fichier '{filename}' n'existe pas.")
        return

    pipeline, classes = load_model()

    if verbose:
        display_model_info()

    # Load and parse file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n', ' ')

    raw_phrases = content.split(',')
    phrases = [p.strip() for p in raw_phrases if p.strip()]

    print(f"\n{COLORS['BOLD']}🚀 ANALYSE DE {len(phrases)} PHRASES{COLORS['RESET']}")
    print("=" * 100)

    # Batch prediction (more efficient)
    all_probs = pipeline.predict_proba(phrases)
    stats = []

    # Display results
    for idx, (phrase, probs) in enumerate(zip(phrases, all_probs), 1):
        best_idx = np.argmax(probs)
        best_conf = probs[best_idx]
        topic = decode_topic(classes[best_idx], CATEGORIES)
        stats.append(topic)

        color = get_color(topic)

        # Confidence bar (0-100%)
        bar_len = int(best_conf * 20)
        bar_visual = "█" * bar_len + "░" * (20 - bar_len)

        # Truncate long phrases
        display_phrase = (phrase[:70] + '...') if len(phrase) > 70 else phrase

        # Show top 2 predictions if confidence < 80%
        extra_info = ""
        if best_conf < 0.8:
            second_idx = np.argsort(probs)[-2]
            second_conf = probs[second_idx]
            second_topic = decode_topic(classes[second_idx], CATEGORIES)
            extra_info = f" {COLORS['DIM']}(2nd: {second_topic} {second_conf:.0%}){COLORS['RESET']}"

        print(f"{idx:3d}. {color}{topic:<15}{COLORS['RESET']} │ "
              f"{best_conf:>4.0%} {bar_visual}{extra_info} │ {display_phrase}")

    # Statistics summary
    total = len(stats)
    print("\n" + "=" * 100)
    print(f"{COLORS['BOLD']}📊 RÉSUMÉ STATISTIQUE{COLORS['RESET']}")
    print("=" * 100)

    for topic, count in Counter(stats).most_common():
        perc = (count / total) * 100
        color = get_color(topic)
        bar_len = int(perc / 5)  # Scale to 20 chars max
        bar = "█" * bar_len
        print(f"{color}{topic:<15}{COLORS['RESET']} : {count:3d} ({perc:>5.1f}%) {bar}")

    # Confidence distribution
    confidences = [np.max(probs) for probs in all_probs]
    avg_conf = np.mean(confidences)
    low_conf_count = sum(1 for c in confidences if c < 0.7)

    print(f"\n{COLORS['BOLD']}🎯 CONFIANCE MOYENNE{COLORS['RESET']}")
    print("=" * 100)
    print(f"   Moyenne         : {avg_conf:.1%}")
    print(f"   < 70% confiance : {low_conf_count}/{total} ({low_conf_count / total * 100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"\n{COLORS['BOLD']}Usage:{COLORS['RESET']}")
        print(f"   python testModelBatch.py <fichier_phrases.txt> [--quiet]")
        print(f"\nOptions:")
        print(f"   --quiet : Désactive l'affichage des infos du modèle")
        sys.exit(1)

    verbose = '--quiet' not in sys.argv
    filename = sys.argv[1]

    analyze_file(filename, verbose=verbose)
