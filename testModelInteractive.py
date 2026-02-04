import joblib
import numpy as np
import os
import string
import json
import sys
import glob

# Configuration Files
MODEL_FILE = 'topic_detection_model.pkl'


# ==========================================
# --- ⚙️ SMART CONFIG LOADER (LATEST) ---
# ==========================================
def load_latest_config():
    search_pattern = "best_results_*.json"
    files = glob.glob(search_pattern)

    if not files:
        print("❌ Erreur : Fichier de config introuvable. Lancez le tuner.")
        sys.exit(1)

    latest_file = sorted(files)[-1]
    with open(latest_file, 'r') as f:
        data = json.load(f)
        return data["best_params"], data.get("test_metrics", {}), latest_file


# Define this in the global scope so the unpickler can find it
def optimized_multi_word_analyzer(text):
    if not isinstance(text, str): return []

    # Local reload to ensure global sync
    search_pattern = "best_results_*.json"
    files = glob.glob(search_pattern)
    latest_file = sorted(files)[-1] if files else print("❌ Erreur : Aucun fichier de config trouvé. Lancez le tuner.")

    with open(latest_file, 'r') as f:
        conf = json.load(f)
        p = conf["best_params"] if "best_params" in conf else conf["params"]

    STOP_WORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en'}
    tokens = []
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = [w for w in text.split()[:25] if w not in STOP_WORDS]

    if not words: return []

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


def load_resources():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : '{MODEL_FILE}' introuvable.")
        sys.exit(1)

    params, metrics, file_name = load_latest_config()

    print(f"📂 Chargement du modèle synchronisé avec : {file_name}")
    data = joblib.load(MODEL_FILE)

    pipeline = data['pipeline'] if isinstance(data, dict) else data
    classes = data['classes'] if isinstance(data, dict) else data.classes_

    # Re-bind analyzer logic
    pipeline.named_steps['vectorizer'].analyzer = optimized_multi_word_analyzer

    return pipeline, classes, params, metrics


def main():
    pipeline, classes, PARAMS, METRICS = load_resources()

    print("\n" + "=" * 50)
    print("✅ SYNC SUCCESSFUL")
    print(f"🧠 Arch: ({PARAMS['hidden_1']},{PARAMS['hidden_2']}) | Alpha: {PARAMS['alpha']}")
    if METRICS:
        print(f"📊 Accuracy attendue: {METRICS.get('mean_recall', 0):.1%}")
        print(f"⚠️  Point faible: {METRICS.get('min_recall', 0):.1%}")
    print("=" * 50)
    print("📝 Tape une phrase (ou 'q' pour quitter).")

    while True:
        try:
            user_input = input("\n💬 Phrase > ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input:
                continue

            probs = pipeline.predict_proba([user_input])[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            topic = classes[idx]

            # Color coding based on confidence
            color = "\033[92m" if conf > 0.85 else "\033[93m" if conf > 0.6 else "\033[91m"

            print(f"🔍 Topic    : {color}\033[1m{topic}\033[0m")
            print(f"📊 Confiance : {conf:.2%}")

            # Show "near-miss" if another topic is close
            second_idx = np.argsort(probs)[-2]
            if probs[second_idx] > 0.20:
                print(f"🤔 Alternative: {classes[second_idx]} ({probs[second_idx]:.1%})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    main()