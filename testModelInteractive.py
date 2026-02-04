import joblib
import numpy as np
import os
import string
import json
import sys
import glob


# ==========================================
# --- 🧩 PICKLE-COMPATIBLE ANALYZER ---
# ==========================================
class CustomAnalyzer:
    """Must be present in __main__ for joblib to load the model correctly"""

    def __init__(self, params):
        self.params = params
        self.stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en'}

    def __call__(self, text):
        if not isinstance(text, str): return []
        text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = [w for w in text.split()[:25] if w not in self.stop_words]
        if not words: return []

        p = self.params
        tokens = []

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

    with open(latest_json, 'r') as f:
        config_data = json.load(f)

    return config_data, latest_model


def load_resources():
    config, model_path = load_latest_resources()
    params = config["best_params"]
    metrics = config.get("test_metrics", {})

    print(f"📂 Chargement du modèle : \033[1m{model_path}\033[0m")
    print(
        f"📂 Avec la config : \033[1m{os.path.basename(model_path).replace('final_model_', 'best_results_').replace('.joblib', '.json')}\033[0m")

    # Load the binary
    pipeline = joblib.load(model_path)

    # Force the latest analyzer parameters from the JSON
    pipeline.named_steps['vectorizer'].analyzer = CustomAnalyzer(params)

    return pipeline, config["categories"], params, metrics


# ==========================================
# --- 💬 MAIN INTERACTIVE LOOP ---
# ==========================================
def main():
    try:
        pipeline, classes, PARAMS, METRICS = load_resources()
    except Exception as e:
        print(f"❌ Erreur au chargement : {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ SYNC SUCCESSFUL")
    print(f"🧠 Arch: ({PARAMS['hidden_1']},{PARAMS['hidden_2']}) | Alpha: {PARAMS['alpha']}")
    if METRICS:
        print(f"📊 Test Recall: {METRICS.get('mean_recall', 0):.1%} (Moyenne)")
        print(f"⚠️  Faiblesse: {METRICS.get('min_recall', 0):.1%} (Pire classe)")
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

            # Confidence-based coloring
            color = "\033[92m" if conf > 0.85 else "\033[93m" if conf > 0.6 else "\033[91m"

            print(f"🔍 Topic    : {color}\033[1m{topic}\033[0m")
            print(f"📊 Confiance : {conf:.2%}")

            # Show near-misses
            second_idx = np.argsort(probs)[-2]
            if probs[second_idx] > 0.15:  # Lowered threshold to see more "battles"
                print(f"🤔 Alternative: {classes[second_idx]} ({probs[second_idx]:.1%})")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    main()