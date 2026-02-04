import pandas as pd
import numpy as np
import string, os, joblib, json, glob
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline


# --- ⚙️ SMART CONFIG LOADER (LATEST) ---
def load_latest_config():
    search_pattern = "best_results_*.json"
    files = glob.glob(search_pattern)

    if not files:
        print("❌ Erreur : Aucun fichier de config trouvé. Lancez le tuner.")
        exit()

    latest_file = sorted(files)[-1]
    print(f"📂 Utilisation de la config : {latest_file}")
    with open(latest_file, 'r') as f:
        data = json.load(f)
        # We need both the params and the category list
        return data["best_params"], data["categories"], latest_file


# Global Load
PARAMS, CATS, SOURCE_FILE = load_latest_config()
STOP_WORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en'}


def optimized_multi_word_analyzer(text):
    if not isinstance(text, str): return []
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = [w for w in text.split()[:25] if w not in STOP_WORDS]
    if not words: return []

    tokens = []
    p = PARAMS
    # Layer 1: Chars
    if p.get('W_CHAR', 0) > 0:
        for word in words:
            if len(word) < p['CHAR_MIN']: continue
            padded = f"<{word}>"
            for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                for i in range(len(padded) - n + 1):
                    tokens.extend([f"C:{padded[i:i + n]}"] * p['W_CHAR'])
    # Layer 2: Words
    for i in range(len(words)):
        tokens.extend([f"W:{words[i]}"] * p['W_WORD'])
        if p.get('W_BI', 0) > 0 and i < len(words) - 1:
            tokens.extend([f"B:{words[i]}_{words[i + 1]}"] * p['W_BI'])
        if p.get('W_TRI', 0) > 0 and i < len(words) - 2:
            tokens.extend([f"T:{words[i]}_{words[i + 1]}_{words[i + 2]}"] * p['W_TRI'])
    # Layer 3: Position
    tokens.extend([f"S:{words[0]}", f"E:{words[-1]}"] * p['W_POS'])
    return tokens


def main():
    df = pd.read_csv("DataSetTeensyv8.csv").dropna(subset=['french_sentence', 'topic'])
    le = LabelEncoder()
    le.classes_ = np.array(CATS)
    y = le.transform(df['topic'])

    # --- DYNAMIC MLP CONFIG ---
    # We pull hidden_1, hidden_2 and alpha directly from the JSON
    mlp = MLPClassifier(
        hidden_layer_sizes=(PARAMS['hidden_1'], PARAMS['hidden_2']),
        alpha=PARAMS['alpha'],
        max_iter=600,
        random_state=42,
        verbose=True
    )

    pipeline = Pipeline([
        ('vectorizer',
         HashingVectorizer(n_features=8192, alternate_sign=True, norm=None, analyzer=optimized_multi_word_analyzer)),
        ('classifier', mlp)
    ])

    print(f"🚀 Training V11.4 Arch: ({PARAMS['hidden_1']},{PARAMS['hidden_2']}) | Alpha: {PARAMS['alpha']}")
    pipeline.fit(df['french_sentence'], y)

    joblib.dump({'pipeline': pipeline, 'classes': CATS}, "topic_detection_model.pkl")

    # --- C++ EXPORT ---
    print("💾 Generating ModelWeights.h for Teensy...")
    with open("ModelWeights.h", "w") as f:
        f.write(f"// AUTO-GENERATED FROM: {SOURCE_FILE}\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n#include <Arduino.h>\n\n")

        # Dimensions
        f.write(f"#define INPUT_SIZE 8192\n")
        f.write(f"#define HIDDEN1_SIZE {PARAMS['hidden_1']}\n")
        f.write(f"#define HIDDEN2_SIZE {PARAMS['hidden_2']}\n")
        f.write(f"#define OUTPUT_SIZE {len(CATS)}\n\n")

        # Hashing Weights
        f.write(f"#define CHAR_MIN {PARAMS['CHAR_MIN']}\n#define CHAR_MAX {PARAMS['CHAR_MAX']}\n")
        f.write(f"#define W_WORD {PARAMS['W_WORD']}\n#define W_BI {PARAMS['W_BI']}\n")
        f.write(f"#define W_TRI {PARAMS['W_TRI']}\n#define W_CHAR {PARAMS['W_CHAR']}\n")
        f.write(f"#define W_POS {PARAMS['W_POS']}\n\n")

        f.write("const char* const TOPICS[] = {\n")
        for c in CATS: f.write(f'    "{c}",\n')
        f.write("};\n\n")

        for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            write_array(f, f"WEIGHTS_{i + 1}", w)
            write_array(f, f"BIAS_{i + 1}", b)
        f.write("#endif\n")
    print("✅ ModelWeights.h and .pkl successfully generated and synced.")


def write_array(f, name, array):
    f.write(f"const float {name}[] PROGMEM = {{\n    ")
    flat = array.flatten()
    for i, val in enumerate(flat):
        f.write(f"{val:.6f}f")
        if i < len(flat) - 1: f.write(", ")
        if (i + 1) % 8 == 0: f.write("\n    ")
    f.write("\n};\n\n")


if __name__ == "__main__":
    main()