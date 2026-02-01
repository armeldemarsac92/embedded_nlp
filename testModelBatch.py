import joblib  # <--- UTILISER JOBLIB, PAS PICKLE
import numpy as np
import sys
import os
from collections import Counter
import string

# Configuration
MODEL_FILE = 'topic_detection_model.pkl'  # Assure-toi que c'est le bon nom (celui défini dans ton script d'entrainement)

# Codes couleurs pour le terminal
COLORS = {
    'CYBER': '\033[91m',  # Rouge
    'INFRA': '\033[93m',  # Jaune
    'TECH': '\033[96m',  # Cyan
    'LOVE': '\033[95m',  # Magenta
    'MISC': '\033[90m',  # Gris
    'RESET': '\033[0m',
    'BOLD': '\033[1m'
}


# --- FONCTION INDISPENSABLE POUR QUE JOBLIB RECONSTITUE LE VECTORIZER ---
def optimized_multi_word_analyzer(text):
    if not isinstance(text, str):
        return []
    tokens = []
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()[:15]
    if len(words) == 0: return []
    for word in words:
        if len(word) < 1: continue
        padded = f"<{word}>"
        for n in range(2, min(5, len(padded) + 1)):
            for i in range(len(padded) - n + 1):
                tokens.append(padded[i:i + n])
    for i in range(len(words) - 1):
        tokens.append(f"W2:{words[i]}_{words[i + 1]}")
    for i in range(len(words) - 2):
        tokens.append(f"W3:{words[i]}_{words[i + 1]}_{words[i + 2]}")
    if len(words) >= 2:
        tokens.append(f"START:{words[0]}_{words[1]}")
        tokens.append(f"END:{words[-2]}_{words[-1]}")
    tokens.append(f"FIRST:{words[0]}")
    tokens.append(f"LAST:{words[-1]}")
    return tokens


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : '{MODEL_FILE}' introuvable.")
        sys.exit(1)

    print(f"📂 Chargement du modèle via Joblib...")
    # On charge le dictionnaire complet
    data = joblib.load(MODEL_FILE)

    # On vérifie si c'est l'ancien format (Pipeline seul) ou le nouveau (Dict)
    if isinstance(data, dict) and 'classes' in data:
        return data['pipeline'], data['classes']
    else:
        print("⚠️ ATTENTION : Le fichier .pkl ne contient pas les classes.")
        print("   Veuillez mettre à jour le script d'entrainement (Étape 1 de la réponse AI).")
        sys.exit(1)


def get_color(topic):
    for key, code in COLORS.items():
        if key in topic: return code
    return '\033[92m'


def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"❌ Erreur : Le fichier '{filename}' n'existe pas.")
        return

    print(f"📂 Lecture des phrases de test : '{filename}'...")
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n', ' ')

    raw_phrases = content.split(',')
    phrases = [p.strip() for p in raw_phrases if p.strip()]

    if not phrases:
        print("⚠️ Fichier vide.")
        return

    print(f"✅ {len(phrases)} phrases à analyser.\n")

    # Chargement
    pipeline, classes = load_model()

    # Prédiction
    all_probs = pipeline.predict_proba(phrases)
    stats = []

    print(f"{'TOPIC':<15} | {'CONF.':<8} | PHRASE")
    print("-" * 80)

    for phrase, probs in zip(phrases, all_probs):
        best_idx = np.argmax(probs)
        best_conf = probs[best_idx]

        # ICI : On utilise la liste 'classes' pour retrouver le nom
        topic = classes[best_idx]
        stats.append(topic)

        color = get_color(topic)
        bar_len = int(best_conf * 10)
        bar_visual = "█" * bar_len + "░" * (10 - bar_len)
        display_phrase = (phrase[:75] + '..') if len(phrase) > 75 else phrase

        print(f"{color}{topic:<15}{COLORS['RESET']} | {best_conf:.0%} {bar_visual} | {display_phrase}")

    print("\n" + "=" * 30)
    print("📊 RÉSUMÉ STATISTIQUE")
    print("=" * 30)
    for topic, count in Counter(stats).most_common():
        print(f"{topic:<15} : {count:3d} ({(count / len(stats)) * 100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testModelBatch.py <fichier_phrases.txt>")
    else:
        analyze_file(sys.argv[1])