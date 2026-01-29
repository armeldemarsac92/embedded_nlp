import pickle
import numpy as np
import sys
import os
from collections import Counter

# Configuration
MODEL_FILE = 'topic_detection_model.pkl'

# Codes couleurs pour le terminal (pour faire joli)
COLORS = {
    'CYBER': '\033[91m',  # Rouge
    'INFRA': '\033[93m',  # Jaune
    'TECH': '\033[96m',  # Cyan
    'LOVE': '\033[95m',  # Magenta
    'MISC': '\033[90m',  # Gris
    'RESET': '\033[0m',
    'BOLD': '\033[1m'
}


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : '{MODEL_FILE}' introuvable.")
        sys.exit(1)
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['pipeline'], data['label_encoder']


def get_color(topic):
    """Renvoie la couleur associée au topic ou blanc par défaut."""
    for key, code in COLORS.items():
        if key in topic:  # Gère HR HIRING / HR COMPLAINT avec une seule couleur si besoin
            return code
    return '\033[92m'  # Vert pour le reste (Business, Finance, etc.)


def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"❌ Erreur : Le fichier '{filename}' n'existe pas.")
        return

    print(f"📂 Lecture du fichier '{filename}'...")

    # 1. Lecture et Nettoyage
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # On remplace les sauts de ligne par des espaces pour éviter de couper les phrases
    content = content.replace('\n', ' ')

    # On découpe par virgule
    raw_phrases = content.split(',')

    # On nettoie les espaces vides
    phrases = [p.strip() for p in raw_phrases if p.strip()]

    if not phrases:
        print("⚠️ Le fichier semble vide ou mal formaté.")
        return

    print(f"✅ {len(phrases)} phrases identifiées. Lancement de l'analyse...\n")

    # 2. Chargement & Prédiction
    pipeline, le = load_model()

    # Prédiction par lot (Batch) -> Beaucoup plus rapide que phrase par phrase
    all_probs = pipeline.predict_proba(phrases)

    # Pour les statistiques finales
    stats = []

    # 3. Affichage des résultats
    print(f"{'TOPIC':<15} | {'CONF.':<8} | PHRASE")
    print("-" * 80)

    for phrase, probs in zip(phrases, all_probs):
        best_idx = np.argmax(probs)
        best_conf = probs[best_idx]
        topic = le.inverse_transform([best_idx])[0]

        stats.append(topic)

        # Couleur et barre de confiance
        color = get_color(topic)
        bar_len = int(best_conf * 10)  # Barre sur 10 caractères
        bar_visual = "█" * bar_len + "░" * (10 - bar_len)

        # Affichage ligne par ligne
        # On tronque la phrase si elle est trop longue pour l'affichage console
        display_phrase = (phrase[:75] + '..') if len(phrase) > 75 else phrase

        print(f"{color}{topic:<15}{COLORS['RESET']} | {best_conf:.0%} {bar_visual} | {display_phrase}")

    # 4. Résumé Statistique
    print("\n" + "=" * 30)
    print("📊 RÉSUMÉ STATISTIQUE")
    print("=" * 30)
    counts = Counter(stats)
    total = len(stats)

    for topic, count in counts.most_common():
        pct = (count / total) * 100
        print(f"{topic:<15} : {count:3d} ({pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_test_model.py <votre_fichier_texte.txt>")
    else:
        analyze_file(sys.argv[1])