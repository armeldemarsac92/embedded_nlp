import pickle
import numpy as np
import sys
import os

# Nom du fichier sauvegardé par le script d'entraînement
MODEL_FILE = 'topic_detection_model.pkl'


def load_model():
    """Charge le pipeline et l'encodeur depuis le fichier pickle."""
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : Le fichier '{MODEL_FILE}' est introuvable.")
        print("   -> Lance d'abord le script d'entraînement (trainingScript.py).")
        sys.exit(1)

    print(f"📂 Chargement du modèle depuis '{MODEL_FILE}'...")
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)

    return data['pipeline'], data['label_encoder']


def main():
    # 1. Chargement
    pipeline, le = load_model()
    print("✅ Modèle chargé avec succès !")
    print("-" * 50)
    print("📝 Tape une phrase pour tester (ou 'exit' pour quitter).")
    print("-" * 50)

    # 2. Boucle d'interaction
    while True:
        try:
            # Récupère l'entrée utilisateur
            user_input = input("\n💬 Phrase > ").strip()

            # Gestion de la sortie
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Au revoir !")
                break

            if not user_input:
                continue

            # 3. Prédiction
            # Note : Le pipeline attend une liste, même pour une seule phrase
            # predict_proba renvoie les probabilités pour chaque classe
            probabilities = pipeline.predict_proba([user_input])[0]

            # On récupère l'index de la probabilité la plus élevée
            best_idx = np.argmax(probabilities)
            best_conf = probabilities[best_idx]

            # On récupère le nom du topic via le LabelEncoder
            predicted_topic = le.inverse_transform([best_idx])[0]

            # 4. Affichage du résultat
            # On affiche une barre de confiance visuelle
            bar_len = int(best_conf * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(f"🔍 Topic   : \033[1m{predicted_topic}\033[0m")  # En gras
            print(f"📊 Confiance : [{bar}] {best_conf:.2%}")

            # Optionnel : Afficher le 2ème choix si le premier n'est pas sûr (< 80%)
            if best_conf < 0.80:
                # Trie les index par probabilité décroissante
                sorted_idxs = np.argsort(probabilities)[::-1]
                second_idx = sorted_idxs[1]
                second_topic = le.inverse_transform([second_idx])[0]
                second_conf = probabilities[second_idx]
                print(f"   (Hésitation avec : {second_topic} à {second_conf:.2%})")

        except KeyboardInterrupt:
            print("\n👋 Arrêt forcé.")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    main()