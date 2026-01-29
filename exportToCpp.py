import pickle
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PICKLE = 'topic_detection_model.pkl'
OUTPUT_HEADER = 'ModelWeights.h'


def export_array_to_cpp(f, name, array, length_var_name=None):
    """
    Écrit un tableau numpy dans le fichier C++ sous forme de tableau const float.
    """
    # Aplatir le tableau (pour les matrices 2D)
    flat_array = array.flatten()
    size = len(flat_array)

    print(f"   -> Export de {name} ({size} éléments)...")

    # Écriture de la déclaration
    f.write(f"// Dimensions: {array.shape}\n")
    if length_var_name:
        f.write(f"#define {length_var_name} {size}\n")

    f.write(f"const float {name}[{size}] = {{\n")

    # Écriture des données
    for i, val in enumerate(flat_array):
        # Format float avec précision
        f.write(f"{val:.8f}f")
        if i < size - 1:
            f.write(", ")

        # Retour à la ligne pour lisibilité (tous les 10 ou 100 éléments)
        if (i + 1) % 16 == 0:
            f.write("\n    ")

    f.write("\n};\n\n")


def main():
    print(f"📂 Chargement du modèle depuis '{INPUT_PICKLE}'...")

    if not os.path.exists(INPUT_PICKLE):
        print("❌ Fichier .pkl introuvable ! Lance d'abord le script d'entraînement.")
        return

    try:
        with open(INPUT_PICKLE, 'rb') as f:
            data = pickle.load(f)
            pipeline = data['pipeline']
            le = data['label_encoder']

        # Récupération du classifieur (MLP) dans le pipeline
        mlp = pipeline.named_steps['classifier']

        # Vérification des couches
        # coefs_ est une liste de matrices de poids [Input->H1, H1->H2, H2->Output]
        # intercepts_ est une liste de vecteurs de biais [B1, B2, B3]

        n_layers = len(mlp.coefs_)
        print(f"✅ Modèle chargé. Architecture détectée : {n_layers} jeux de poids (3 couches si Input->H1->H2->Out).")

        if n_layers != 3:
            print("⚠️ ATTENTION : Ce script est conçu pour 2 couches cachées (3 matrices de poids).")
            print(f"   Ton modèle a {n_layers} matrices. Le C++ devra être adapté si ce n'est pas ce que tu voulais.")

        # Récupération des dimensions
        input_size = mlp.coefs_[0].shape[0]  # 4096
        hidden1_size = mlp.coefs_[0].shape[1]  # 128
        hidden2_size = mlp.coefs_[1].shape[1]  # 64
        output_size = mlp.coefs_[2].shape[1]  # N classes

        print(f"   Input Size:   {input_size}")
        print(f"   Hidden 1:     {hidden1_size}")
        print(f"   Hidden 2:     {hidden2_size}")
        print(f"   Output Size:  {output_size}")
        print(f"   Classes:      {le.classes_}")

        # --- ÉCRITURE DU FICHIER .H ---
        print(f"\n💾 Génération de '{OUTPUT_HEADER}'...")

        with open(OUTPUT_HEADER, 'w') as f:
            f.write("#ifndef MODEL_WEIGHTS_H\n")
            f.write("#define MODEL_WEIGHTS_H\n\n")
            f.write("#include <Arduino.h>\n\n")

            f.write("// =========================================\n")
            f.write("// POIDS DU RÉSEAU DE NEURONES (AUTO-GÉNÉRÉ)\n")
            f.write("// =========================================\n\n")

            # Constantes de taille
            f.write(f"#define INPUT_SIZE {input_size}\n")
            f.write(f"#define HIDDEN1_SIZE {hidden1_size}\n")
            f.write(f"#define HIDDEN2_SIZE {hidden2_size}\n")
            f.write(f"#define OUTPUT_SIZE {output_size}\n\n")

            # --- LAYER 1 (Input -> Hidden1) ---
            export_array_to_cpp(f, "W1", mlp.coefs_[0])
            export_array_to_cpp(f, "B1", mlp.intercepts_[0])

            # --- LAYER 2 (Hidden1 -> Hidden2) ---
            export_array_to_cpp(f, "W2", mlp.coefs_[1])
            export_array_to_cpp(f, "B2", mlp.intercepts_[1])

            # --- LAYER 3 (Hidden2 -> Output) ---
            export_array_to_cpp(f, "W3", mlp.coefs_[2])
            export_array_to_cpp(f, "B3", mlp.intercepts_[2])

            # --- LABELS (TOPICS) ---
            print("   -> Export des labels...")
            f.write(f"const char* TOPICS[{output_size}] = {{\n")
            for i, label in enumerate(le.classes_):
                f.write(f'    "{label}"')
                if i < output_size - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

            f.write("#endif // MODEL_WEIGHTS_H\n")

        print("✅ Terminé avec succès ! Copie 'ModelWeights.h' dans ton dossier src Teensy.")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()