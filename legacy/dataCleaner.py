import pandas as pd
import re
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def clean_dataset(file_path, output_path=_DATA_DIR / "DataSetTeensyv9_ULTRA_CLEAN.csv"):
    # Chargement avec gestion de l'encoding au cas où
    df = pd.read_csv(file_path)
    initial_count = len(df)

    # --- 1. NETTOYAGE DES TEXTES ---
    def deep_clean(text):
        if not isinstance(text, str): return ""

        # Suppression des templates (e), (s), (es), (s)
        text = re.sub(r'\([esx]{1,2}\)', '', text)

        # Suppression des suffixes de biais identifiés
        suffixes = [
            r"selon les normes IFRS\.?",
            r"d'un point de vue comptable\.?",
            r"pour la clôture mensuelle\.?",
            r"vis-à-vis de l'équipe\.?",
            r"selon le règlement intérieur\.?",
            r"dans les livres légaux\.?",
            r"sur l'exercice N-1\.?",
            r"avant l'audit\.?",
            r"à la machine à café\.?",
            r"pour augmenter la marge\.?",
            r"pour atteindre les objectifs\.?"
        ]
        for s in suffixes:
            text = re.sub(s, '', text, flags=re.IGNORECASE)

        # Nettoyage des doubles espaces et ponctuation inutile en fin de phrase
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().strip('.').strip(',')

        return text

    df['french_sentence'] = df['french_sentence'].apply(deep_clean)

    # --- 2. GESTION DES DOUBLONS ET DU BRUIT ---

    # Suppression des doublons de texte (indépendamment de la casse)
    df['temp_lower'] = df['french_sentence'].str.lower()
    df = df.drop_duplicates(subset=['temp_lower'])

    # Suppression des phrases qui n'ont plus de sens après nettoyage ou trop courtes
    # (On remonte à 15 caractères pour éliminer les restes de phrases tronquées)
    df = df[df['french_sentence'].str.len() > 15]

    # --- 3. RÉÉQUILIBRAGE RAPIDE (Optionnel) ---
    # Si une catégorie a été trop "vidée" par le clean, on le verra ici
    print(df['topic'].value_counts())

    # --- 4. EXPORT ---
    df = df[['french_sentence', 'topic']]  # On ne garde que les colonnes utiles
    df.to_csv(output_path, index=False, encoding='utf-8')

    final_count = len(df)
    print(f"--- Rapport de Nettoyage ---")
    print(f"Lignes initiales : {initial_count}")
    print(f"Lignes supprimées : {initial_count - final_count}")
    print(f"Dataset final : {final_count} lignes.")


# Exécution
clean_dataset(_DATA_DIR / "DataSetTeensyv8.csv")
