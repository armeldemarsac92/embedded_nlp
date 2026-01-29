import pandas as pd
import numpy as np
import pickle
from time import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
# On augmente la taille du vecteur car les matrices creuses (Sparse) le permettent sans saturer la RAM.
# 2**14 = 16384 caractéristiques. Moins de collisions = meilleure précision.
N_FEATURES = 2 ** 12

# --- CHARGEMENT DES DONNÉES ---
print("📂 Chargement du dataset...")
try:
    # Assure-toi que le nom de la colonne correspond bien à ton CSV (french_sentence ou french_sentence)
    df = pd.read_csv("DataSetTeensyv3.csv")

    # Nettoyage basique : on s'assure que tout est en string
    df['french_sentence'] = df['french_sentence'].astype(str)

    X = df['french_sentence']
    y_raw = df['topic']
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

# Encodage des labels (Accounting -> 0, Banking -> 1...)
le = LabelEncoder()
y = le.fit_transform(y_raw)
categories = le.classes_
print(f"✅ Catégories détectées : {len(categories)} {categories}")

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# --- DÉFINITION DU PIPELINE HYBRIDE ---
# C'est ici que la magie opère pour remplacer ta fonction get_enhanced_vector
# On combine deux "cerveaux" de vectorisation :
# 1. 'char_wb' (3-grammes) : Excellent pour les fautes de frappe (ex: "bnaque" ~ "banque")
# 2. 'word' (1-grammes & 2-grammes) : Capture le contexte (ex: "virement" + "bancaire")

vectorizer_morpho = HashingVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),  # Capture les racines des mots malgré les typos
    n_features=N_FEATURES,
    norm='l2',
    alternate_sign=False  # Garde les valeurs positives (mieux pour ReLU)
)

vectorizer_context = HashingVectorizer(
    analyzer='word',
    ngram_range=(1, 2),  # Capture les mots seuls et les paires de mots
    n_features=N_FEATURES,
    norm='l2',
    alternate_sign=False
)

# FeatureUnion combine les deux vecteurs en un seul grand vecteur
combined_features = FeatureUnion([
    ("morpho", vectorizer_morpho),
    ("context", vectorizer_context)
])

# --- DÉFINITION DU MODÈLE ---
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Architecture
    activation='relu',
    solver='adam',
    alpha=0.001,  # Regularization L2 (ajusté)
    learning_rate_init=0.001,
    batch_size=64,  # Mini-batch pour accélérer la convergence
    max_iter=200,  # Souvent suffisant avec Adam
    early_stopping=True,  # Arrête si ça ne s'améliore plus (gain de temps)
    n_iter_no_change=10,
    verbose=True,
    random_state=42
)

# Le Pipeline final : Texte brut -> Vectorisation Hybride -> MLP
pipeline = Pipeline([
    ('features', combined_features),
    ('classifier', mlp)
])

# --- ENTRAÎNEMENT ---
print(f"\n🚀 Démarrage de l'entraînement sur {len(X_train)} exemples...")
t0 = time()
pipeline.fit(X_train, y_train)
print(f"⏱️ Entraînement terminé en {time() - t0:.2f} secondes.")

# --- ÉVALUATION ---
print("\n📊 Évaluation...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 PRÉCISION FINALE : {acc:.2%}")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=categories))

# --- SAUVEGARDE ---
# On sauvegarde tout le pipeline (vectorizer + model) + l'encodeur de labels
print("💾 Sauvegarde du modèle...")
with open('topic_detection_model.pkl', 'wb') as f:
    pickle.dump({'pipeline': pipeline, 'label_encoder': le}, f)
print("✅ Modèle sauvegardé sous 'topic_detection_model.pkl'")