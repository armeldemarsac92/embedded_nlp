import pandas as pd
import numpy as np
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- CONFIGURATION ---
INPUT_SIZE = 8192  # Increased from 4096 for better feature separation
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
DATASET_FILE = "DataSetTeensyv6.csv"
OUTPUT_HEADER_FILE = "ModelWeights.h"
OUTPUT_PKL_FILE = "topic_detection_model.pkl"


# --- OPTIMIZED MULTI-WORD ANALYZER ---
def optimized_multi_word_analyzer(text):
    """
    Enhanced analyzer combining:
    - Character n-grams (2-4 chars) for typo tolerance
    - Word bigrams for phrasal context
    - Word trigrams for stronger semantic context
    - Position markers for sentence structure
    """
    if not isinstance(text, str):
        return []

    tokens = []
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()[:15]  # Limit to 15 words for Teensy memory constraints

    if len(words) == 0:
        return []

    # Layer 1: Character n-grams (handle typos, unknown words)
    for word in words:
        if len(word) < 1:
            continue
        padded = f"<{word}>"

        # Variable length n-grams (2-4 characters)
        for n in range(2, min(5, len(padded) + 1)):
            for i in range(len(padded) - n + 1):
                tokens.append(padded[i:i + n])

    # Layer 2: Word bigrams (phrasal context)
    for i in range(len(words) - 1):
        bigram = f"W2:{words[i]}_{words[i + 1]}"
        tokens.append(bigram)

    # Layer 3: Word trigrams (stronger semantic context)
    for i in range(len(words) - 2):
        trigram = f"W3:{words[i]}_{words[i + 1]}_{words[i + 2]}"
        tokens.append(trigram)

    # Layer 4: Positional markers (sentence structure awareness)
    if len(words) >= 2:
        # Beginning of sentence
        tokens.append(f"START:{words[0]}_{words[1]}")
        # End of sentence
        tokens.append(f"END:{words[-2]}_{words[-1]}")

    # Layer 5: First and last word markers (additional context)
    tokens.append(f"FIRST:{words[0]}")
    tokens.append(f"LAST:{words[-1]}")

    return tokens


# --- LOAD DATASET ---
if not os.path.exists(DATASET_FILE):
    print(f"❌ Error: {DATASET_FILE} not found.")
    print("Please ensure your dataset CSV file is in the same directory.")
    exit()

print("📂 Loading dataset...")
df = pd.read_csv(DATASET_FILE).dropna(subset=['french_sentence', 'topic'])
print(f"✓ Loaded {len(df)} samples")

X = df['french_sentence']
y_labels = df['topic']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
categories = label_encoder.classes_
OUTPUT_SIZE = len(categories)

print(f"✓ Found {OUTPUT_SIZE} categories: {', '.join(categories)}")

# --- TRAIN/TEST SPLIT FOR VALIDATION ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Dataset split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# --- BUILD MODEL PIPELINE ---
print("\n🔧 Building model pipeline...")

vectorizer = HashingVectorizer(
    n_features=INPUT_SIZE,
    alternate_sign=False,  # Avoid negative features
    norm=None,  # No normalization (we'll use raw counts)
    binary=False,  # Count occurrences (gives better results than binary)
    analyzer=optimized_multi_word_analyzer
)

mlp = MLPClassifier(
    hidden_layer_sizes=(HIDDEN1_SIZE, HIDDEN2_SIZE),
    activation='relu',
    solver='adam',
    max_iter=500,  # Increased for better convergence
    random_state=42,
    early_stopping=True,  # Stop when validation performance plateaus
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True
)

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', mlp)
])

# --- TRAINING ---
print("\n🚀 Training model...")
print("=" * 60)
pipeline.fit(X_train, y_train)
print("=" * 60)

# --- EVALUATION ---
print("\n📈 Evaluating model performance...")

# Training accuracy
train_accuracy = pipeline.score(X_train, y_train)
print(f"\n✓ Training Accuracy:   {train_accuracy:.2%}")

# Test accuracy
test_accuracy = pipeline.score(X_test, y_test)
print(f"✓ Test Accuracy:       {test_accuracy:.2%}")

# Detailed classification report
y_pred = pipeline.predict(X_test)
print("\n📋 Detailed Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=categories))

# Confusion matrix
print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Per-class accuracy
print("\n📊 Per-class Accuracy:")
for i, category in enumerate(categories):
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"   {category:20s}: {class_acc:.2%} ({class_mask.sum()} samples)")

# --- CALCULATE MODEL SIZE ---
weights = mlp.coefs_
biases = mlp.intercepts_

total_params = (
        weights[0].size +  # INPUT -> HIDDEN1
        biases[0].size +
        weights[1].size +  # HIDDEN1 -> HIDDEN2
        biases[1].size +
        weights[2].size +  # HIDDEN2 -> OUTPUT
        biases[2].size
)

flash_size_kb = total_params * 4 / 1024  # 4 bytes per float
flash_size_mb = flash_size_kb / 1024

print(f"\n💾 Model Size:")
print(f"   Total parameters: {total_params:,}")
print(f"   Flash memory:     {flash_size_kb:.1f} KB ({flash_size_mb:.2f} MB)")
print(f"   Teensy 4.1 flash: 8 MB (using {flash_size_mb / 8 * 100:.1f}%)")

# Warning if model is too large
if flash_size_mb > 6:
    print("\n⚠️  WARNING: Model is large! Consider reducing INPUT_SIZE or hidden layer sizes.")
elif flash_size_mb > 4:
    print("\n⚠️  Note: Model is moderately large, should fit but monitor flash usage.")
else:
    print("\n✓ Model size is reasonable for Teensy 4.1")

# --- EXPORT C++ HEADER FILE ---
print(f"\n💾 Generating {OUTPUT_HEADER_FILE}...")


def write_array(f, name, array):
    """Write a numpy array as a C++ const array in PROGMEM"""
    f.write(f"const float {name}[] PROGMEM = {{\n    ")
    flat = array.flatten()
    for i, val in enumerate(flat):
        f.write(f"{val:.6f}f")
        if i < len(flat) - 1:
            f.write(", ")
        if (i + 1) % 8 == 0 and i < len(flat) - 1:  # 8 values per line for readability
            f.write("\n    ")
    f.write("\n};\n\n")


with open(OUTPUT_HEADER_FILE, "w") as f:
    # Header guard and includes
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")
    f.write("#include <Arduino.h>\n\n")

    # Model dimensions
    f.write("// Model Architecture\n")
    f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
    f.write(f"#define HIDDEN1_SIZE {HIDDEN1_SIZE}\n")
    f.write(f"#define HIDDEN2_SIZE {HIDDEN2_SIZE}\n")
    f.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")

    # Topic labels
    f.write("// Topic Labels\n")
    f.write("const char* const TOPICS[] = {\n")
    for cat in categories:
        f.write(f'    "{cat}",\n')
    f.write("};\n\n")

    # Model weights and biases
    f.write("// Layer 1: Input -> Hidden1\n")
    write_array(f, "WEIGHTS_1", weights[0])
    write_array(f, "BIAS_1", biases[0])

    f.write("// Layer 2: Hidden1 -> Hidden2\n")
    write_array(f, "WEIGHTS_2", weights[1])
    write_array(f, "BIAS_2", biases[1])

    f.write("// Layer 3: Hidden2 -> Output\n")
    write_array(f, "WEIGHTS_3", weights[2])
    write_array(f, "BIAS_3", biases[2])

    f.write("#endif // MODEL_WEIGHTS_H\n")

print(f"✓ Generated {OUTPUT_HEADER_FILE}")

# --- SAVE SOME TEST EXAMPLES ---
print("\n💡 Saving test examples for C++ validation...")
test_examples_file = "test_examples.txt"
with open(test_examples_file, "w", encoding='utf-8') as f:
    f.write("# Test Examples for C++ Validation\n")
    f.write("# Format: sentence | predicted_topic | actual_topic | confidence\n\n")

    # Get prediction probabilities
    y_pred_proba = mlp.predict_proba(vectorizer.transform(X_test))

    # Sample 20 examples
    num_samples = min(20, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Convert to proper array access
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_array = y_test if isinstance(y_test, np.ndarray) else y_test.values

    for i, idx in enumerate(indices):
        sentence = X_test_array[idx] if isinstance(X_test_array, np.ndarray) else X_test.iloc[idx]
        actual = categories[y_test_array[idx]]
        predicted = categories[y_pred[idx]]
        confidence = y_pred_proba[idx].max()

        f.write(f"{sentence} | {predicted} | {actual} | {confidence:.2%}\n")

print(f"✓ Saved {test_examples_file}")

# --- FINAL SUMMARY ---
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy:     {test_accuracy:.2%}")
print(f"Model Size:        {flash_size_mb:.2f} MB")
print(f"Output Files:")
print(f"  - {OUTPUT_HEADER_FILE}")
print(f"  - {test_examples_file}")
print("\nNext steps:")
print("1. Copy ModelWeights.h to your Arduino project")
print("2. Upload the updated NlpManager.cpp")
print("3. Flash to your Teensy 4.1")
print("4. Test with examples from test_examples.txt")
print("=" * 60)

# --- REMPLACER LA SECTION "SAVE PYTHON PICKLE MODEL" PAR CECI ---
print(f"\n🥒 Saving Python model to {OUTPUT_PKL_FILE}...")
try:
    # On sauvegarde un dictionnaire contenant le modèle ET les noms des catégories
    model_data = {
        'pipeline': pipeline,
        'classes': list(label_encoder.classes_) # On sauvegarde la liste des topics (ex: ACCOUNTING, TECH...)
    }
    joblib.dump(model_data, OUTPUT_PKL_FILE)
    print(f"✓ Saved {OUTPUT_PKL_FILE} (Pipeline + Classes)")
except Exception as e:
    print(f"❌ Error saving pickle file: {e}")