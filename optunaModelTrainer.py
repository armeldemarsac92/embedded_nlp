import pandas as pd
import numpy as np
import string, os, joblib, json, random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import optuna
from optuna.samplers import TPESampler
from datetime import datetime

# --- CONFIG ---
DATASET_FILE = "DataSetTeensyv9_ULTRA_CLEAN.csv"
N_TRIALS = 70
RANDOM_SEED = 42
INPUT_SIZE = 8192

# Expanded Stop Words to handle SMS/Informal noise
STOP_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en',
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se',
    'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
    'est', 'sont', 'ai', 'as', 'a', 'avez', 'ont', 'etait',
    'que', 'qui', 'qu', 'dans', 'sur', 'avec', 'pour', 'par', 'ne', 'pas', 'plus',
    're', 'ok', 'mdr', 'lol', 'ptdr', 'donc', 'chez', 'pourquoi', 'comment',
    'svp', 'stp', 'wtf', 'plz', 'c','t','nn'
}

# Search space for Optuna
SEARCH_SPACE = {
    'W_CHAR': [0, 1, 2],
    'W_WORD': [4, 6, 8, 10],
    'W_BI': [0, 5, 10],
    'W_TRI': [0, 5, 10, 15],
    'W_POS': [2, 4, 6],
    'CHAR_MIN': [2, 3],
    'CHAR_MAX': [4, 5],
    'alpha': [0.0001, 0.001, 0.005, 0.01],
    'hidden_1': [64, 128],
    'hidden_2': [32, 64]
}


class CustomAnalyzer:
    """Picklable analyzer class for text feature extraction"""

    def __init__(self, params):
        self.params = params

    def __call__(self, text):
        if not isinstance(text, str): return []
        text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words: return []

        p = self.params
        tokens = []

        if p['W_CHAR'] > 0:
            for word in words:
                if len(word) < p['CHAR_MIN']: continue
                padded = f"<{word}>"
                for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                    for i in range(len(padded) - n + 1):
                        tokens.extend([f"C:{padded[i:i + n]}"] * p['W_CHAR'])

        for i in range(len(words)):
            tokens.extend([f"W:{words[i]}"] * p['W_WORD'])
            if p['W_BI'] > 0 and i < len(words) - 1:
                tokens.extend([f"B:{words[i]}_{words[i + 1]}"] * p['W_BI'])
            if p['W_TRI'] > 0 and i < len(words) - 2:
                tokens.extend([f"T:{words[i]}_{words[i + 1]}_{words[i + 2]}"] * p['W_TRI'])

        tokens.extend([f"S:{words[0]}", f"E:{words[-1]}"] * p['W_POS'])
        return tokens


class Objective:
    def __init__(self, X_train, X_val, y_train, y_val, categories):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.categories = categories
        self.trial_log = []
        self.best_actual_fitness = -1

    def __call__(self, trial):
        params = {
            'W_CHAR': trial.suggest_categorical('W_CHAR', SEARCH_SPACE['W_CHAR']),
            'W_WORD': trial.suggest_categorical('W_WORD', SEARCH_SPACE['W_WORD']),
            'W_BI': trial.suggest_categorical('W_BI', SEARCH_SPACE['W_BI']),
            'W_TRI': trial.suggest_categorical('W_TRI', SEARCH_SPACE['W_TRI']),
            'W_POS': trial.suggest_categorical('W_POS', SEARCH_SPACE['W_POS']),
            'CHAR_MIN': trial.suggest_categorical('CHAR_MIN', SEARCH_SPACE['CHAR_MIN']),
            'CHAR_MAX': trial.suggest_categorical('CHAR_MAX', SEARCH_SPACE['CHAR_MAX']),
            'alpha': trial.suggest_categorical('alpha', SEARCH_SPACE['alpha']),
            'hidden_1': trial.suggest_categorical('hidden_1', SEARCH_SPACE['hidden_1']),
            'hidden_2': trial.suggest_categorical('hidden_2', SEARCH_SPACE['hidden_2'])
        }

        pipeline = Pipeline([
            ('vectorizer', HashingVectorizer(
                n_features=INPUT_SIZE, alternate_sign=True, norm=None,
                analyzer=CustomAnalyzer(params)
            )),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(params['hidden_1'], params['hidden_2']),
                alpha=params['alpha'], max_iter=300, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=10, random_state=RANDOM_SEED
            ))
        ])

        pipeline.fit(self.X_train, self.y_train)

        y_val_pred = pipeline.predict(self.X_val)
        val_recalls = recall_score(self.y_val, y_val_pred, average=None)
        val_fitness = (np.mean(val_recalls) * 0.4) + (np.min(val_recalls) * 0.6)

        y_train_pred = pipeline.predict(self.X_train)
        train_recalls = recall_score(self.y_train, y_train_pred, average=None)
        train_fitness = (np.mean(train_recalls) * 0.4) + (np.min(train_recalls) * 0.6)

        gen_gap = max(0, train_fitness - val_fitness)

        # 🔥 PENALTY LOGIC: Punish the score if Gen Gap > 8%
        final_score = val_fitness
        if gen_gap > 0.08:
            final_score -= (gen_gap - 0.08) * 1.5

        trial_info = {
            'trial_number': trial.number,
            'params': params,
            'val_fitness': float(val_fitness),
            'train_fitness': float(train_fitness),
            'gen_gap': float(gen_gap),
            'final_score': float(final_score),
            'val_mean_recall': float(np.mean(val_recalls)),
            'val_min_recall': float(np.min(val_recalls)),
            'n_iterations': pipeline.named_steps['classifier'].n_iter_
        }
        self.trial_log.append(trial_info)

        if val_fitness > self.best_actual_fitness:
            self.best_actual_fitness = val_fitness
            print(f"\n🏆 NEW BEST (Trial {trial.number + 1})")
            print(f"📈 Fitness: {val_fitness:.2%} | Final Score (w/ Penalty): {final_score:.2%}")

            gap_color = "\033[92m" if gen_gap < 0.05 else "\033[93m" if gen_gap < 0.08 else "\033[91m"
            print(f"🧬 Gen Gap: {gap_color}{gen_gap:+.2%}\033[0m (Train: {train_fitness:.2%})")

            cm = confusion_matrix(self.y_val, y_val_pred)
            confusions = []
            for r in range(len(self.categories)):
                for c in range(len(self.categories)):
                    if r != c and cm[r, c] > 0:
                        confusions.append((cm[r, c], self.categories[r], self.categories[c]))
            top_conf = sorted(confusions, reverse=True)[:2]
            if top_conf:
                print(f"⚔️  Top Confusions: {' | '.join([f'{r}→{c}({v})' for v, r, c in top_conf])}")

        return final_score


def export_to_cpp(pipeline, params, categories, filename="ModelWeights.h"):
    mlp = pipeline.named_steps['classifier']
    with open(filename, "w") as f:
        f.write(f"// AUTO-GENERATED ON {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n#include <Arduino.h>\n\n")
        f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define HIDDEN1_SIZE {params['hidden_1']}\n")
        f.write(f"#define HIDDEN2_SIZE {params['hidden_2']}\n")
        f.write(f"#define OUTPUT_SIZE {len(categories)}\n\n")
        f.write(f"#define CHAR_MIN {params['CHAR_MIN']}\n#define CHAR_MAX {params['CHAR_MAX']}\n")
        f.write(f"#define W_CHAR {params['W_CHAR']}\n#define W_WORD {params['W_WORD']}\n")
        f.write(f"#define W_BI {params['W_BI']}\n#define W_TRI {params['W_TRI']}\n")
        f.write(f"#define W_POS {params['W_POS']}\n\n")
        f.write("const char* const TOPICS[] = {\n")
        for cat in categories: f.write(f'    "{cat}",\n')
        f.write("};\n\n")

        def write_array(f, name, array):
            f.write(f"const float {name}[] PROGMEM = {{\n    ")
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:.6f}f")
                if i < len(flat) - 1: f.write(", ")
                if (i + 1) % 8 == 0: f.write("\n    ")
            f.write("\n};\n\n")

        write_array(f, "WEIGHTS_1", mlp.coefs_[0]);
        write_array(f, "BIAS_1", mlp.intercepts_[0])
        write_array(f, "WEIGHTS_2", mlp.coefs_[1]);
        write_array(f, "BIAS_2", mlp.intercepts_[1])
        write_array(f, "WEIGHTS_3", mlp.coefs_[2]);
        write_array(f, "BIAS_3", mlp.intercepts_[2])
        f.write("#endif\n")
    print(f"✅ Master Model exported to: {filename}")


def run_tuner():
    random.seed(RANDOM_SEED);
    np.random.seed(RANDOM_SEED)
    print(f"🚀 Starting Improved Hyperparameter Tuner\n🎲 Seed: {RANDOM_SEED} | 🔬 Trials: {N_TRIALS}\n")
    df = pd.read_csv(DATASET_FILE).dropna(subset=['french_sentence', 'topic'])
    le = LabelEncoder();
    y = le.fit_transform(df['topic']);
    categories = le.classes_
    X_trainval, X_test, y_trainval, y_test = train_test_split(df['french_sentence'], y, test_size=0.2,
                                                              random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_SEED,
                                                      stratify=y_trainval)
    print(f"📊 Dataset Split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}\n")
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = Objective(X_train, X_val, y_train, y_val, categories)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 80}\n🏁 TUNING COMPLETE | Best Fitness: {study.best_value:.2%}\n{'=' * 80}")
    best_params = study.best_params

    print("\nRetraining final model with best parameters...")
    final_pipeline = Pipeline([
        ('vectorizer', HashingVectorizer(n_features=INPUT_SIZE, alternate_sign=True, norm=None,
                                         analyzer=CustomAnalyzer(best_params))),
        ('classifier', MLPClassifier(hidden_layer_sizes=(best_params['hidden_1'], best_params['hidden_2']),
                                     alpha=best_params['alpha'], max_iter=300, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=10, random_state=RANDOM_SEED))
    ])
    final_pipeline.fit(X_trainval, y_trainval)

    y_test_pred = final_pipeline.predict(X_test)
    test_recalls = recall_score(y_test, y_test_pred, average=None)
    print(f"\n📊 FINAL TEST RECALL: Mean {np.mean(test_recalls):.2%} | Min {np.min(test_recalls):.2%}")
    print(classification_report(y_test, y_test_pred, target_names=categories))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"best_params": best_params, "categories": list(categories),
               "test_metrics": {"mean_recall": float(np.mean(test_recalls)), "min_recall": float(np.min(test_recalls))},
               "timestamp": timestamp}

    with open(f"best_results_{timestamp}.json", "w") as f: json.dump(results, f, indent=2)
    joblib.dump(final_pipeline, f"final_model_{timestamp}.joblib")
    export_to_cpp(final_pipeline, best_params, categories)
    pd.DataFrame(objective.trial_log).to_csv(f"trial_summary_{timestamp}.csv", index=False)


if __name__ == "__main__":
    run_tuner()