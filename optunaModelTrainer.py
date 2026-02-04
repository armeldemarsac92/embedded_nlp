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
DATASET_FILE = "DataSetTeensyv8.csv"
N_TRIALS = 70
RANDOM_SEED = 42
INPUT_SIZE = 8192
STOP_WORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'ci', 'ca', 'et', 'en'}

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
        self.best_fitness = -1

    def __call__(self, trial):
        # Sample hyperparameters
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

        # Build pipeline with early stopping
        pipeline = Pipeline([
            ('vectorizer', HashingVectorizer(
                n_features=INPUT_SIZE,
                alternate_sign=True,
                norm=None,
                analyzer=CustomAnalyzer(params)
            )),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(params['hidden_1'], params['hidden_2']),
                alpha=params['alpha'],
                max_iter=300,
                early_stopping=True,  # Enable early stopping
                validation_fraction=0.1,  # Use 10% of training for early stopping
                n_iter_no_change=10,  # Stop if no improvement for 10 iterations
                random_state=RANDOM_SEED
            ))
        ])

        # Train
        pipeline.fit(self.X_train, self.y_train)

        # Validation performance
        y_val_pred = pipeline.predict(self.X_val)
        val_recalls = recall_score(self.y_val, y_val_pred, average=None)
        val_fitness = (np.mean(val_recalls) * 0.4) + (np.min(val_recalls) * 0.6)

        # Training performance (for overfitting detection)
        y_train_pred = pipeline.predict(self.X_train)
        train_recalls = recall_score(self.y_train, y_train_pred, average=None)
        train_fitness = (np.mean(train_recalls) * 0.4) + (np.min(train_recalls) * 0.6)

        # Generalization gap
        gen_gap = train_fitness - val_fitness

        # Log this trial
        trial_info = {
            'trial_number': trial.number,
            'params': params,
            'val_fitness': float(val_fitness),
            'train_fitness': float(train_fitness),
            'gen_gap': float(gen_gap),
            'val_mean_recall': float(np.mean(val_recalls)),
            'val_min_recall': float(np.min(val_recalls)),
            'val_max_recall': float(np.max(val_recalls)),
            'train_mean_recall': float(np.mean(train_recalls)),
            'train_min_recall': float(np.min(train_recalls)),
            'n_iterations': pipeline.named_steps['classifier'].n_iter_
        }
        self.trial_log.append(trial_info)

        # Print if best
        if val_fitness > self.best_fitness:
            self.best_fitness = val_fitness
            print(f"\n🏆 NEW BEST (Trial {trial.number + 1})")
            print(f"📈 Fitness: {val_fitness:.2%} | Worst Class: {np.min(val_recalls):.2%}")

            gap_color = "\033[92m" if gen_gap < 0.05 else "\033[93m" if gen_gap < 0.12 else "\033[91m"
            print(f"🧬 Gen Gap: {gap_color}{gen_gap:+.2%}\033[0m (Train: {train_fitness:.2%})")
            print(f"⏱️  Converged in {pipeline.named_steps['classifier'].n_iter_} iterations")

            # Confusion info
            cm = confusion_matrix(self.y_val, y_val_pred)
            confusions = []
            for r in range(len(self.categories)):
                for c in range(len(self.categories)):
                    if r != c and cm[r, c] > 0:
                        confusions.append((cm[r, c], self.categories[r], self.categories[c]))
            top_conf = sorted(confusions, reverse=True)[:2]
            if top_conf:
                print(f"⚔️  Top Confusions: {' | '.join([f'{r}→{c}({v})' for v, r, c in top_conf])}")

        return val_fitness


def export_to_cpp(pipeline, params, categories, filename="ModelWeights.h"):
    """Extracts weights from the trained MLP and writes the Arduino header"""
    mlp = pipeline.named_steps['classifier']

    with open(filename, "w") as f:
        f.write(f"// AUTO-GENERATED ON {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n#include <Arduino.h>\n\n")

        # --- 1. Dimensions & Hashing Params ---
        f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define HIDDEN1_SIZE {params['hidden_1']}\n")
        f.write(f"#define HIDDEN2_SIZE {params['hidden_2']}\n")
        f.write(f"#define OUTPUT_SIZE {len(categories)}\n\n")

        f.write(f"#define CHAR_MIN {params['CHAR_MIN']}\n#define CHAR_MAX {params['CHAR_MAX']}\n")
        f.write(f"#define W_CHAR {params['W_CHAR']}\n#define W_WORD {params['W_WORD']}\n")
        f.write(f"#define W_BI {params['W_BI']}\n#define W_TRI {params['W_TRI']}\n")
        f.write(f"#define W_POS {params['W_POS']}\n\n")

        # --- 2. Topic Names ---
        f.write("const char* const TOPICS[] = {\n")
        for cat in categories:
            f.write(f'    "{cat}",\n')
        f.write("};\n\n")

        # --- 3. Neural Network Weights (PROGMEM) ---
        def write_array(f, name, array):
            f.write(f"const float {name}[] PROGMEM = {{\n    ")
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:.6f}f")
                if i < len(flat) - 1: f.write(", ")
                if (i + 1) % 8 == 0: f.write("\n    ")
            f.write("\n};\n\n")

        # Layer 1 weights and bias
        write_array(f, "WEIGHTS_1", mlp.coefs_[0])
        write_array(f, "BIAS_1", mlp.intercepts_[0])

        # Layer 2 weights and bias
        write_array(f, "WEIGHTS_2", mlp.coefs_[1])
        write_array(f, "BIAS_2", mlp.intercepts_[1])

        # Output layer weights and bias
        write_array(f, "WEIGHTS_3", mlp.coefs_[2])
        write_array(f, "BIAS_3", mlp.intercepts_[2])

        f.write("#endif\n")
    print(f"✅ Master Model exported to: {filename}")

def run_tuner():
    # Set all random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"🚀 Starting Improved Hyperparameter Tuner")
    print(f"🎲 Random Seed: {RANDOM_SEED}")
    print(f"🔬 Trials: {N_TRIALS}\n")

    # Load data and create train/val/test split
    df = pd.read_csv(DATASET_FILE).dropna(subset=['french_sentence', 'topic'])

    le = LabelEncoder()
    y = le.fit_transform(df['topic'])
    categories = le.classes_

    # First split: separate test set (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df['french_sentence'], y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: separate train and validation (80/20 of remaining data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
        random_state=RANDOM_SEED,
        stratify=y_trainval
    )

    print(f"📊 Dataset Split:")
    print(f"   Train: {len(X_train)} samples ({len(X_train) / len(df) * 100:.1f}%)")
    print(f"   Val:   {len(X_val)} samples ({len(X_val) / len(df) * 100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples ({len(X_test) / len(df) * 100:.1f}%)\n")

    # Create Optuna study with TPE sampler for efficiency
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='french_text_classifier_tuning'
    )

    # Create objective
    objective = Objective(X_train, X_val, y_train, y_val, categories)

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 80}")
    print(f"🏁 TUNING COMPLETE")
    print(f"{'=' * 80}\n")

    # Get best parameters
    best_params = study.best_params
    best_val_fitness = study.best_value

    print(f"🥇 Best Validation Fitness: {best_val_fitness:.2%}")
    print(f"⚙️  Best Parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Retrain on full train+val with best parameters for final test evaluation
    print(f"\n{'=' * 80}")
    print(f"🔬 FINAL TEST SET EVALUATION")
    print(f"{'=' * 80}\n")

    print("Retraining with best parameters on train+val set...")

    final_pipeline = Pipeline([
        ('vectorizer', HashingVectorizer(
            n_features=INPUT_SIZE,
            alternate_sign=True,
            norm=None,
            analyzer=CustomAnalyzer(best_params)
        )),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(best_params['hidden_1'], best_params['hidden_2']),
            alpha=best_params['alpha'],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_SEED
        ))
    ])

    final_pipeline.fit(X_trainval, y_trainval)

    # Test set evaluation
    y_test_pred = final_pipeline.predict(X_test)
    test_recalls = recall_score(y_test, y_test_pred, average=None)
    test_fitness = (np.mean(test_recalls) * 0.4) + (np.min(test_recalls) * 0.6)

    print(f"\n📊 TEST SET RESULTS:")
    print(f"   Fitness Score: {test_fitness:.2%}")
    print(f"   Mean Recall: {np.mean(test_recalls):.2%}")
    print(f"   Min Recall: {np.min(test_recalls):.2%}")
    print(f"   Max Recall: {np.max(test_recalls):.2%}")
    print(f"   Converged in {final_pipeline.named_steps['classifier'].n_iter_} iterations")

    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=categories))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(f"{'':12s} " + " ".join([f"{cat[:8]:>8s}" for cat in categories]))
    for i, cat in enumerate(categories):
        print(f"{cat[:12]:12s} " + " ".join([f"{cm[i, j]:8d}" for j in range(len(categories))]))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all trial logs
    trial_log_file = f"trial_log_{timestamp}.json"
    with open(trial_log_file, "w") as f:
        json.dump(objective.trial_log, f, indent=2)
    print(f"\n💾 Full trial log saved to: {trial_log_file}")

    # Save best results and test performance
    results = {
        "best_params": best_params,
        "validation_fitness": float(best_val_fitness),
        "test_fitness": float(test_fitness),
        "test_metrics": {
            "mean_recall": float(np.mean(test_recalls)),
            "min_recall": float(np.min(test_recalls)),
            "max_recall": float(np.max(test_recalls)),
            "per_class_recall": {cat: float(rec) for cat, rec in zip(categories, test_recalls)}
        },
        "categories": list(categories),
        "n_trials": N_TRIALS,
        "random_seed": RANDOM_SEED,
        "timestamp": timestamp
    }

    results_file = f"best_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Best results saved to: {results_file}")

    # Save the final model
    model_file = f"final_model_{timestamp}.joblib"
    try:
        joblib.dump(final_pipeline, model_file)
        print(f"💾 Final model saved to: {model_file}")

        # NEW: Export the C++ header for Teensy
        try:
            export_to_cpp(final_pipeline, best_params, categories)
        except Exception as e:
            print(f"⚠️  C++ Export failed: {e}")

        print(f"\n✅ All done! Check the saved files for detailed results.")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")
        print(f"   Model is still available in memory as 'final_pipeline'")
        # Save just the parameters as a fallback
        fallback_file = f"model_params_only_{timestamp}.json"
        with open(fallback_file, "w") as f:
            json.dump({
                "best_params": best_params,
                "model_config": {
                    "hidden_layers": (best_params['hidden_1'], best_params['hidden_2']),
                    "alpha": best_params['alpha'],
                    "n_features": INPUT_SIZE
                }
            }, f, indent=2)
        print(f"💾 Model parameters saved to: {fallback_file}")

    # Save a summary visualization-ready CSV
    summary_df = pd.DataFrame(objective.trial_log)
    summary_csv = f"trial_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"💾 Trial summary CSV saved to: {summary_csv}")

    print(f"\n✅ All done! Check the saved files for detailed results.")


if __name__ == "__main__":
    run_tuner()