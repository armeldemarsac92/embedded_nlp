import pandas as pd
import numpy as np
import string, os, joblib, json, random, unicodedata
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, classification_report, balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from datetime import datetime
import logging

# --- CONFIG ---
DATASET_FILE = "DataSetTeensyv9_ULTRA_CLEAN.csv"
N_TRIALS = 80
RANDOM_SEED = 44
INPUT_SIZE = 8192

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expanded Stop Words
STOP_WORDS = set()

# Search space optimisé pour Optuna
SEARCH_SPACE = {
    'W_CHAR': (0, 3),
    'W_WORD': (0, 10),
    'W_BI': (2, 15),
    'W_TRI': (1, 15),
    'W_POS': (0, 6),
    'CHAR_MIN': (2, 3),
    'CHAR_MAX': (4, 5),
    'alpha': (1e-5, 1e-1),
    'learning_rate_init': (1e-4, 1e-2),
    'activation': ['relu', 'tanh'],
    'hidden_1': (32, 256),
    'hidden_2': (16, 128)
}


class CustomAnalyzer:
    """Picklable analyzer class for text feature extraction - WITH PADDING"""

    def __init__(self, params):
        self.params = params
        # Pré-compile la table de traduction pour performance
        self.punct_trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    @staticmethod
    def normalize_text(text):
        """Normalisation unicode avec removal des accents"""
        # NFD normalization + accent removal
        normalized = unicodedata.normalize('NFD', text)
        result = "".join([c for c in normalized if unicodedata.category(c) != 'Mn'])
        return result

    def __call__(self, text):
        if not isinstance(text, str):
            return []

        # Normalisation et nettoyage
        text = self.normalize_text(text)
        text = text.lower().translate(self.punct_trans)

        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words:
            return []

        p = self.params
        tokens = []

        # ==========================================
        # CHARACTER N-GRAMS WITH PADDING <word>
        # ==========================================
        if p['W_CHAR'] > 0:
            for word in words:
                padded = f"<{word}>"  # ✅ PADDING ACTIVÉ
                padded_len = len(padded)

                # ⚠️ CRITICAL: Position FIRST, then n-gram size
                for i in range(padded_len):
                    for n in range(p['CHAR_MIN'], p['CHAR_MAX'] + 1):
                        if i + n <= padded_len:
                            ngram = padded[i:i + n]
                            tokens.extend([f"C_{ngram}"] * p['W_CHAR'])

        # ==========================================
        # WORD UNIGRAMS
        # ==========================================
        if p['W_WORD'] > 0:
            tokens.extend([f"W_{w}" for w in words] * p['W_WORD'])

        # ==========================================
        # WORD BIGRAMS
        # ==========================================
        if p['W_BI'] > 0 and len(words) > 1:
            tokens.extend([f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p['W_BI'])

        # ==========================================
        # WORD TRIGRAMS
        # ==========================================
        if p['W_TRI'] > 0 and len(words) > 2:
            tokens.extend([f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p['W_TRI'])

        # ==========================================
        # POSITIONAL FEATURES
        # ==========================================
        if p['W_POS'] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p['W_POS'])

        return tokens


class Objective:
    """Objective function avec pruning et meilleures pratiques Optuna"""

    def __init__(self, X_train, X_val, y_train, y_val, categories):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.categories = categories
        self.trial_log = []
        self.best_actual_fitness = -1

    def __call__(self, trial):
        # Suggestion des hyperparamètres
        params = {
            'W_CHAR': trial.suggest_int('W_CHAR', SEARCH_SPACE['W_CHAR'][0], SEARCH_SPACE['W_CHAR'][1]),
            'W_WORD': trial.suggest_int('W_WORD', SEARCH_SPACE['W_WORD'][0], SEARCH_SPACE['W_WORD'][1]),
            'W_BI': trial.suggest_int('W_BI', SEARCH_SPACE['W_BI'][0], SEARCH_SPACE['W_BI'][1]),
            'W_TRI': trial.suggest_int('W_TRI', SEARCH_SPACE['W_TRI'][0], SEARCH_SPACE['W_TRI'][1]),
            'W_POS': trial.suggest_int('W_POS', SEARCH_SPACE['W_POS'][0], SEARCH_SPACE['W_POS'][1]),
            'CHAR_MIN': trial.suggest_int('CHAR_MIN', SEARCH_SPACE['CHAR_MIN'][0], SEARCH_SPACE['CHAR_MIN'][1]),
            'CHAR_MAX': trial.suggest_int('CHAR_MAX', SEARCH_SPACE['CHAR_MAX'][0], SEARCH_SPACE['CHAR_MAX'][1]),
            'alpha': trial.suggest_float('alpha', SEARCH_SPACE['alpha'][0], SEARCH_SPACE['alpha'][1], log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', SEARCH_SPACE['learning_rate_init'][0],
                                                      SEARCH_SPACE['learning_rate_init'][1], log=True),
            'activation': trial.suggest_categorical('activation', SEARCH_SPACE['activation']),
            'hidden_1': trial.suggest_int('hidden_1', SEARCH_SPACE['hidden_1'][0], SEARCH_SPACE['hidden_1'][1],
                                          step=16),
            'hidden_2': trial.suggest_int('hidden_2', SEARCH_SPACE['hidden_2'][0], SEARCH_SPACE['hidden_2'][1], step=8)
        }

        # Contrainte logique
        if params['CHAR_MIN'] > params['CHAR_MAX']:
            raise optuna.TrialPruned()

        try:
            # Construction du pipeline
            pipeline = Pipeline([
                ('vectorizer', HashingVectorizer(
                    n_features=INPUT_SIZE,
                    alternate_sign=True,
                    norm=None,
                    analyzer=CustomAnalyzer(params)
                )),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(params['hidden_1'], params['hidden_2']),
                    activation=params['activation'],
                    alpha=params['alpha'],
                    learning_rate_init=params['learning_rate_init'],
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=RANDOM_SEED,
                    verbose=False
                ))
            ])

            # Entraînement
            pipeline.fit(self.X_train, self.y_train)

            # Prédictions
            y_train_pred = pipeline.predict(self.X_train)
            y_val_pred = pipeline.predict(self.X_val)

            # Métriques
            train_recalls = recall_score(self.y_train, y_train_pred, average=None, zero_division=0)
            val_recalls = recall_score(self.y_val, y_val_pred, average=None, zero_division=0)

            train_balanced_acc = balanced_accuracy_score(self.y_train, y_train_pred)
            val_balanced_acc = balanced_accuracy_score(self.y_val, y_val_pred)

            train_mean_recall = np.mean(train_recalls)
            val_mean_recall = np.mean(val_recalls)
            val_min_recall = np.min(val_recalls)

            # Calcul du fitness avec pénalités
            overfitting_penalty = max(0, train_mean_recall - val_mean_recall - 0.05)
            complexity = params['hidden_1'] + params['hidden_2']
            complexity_penalty = (complexity / 400) * 0.02

            actual_fitness = val_mean_recall * 0.7 + val_min_recall * 0.3
            fitness = actual_fitness - (overfitting_penalty * 0.5) - complexity_penalty

            # Logging
            if actual_fitness > self.best_actual_fitness:
                self.best_actual_fitness = actual_fitness
                logger.info(f"🎯 New Best | Trial {trial.number} | Fitness: {fitness:.4f} | "
                            f"Val Mean: {val_mean_recall:.4f} | Val Min: {val_min_recall:.4f} | "
                            f"Val Balanced: {val_balanced_acc:.4f}")

            # Sauvegarde des logs
            self.trial_log.append({
                'trial': trial.number,
                'train_mean_recall': train_mean_recall,
                'val_mean_recall': val_mean_recall,
                'val_min_recall': val_min_recall,
                'val_balanced_acc': val_balanced_acc,
                'overfitting': train_mean_recall - val_mean_recall,
                'fitness': fitness,
                **params
            })

            # Pruning intermédiaire
            trial.report(fitness, trial.number)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return fitness

        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise
            logger.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()


def export_to_cpp(pipeline, params, categories):
    """Export du modèle au format C++ pour Teensy"""

    vectorizer = pipeline.named_steps['vectorizer']
    clf = pipeline.named_steps['classifier']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpp_filename = f"model_teensy_{timestamp}.h"

    with open(cpp_filename, 'w') as f:
        f.write("// Auto-generated MLP model for Teensy 4.1\n")
        f.write(f"// Generated: {timestamp}\n")
        f.write(f"// Input features: {INPUT_SIZE}\n\n")

        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write("#include <Arduino.h>\n\n")

        # Paramètres du vectorizer
        f.write("// Feature extraction parameters\n")
        f.write("namespace FeatureParams {\n")
        for key, val in params.items():
            if isinstance(val, (int, float)):
                f.write(f"    const int {key} = {val};\n")
        f.write("}\n\n")

        # Architecture du réseau
        f.write("// Network architecture\n")
        f.write(f"const int INPUT_SIZE = {INPUT_SIZE};\n")
        f.write(f"const int HIDDEN1_SIZE = {clf.hidden_layer_sizes[0]};\n")
        f.write(f"const int HIDDEN2_SIZE = {clf.hidden_layer_sizes[1]};\n")
        f.write(f"const int OUTPUT_SIZE = {len(categories)};\n\n")

        # Catégories
        f.write("// Categories\n")
        f.write(f"const char* CATEGORIES[] = {{\n")
        for cat in categories:
            f.write(f'    "{cat}",\n')
        f.write("};\n\n")

        # Poids et biais
        def write_matrix(name, matrix, comment=""):
            if comment:
                f.write(f"// {comment}\n")
            rows, cols = matrix.shape
            f.write(f"const float {name}[{rows}][{cols}] PROGMEM = {{\n")
            for row in matrix:
                f.write("    {" + ", ".join([f"{val:.6f}f" for val in row]) + "},\n")
            f.write("};\n\n")

        def write_vector(name, vector, comment=""):
            if comment:
                f.write(f"// {comment}\n")
            f.write(f"const float {name}[{len(vector)}] PROGMEM = {{\n    ")
            f.write(", ".join([f"{val:.6f}f" for val in vector]))
            f.write("\n};\n\n")

        write_matrix("W1", clf.coefs_[0].T, "Input -> Hidden1 weights")
        write_vector("b1", clf.intercepts_[0], "Hidden1 biases")
        write_matrix("W2", clf.coefs_[1].T, "Hidden1 -> Hidden2 weights")
        write_vector("b2", clf.intercepts_[1], "Hidden2 biases")
        write_matrix("W3", clf.coefs_[2].T, "Hidden2 -> Output weights")
        write_vector("b3", clf.intercepts_[2], "Output biases")

        # Fonction d'activation
        activation_func = {
            'relu': 'return x > 0 ? x : 0;',
            'tanh': 'return tanh(x);'
        }.get(params['activation'], 'return x;')

        f.write(f"// Activation function: {params['activation']}\n")
        f.write("inline float activation(float x) {\n")
        f.write(f"    {activation_func}\n")
        f.write("}\n\n")

        # Fonction softmax
        f.write("// Softmax for output layer\n")
        f.write("void softmax(float* input, int size) {\n")
        f.write("    float max_val = input[0];\n")
        f.write("    for(int i = 1; i < size; i++) if(input[i] > max_val) max_val = input[i];\n")
        f.write("    float sum = 0.0f;\n")
        f.write("    for(int i = 0; i < size; i++) {\n")
        f.write("        input[i] = exp(input[i] - max_val);\n")
        f.write("        sum += input[i];\n")
        f.write("    }\n")
        f.write("    for(int i = 0; i < size; i++) input[i] /= sum;\n")
        f.write("}\n\n")

        # Estimation mémoire
        total_params = sum([w.size for w in clf.coefs_] + [b.size for b in clf.intercepts_])
        memory_bytes = total_params * 4  # 4 bytes per float
        f.write(f"// Memory estimation: ~{memory_bytes / 1024:.2f} KB\n")
        f.write(f"// Total parameters: {total_params}\n\n")

        f.write("#endif // MODEL_H\n")

    logger.info(f"✅ C++ model exported to: {cpp_filename}")
    logger.info(f"   Parameters: {total_params} | Memory: ~{memory_bytes / 1024:.2f} KB")


def run_tuner():
    """Main tuning function avec toutes les améliorations"""

    logger.info("🚀 Starting Optimized Hyperparameter Tuner")
    logger.info(f"🎲 Seed: {RANDOM_SEED} | 🔬 Trials: {N_TRIALS}\n")

    # Chargement des données
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)

    # Validation des données
    required_cols = {'french_sentence', 'topic'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    df = df.dropna(subset=['french_sentence', 'topic'])
    logger.info(f"📊 Loaded {len(df)} samples")

    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(df['topic'])
    categories = le.classes_

    logger.info(f"🏷️  Categories: {len(categories)} - {list(categories)}")

    # Split des données
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df['french_sentence'], y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_trainval
    )

    logger.info(f"📊 Dataset Split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}\n")

    # Configuration Optuna
    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"nlp_teensy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    objective = Objective(X_train, X_val, y_train, y_val, categories)

    # Optimisation
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1  # MLP n'est pas thread-safe
    )

    logger.info(f"\n{'=' * 80}")
    logger.info(f"🏁 TUNING COMPLETE | Best Fitness: {study.best_value:.4f}")
    logger.info(f"{'=' * 80}\n")

    best_params = study.best_params
    logger.info("🎯 Best Parameters:")
    for key, val in best_params.items():
        logger.info(f"   {key}: {val}")

    # Réentraînement du modèle final
    logger.info("\n🔄 Retraining final model with best parameters...")

    final_pipeline = Pipeline([
        ('vectorizer', HashingVectorizer(
            n_features=INPUT_SIZE,
            alternate_sign=True,
            norm=None,
            analyzer=CustomAnalyzer(best_params)
        )),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(best_params['hidden_1'], best_params['hidden_2']),
            activation=best_params['activation'],
            alpha=best_params['alpha'],
            learning_rate_init=best_params['learning_rate_init'],
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_SEED,
            verbose=True
        ))
    ])

    final_pipeline.fit(X_trainval, y_trainval)

    # Évaluation finale
    y_test_pred = final_pipeline.predict(X_test)
    test_recalls = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 FINAL TEST METRICS")
    logger.info(f"{'=' * 80}")
    logger.info(f"Mean Recall: {np.mean(test_recalls):.4f}")
    logger.info(f"Min Recall: {np.min(test_recalls):.4f}")
    logger.info(f"Balanced Accuracy: {test_balanced_acc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_test_pred, target_names=categories)}")

    # Per-class recalls
    logger.info("\n📊 Per-Class Recall:")
    for cat, rec in zip(categories, test_recalls):
        logger.info(f"   {cat}: {rec:.4f}")

    # Vérification de convergence
    clf = final_pipeline.named_steps['classifier']
    logger.info(f"\n🔍 Model Convergence:")
    logger.info(f"   Iterations: {clf.n_iter_}")
    logger.info(f"   Loss: {clf.loss_:.6f}")

    # Sauvegarde des résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "best_params": best_params,
        "categories": list(categories),
        "test_metrics": {
            "mean_recall": float(np.mean(test_recalls)),
            "min_recall": float(np.min(test_recalls)),
            "balanced_accuracy": float(test_balanced_acc),
            "per_class_recall": {cat: float(rec) for cat, rec in zip(categories, test_recalls)}
        },
        "model_info": {
            "n_iterations": int(clf.n_iter_),
            "final_loss": float(clf.loss_),
            "input_size": INPUT_SIZE,
            "hidden_layers": [best_params['hidden_1'], best_params['hidden_2']],
            "total_params": sum([w.size for w in clf.coefs_] + [b.size for b in clf.intercepts_])
        },
        "training_config": {
            "n_trials": N_TRIALS,
            "random_seed": RANDOM_SEED,
            "dataset_file": DATASET_FILE,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "timestamp": timestamp
    }

    results_file = f"best_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    model_file = f"final_model_{timestamp}.joblib"
    joblib.dump(final_pipeline, model_file)

    trials_file = f"trial_summary_{timestamp}.csv"
    pd.DataFrame(objective.trial_log).to_csv(trials_file, index=False)

    cpp_file = export_to_cpp(final_pipeline, best_params, categories)

    # Visualisations Optuna
    try:
        from optuna.visualization import plot_param_importances, plot_optimization_history

        logger.info("\n📊 Generating Optuna visualizations...")

        # Importance des paramètres
        importance_fig = plot_param_importances(study)
        importance_fig.write_html(f"param_importance_{timestamp}.html")

        # Historique d'optimisation
        history_fig = plot_optimization_history(study)
        history_fig.write_html(f"optimization_history_{timestamp}.html")

        logger.info("📊 Visualization files generated")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    logger.info(f"\n✅ All files saved:")
    logger.info(f"   Results: {results_file}")
    logger.info(f"   Model: {model_file}")
    logger.info(f"   Trials: {trials_file}")
    logger.info(f"   C++ Export: model_teensy_{timestamp}.h")

    return study, final_pipeline, results


if __name__ == "__main__":
    study, model, results = run_tuner()
