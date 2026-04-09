"""
Entraînement avec Optuna - Optimisation hyperparamètres
"""

import numpy as np
import pandas as pd
import optuna
import joblib
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report, recall_score, confusion_matrix
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Imports locaux
from config import CONFIG
from feature_extractor import FeatureExtractor
from collision_tracker import CollisionTracker
from model_exporter import ModelExporter
from bpe_tokenizer import BpeTokenizer

logger = logging.getLogger(__name__)


def _top_confusions(cm: np.ndarray, labels: List[str], top_k: int) -> List[dict]:
    """
    Return top off-diagonal confusions as list of dicts.
    """
    confusions = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                confusions.append({
                    'true': labels[i],
                    'pred': labels[j],
                    'count': count
                })
    confusions.sort(key=lambda x: -x['count'])
    return confusions[:top_k]


def _benchmark_vectorization(
    texts: List[str],
    n_features: int,
    weights: Dict[str, float],
    char_min: int,
    char_max: int,
    max_words: int,
    bpe: Optional[BpeTokenizer],
    repeats: int
) -> dict:
    """
    Compare custom FeatureExtractor vs sklearn HashingVectorizer speed.
    """
    from time import perf_counter
    from sklearn.feature_extraction.text import HashingVectorizer

    extractor = FeatureExtractor(
        n_features=n_features,
        weights=weights,
        char_ngram_min=char_min,
        char_ngram_max=char_max,
        max_words=max_words,
        bpe_tokenizer=bpe
    )

    def analyzer(text: str):
        # Mirror FeatureExtractor tokenization
        tokens = []
        words = extractor._get_words_for_benchmark(text)  # internal helper
        w_char = weights.get('w_char', 0.0)
        w_word = weights.get('w_word', 0.0)
        w_bigram = weights.get('w_bigram', 0.0)
        w_trigram = weights.get('w_trigram', 0.0)
        w_position = weights.get('w_position', 0.0)
        w_bpe = weights.get('w_bpe', 0.0)

        if w_char > 0:
            for word in words:
                padded = f"<{word}>"
                for n in range(char_min, char_max + 1):
                    for i in range(len(padded) - n + 1):
                        ngram = padded[i:i+n]
                        tokens.append(f"C_{ngram}")

        if w_word > 0:
            for word in words:
                tokens.append(f"W_{word}")

        if w_bigram > 0 and len(words) > 1:
            for i in range(len(words) - 1):
                tokens.append(f"B_{words[i]}_{words[i+1]}")

        if w_trigram > 0 and len(words) > 2:
            for i in range(len(words) - 2):
                tokens.append(f"T_{words[i]}_{words[i+1]}_{words[i+2]}")

        if w_position > 0 and len(words) > 0:
            tokens.append(f"POS_START_{words[0]}")
            tokens.append(f"POS_END_{words[-1]}")

        if bpe and w_bpe > 0:
            for word in words:
                for bpe_tok in bpe.tokenize(word):
                    tokens.append(bpe_tok)

        return tokens

    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=True,
        norm=None,
        analyzer=analyzer
    )

    # Custom extractor timing
    t_custom = 0.0
    for _ in range(repeats):
        t0 = perf_counter()
        _ = extractor.transform(texts)
        t_custom += perf_counter() - t0

    # Sklearn vectorizer timing
    t_sklearn = 0.0
    for _ in range(repeats):
        t0 = perf_counter()
        _ = vec.transform(texts)
        t_sklearn += perf_counter() - t0

    return {
        'custom_total_sec': t_custom,
        'sklearn_total_sec': t_sklearn,
        'custom_per_sample_ms': (t_custom / max(1, len(texts))) * 1000.0,
        'sklearn_per_sample_ms': (t_sklearn / max(1, len(texts))) * 1000.0,
        'samples': len(texts),
        'repeats': repeats
    }


class Objective:
    """Objectif Optuna pour optimisation"""

    def __init__(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        categories: List[str],
        bpe: Optional[BpeTokenizer] = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.categories = categories
        self.bpe = bpe
        self.trial_log = []

    def __call__(self, trial: optuna.Trial) -> float:
        fs = CONFIG.feature_search
        ms = CONFIG.model_search
        fit = CONFIG.fitness
        n_features = trial.suggest_int('n_features', ms.n_features[0], ms.n_features[1], step=ms.n_features[2])
        # === Hyperparamètres features ===
        weights = {
            'w_char': trial.suggest_float('w_char', fs.w_char[0], fs.w_char[1]),
            'w_word': trial.suggest_float('w_word', fs.w_word[0], fs.w_word[1]),
            'w_bigram': trial.suggest_float('w_bigram', fs.w_bigram[0], fs.w_bigram[1]),
            'w_trigram': trial.suggest_float('w_trigram', fs.w_trigram[0], fs.w_trigram[1]),
            'w_position': trial.suggest_float('w_position', fs.w_position[0], fs.w_position[1]),
            'w_bpe': trial.suggest_float('w_bpe', fs.w_bpe[0], fs.w_bpe[1]) if self.bpe else 0.0,
        }
        char_ngram_min = trial.suggest_int('char_min', fs.char_ngram_min[0], fs.char_ngram_min[1])
        char_ngram_max = trial.suggest_int('char_max', fs.char_ngram_max[0], fs.char_ngram_max[1])

        # Contrainte: char_max >= char_min
        if char_ngram_max < char_ngram_min:
            char_ngram_max = char_ngram_min

        # === Hyperparamètres MLP ===
        hidden1 = trial.suggest_int('hidden1', ms.hidden1[0], ms.hidden1[1], step=ms.hidden1[2])
        hidden2 = trial.suggest_int('hidden2', ms.hidden2[0], ms.hidden2[1], step=ms.hidden2[2])
        activation = trial.suggest_categorical('activation', list(ms.activation))
        alpha = trial.suggest_float('alpha', ms.alpha[0], ms.alpha[1], log=True)
        learning_rate = trial.suggest_float('learning_rate', ms.learning_rate[0], ms.learning_rate[1], log=True)

        try:
            # === Extraction features ===
            extractor = FeatureExtractor(
                n_features=n_features,
                weights=weights,
                char_ngram_min=char_ngram_min,
                char_ngram_max=char_ngram_max,
                max_words=CONFIG.tokenization.max_words,
                bpe_tokenizer=self.bpe
            )

            X_train_vec = extractor.transform(self.X_train)
            X_val_vec = extractor.transform(self.X_val)

            # === Entraînement ===
            clf = MLPClassifier(
                hidden_layer_sizes=(hidden1, hidden2),
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate,
                max_iter=CONFIG.training.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=CONFIG.training.early_stopping_patience,
                random_state=CONFIG.model.random_seed,
                verbose=False
            )

            clf.fit(X_train_vec, self.y_train)

            # === Évaluation ===
            y_train_pred = clf.predict(X_train_vec)
            y_val_pred = clf.predict(X_val_vec)

            train_recalls = recall_score(self.y_train, y_train_pred, average=None, zero_division=0)
            val_recalls = recall_score(self.y_val, y_val_pred, average=None, zero_division=0)

            train_mean_recall = float(np.mean(train_recalls))
            val_mean_recall = float(np.mean(val_recalls))
            val_min_recall = float(np.min(val_recalls))
            val_balanced_acc = float(balanced_accuracy_score(self.y_val, y_val_pred))

            # Pénalités
            overfitting_penalty = max(0.0, train_mean_recall - val_mean_recall - fit.overfit_margin)

            n_params = (
                n_features * hidden1 +
                hidden1 * hidden2 +
                hidden2 * len(np.unique(self.y_train))
            )
            size_penalty = min(fit.max_size_penalty, n_params / fit.size_penalty_divisor)

            collision_stats = extractor.collision_tracker.get_stats()
            collision_penalty = min(fit.max_collision_penalty, collision_stats.collision_rate * fit.collision_penalty_scale)

            resource_penalty = size_penalty + collision_penalty

            # Multi-objective (5 composantes)
            fitness = (
                val_mean_recall * fit.weight_val_mean_recall +
                val_min_recall * fit.weight_val_min_recall +
                val_balanced_acc * fit.weight_val_balanced_acc -
                overfitting_penalty -
                resource_penalty
            )

            # Log
            self.trial_log.append({
                'trial': trial.number,
                'train_mean_recall': train_mean_recall,
                'val_mean_recall': val_mean_recall,
                'val_min_recall': val_min_recall,
                'val_balanced_acc': val_balanced_acc,
                'overfitting': train_mean_recall - val_mean_recall,
                'n_params': n_params,
                'n_features': n_features,
                'collision_rate': collision_stats.collision_rate,
                'fill_rate': collision_stats.fill_rate,
                'max_bucket_size': collision_stats.max_bucket_size,
                'size_penalty': size_penalty,
                'collision_penalty': collision_penalty,
                'resource_penalty': resource_penalty,
                'fitness': fitness,
                'hidden1': hidden1,
                'hidden2': hidden2,
                'char_min': char_ngram_min,
                'char_max': char_ngram_max,
                **weights
            })

            # Logging détaillé par trial
            if CONFIG.training.optuna_log_every_n > 0 and (trial.number % CONFIG.training.optuna_log_every_n == 0):
                msg = (
                    f"Trial {trial.number} | fitness={fitness:.4f} | "
                    f"val_mean={val_mean_recall:.4f} | val_min={val_min_recall:.4f} | "
                    f"val_bal={val_balanced_acc:.4f} | n_features={n_features} | "
                    f"h=({hidden1},{hidden2})"
                )
                logger.info(msg)

                if CONFIG.training.optuna_log_params:
                    params_payload = {
                        "n_features": n_features,
                        "hidden1": hidden1,
                        "hidden2": hidden2,
                        "activation": activation,
                        "alpha": alpha,
                        "learning_rate": learning_rate,
                        "char_min": char_ngram_min,
                        "char_max": char_ngram_max,
                        **weights
                    }
                    logger.info("Trial %s params: %s", trial.number, json.dumps(params_payload, sort_keys=True))

                if CONFIG.training.optuna_log_metrics:
                    metrics_payload = {
                        "fitness": fitness,
                        "train_mean_recall": train_mean_recall,
                        "val_mean_recall": val_mean_recall,
                        "val_min_recall": val_min_recall,
                        "val_balanced_acc": val_balanced_acc,
                        "overfitting_penalty": overfitting_penalty,
                        "n_params": n_params,
                        "size_penalty": size_penalty,
                        "collision_rate": collision_stats.collision_rate,
                        "fill_rate": collision_stats.fill_rate,
                        "max_bucket_size": collision_stats.max_bucket_size,
                        "collision_penalty": collision_penalty,
                        "resource_penalty": resource_penalty
                    }
                    logger.info("Trial %s metrics: %s", trial.number, json.dumps(metrics_payload, sort_keys=True))

                if CONFIG.training.optuna_log_confusions:
                    cm = confusion_matrix(self.y_val, y_val_pred, labels=np.arange(len(self.categories)))
                    top_conf = _top_confusions(cm, self.categories, CONFIG.training.optuna_top_confusions)
                    if top_conf:
                        logger.info("Top confusions: " + "; ".join(
                            f"{c['true']}→{c['pred']} ({c['count']})" for c in top_conf
                        ))

            trial.report(fitness, step=clf.n_iter_)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return fitness

        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise
            logger.exception("Trial %s failed (params=%s)", trial.number, trial.params)
            raise optuna.TrialPruned()


def run_training(
    df: pd.DataFrame,
    text_column: str,
    label_column: str
) -> Tuple[optuna.Study, MLPClassifier, Dict, List[str], Dict[str, Any]]:
    """
    Lance l'entraînement complet avec Optuna.

    Returns:
        (study, model, best_params, categories, feature_params)
    """
    cfg = CONFIG

    # === Préparation données ===
    logger.info("Preparing data...")

    # Nettoyer
    df = df.dropna(subset=[text_column, label_column])

    X = df[text_column].astype(str).tolist()

    # Encoder labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])
    categories = le.classes_.tolist()

    logger.info(f"Classes: {categories}")
    logger.info(f"Samples per class: {dict(zip(*np.unique(y, return_counts=True)))}")

    # === Splits ===
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg.model.test_size,
        random_state=cfg.model.random_seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=cfg.model.val_size / (1 - cfg.model.test_size),
        random_state=cfg.model.random_seed,
        stratify=y_trainval
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # === BPE (optionnel) ===
    bpe = None
    if cfg.model.use_bpe:
        logger.info("Training BPE tokenizer...")
        bpe = BpeTokenizer(
            vocab_size=cfg.model.bpe_vocab_size,
            min_freq=cfg.model.bpe_min_freq,
            max_words=cfg.tokenization.max_words
        )
        bpe.fit(X_train)

    # === Analyse collisions (debug) ===
    logger.info("Analyzing hash collisions...")
    fs = cfg.feature_search
    test_weights = {
        'w_char': 1.0 if fs.w_char[1] > 0 else 0.0,
        'w_word': 1.0 if fs.w_word[1] > 0 else 0.0,
        'w_bigram': 1.0 if fs.w_bigram[1] > 0 else 0.0,
        'w_trigram': 1.0 if fs.w_trigram[1] > 0 else 0.0,
        'w_position': 1.0 if fs.w_position[1] > 0 else 0.0,
        'w_bpe': 1.0 if (bpe and fs.w_bpe[1] > 0) else 0.0,
    }
    test_char_min = int(round((fs.char_ngram_min[0] + fs.char_ngram_min[1]) / 2))
    test_char_max = int(round((fs.char_ngram_max[0] + fs.char_ngram_max[1]) / 2))
    if test_char_max < test_char_min:
        test_char_max = test_char_min

    ms = cfg.model_search
    test_n_features = int(round((ms.n_features[0] + ms.n_features[1]) / 2))
    test_n_features = max(ms.n_features[0], min(ms.n_features[1], test_n_features))
    test_n_features = (test_n_features // ms.n_features[2]) * ms.n_features[2]
    test_extractor = FeatureExtractor(
        n_features=test_n_features,
        weights=test_weights,
        char_ngram_min=test_char_min,
        char_ngram_max=test_char_max,
        max_words=cfg.tokenization.max_words,
        bpe_tokenizer=bpe
    )
    _ = test_extractor.transform(X_train[:cfg.debug.collision_sample_size])
    test_extractor.collision_tracker.print_report()

    # === Benchmark vectorization (optional) ===
    if cfg.training.run_benchmark:
        logger.info("Running vectorization benchmark...")
        sample_texts = X_train[:cfg.training.benchmark_samples]
        if len(sample_texts) == 0:
            logger.info("Benchmark skipped (no samples).")
        else:
            fs = cfg.feature_search
            ms = cfg.model_search
            bench_weights = {
                'w_char': 1.0 if fs.w_char[1] > 0 else 0.0,
                'w_word': 1.0 if fs.w_word[1] > 0 else 0.0,
                'w_bigram': 1.0 if fs.w_bigram[1] > 0 else 0.0,
                'w_trigram': 1.0 if fs.w_trigram[1] > 0 else 0.0,
                'w_position': 1.0 if fs.w_position[1] > 0 else 0.0,
                'w_bpe': 1.0 if (bpe and fs.w_bpe[1] > 0) else 0.0,
            }
            bench_char_min = int(round((fs.char_ngram_min[0] + fs.char_ngram_min[1]) / 2))
            bench_char_max = int(round((fs.char_ngram_max[0] + fs.char_ngram_max[1]) / 2))
            if bench_char_max < bench_char_min:
                bench_char_max = bench_char_min
            bench_n_features = int(round((ms.n_features[0] + ms.n_features[1]) / 2))
            bench_n_features = (bench_n_features // ms.n_features[2]) * ms.n_features[2]
            bench = _benchmark_vectorization(
                texts=sample_texts,
                n_features=bench_n_features,
                weights=bench_weights,
                char_min=bench_char_min,
                char_max=bench_char_max,
                max_words=cfg.tokenization.max_words,
                bpe=bpe,
                repeats=cfg.training.benchmark_repeats
            )
            logger.info(
                f"Benchmark: samples={bench['samples']} repeats={bench['repeats']} | "
                f"custom={bench['custom_total_sec']:.3f}s ({bench['custom_per_sample_ms']:.3f} ms/sample) | "
                f"sklearn={bench['sklearn_total_sec']:.3f}s ({bench['sklearn_per_sample_ms']:.3f} ms/sample)"
            )

    # === Optuna ===
    objective = Objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categories=categories,
        bpe=bpe
    )

    sampler = TPESampler(
        seed=cfg.model.random_seed,
        multivariate=cfg.optuna.tpe_multivariate,
        n_startup_trials=cfg.optuna.tpe_startup_trials
    )

    pruner = HyperbandPruner(
        min_resource=cfg.optuna.hyperband_min_resource,
        max_resource=cfg.training.n_trials,
        reduction_factor=cfg.optuna.hyperband_reduction_factor
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    logger.info(f"Starting Optuna optimization ({cfg.training.n_trials} trials)...")
    study.optimize(
        objective,
        n_trials=cfg.training.n_trials,
        show_progress_bar=cfg.training.show_progress_bar
    )

    # === Meilleurs paramètres ===
    best_params = study.best_params
    logger.info(f"\n{'='*60}")
    logger.info("BEST PARAMETERS:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Best fitness: {study.best_value:.4f}")
    logger.info(f"{'='*60}\n")

    # === Réentraînement final ===
    logger.info("Training final model on train+val...")

    final_weights = {
        'w_char': best_params['w_char'],
        'w_word': best_params['w_word'],
        'w_bigram': best_params['w_bigram'],
        'w_trigram': best_params['w_trigram'],
        'w_position': best_params['w_position'],
        'w_bpe': best_params.get('w_bpe', 0.0),
    }
    final_char_min = best_params['char_min']
    final_char_max = best_params['char_max']
    if final_char_max < final_char_min:
        final_char_max = final_char_min

    best_n_features = best_params['n_features']

    final_extractor = FeatureExtractor(
        n_features=best_n_features,
        weights=final_weights,
        char_ngram_min=final_char_min,
        char_ngram_max=final_char_max,
        max_words=cfg.tokenization.max_words,
        bpe_tokenizer=bpe
    )

    X_trainval_vec = final_extractor.transform(X_trainval)
    X_test_vec = final_extractor.transform(X_test)

    final_clf = MLPClassifier(
        hidden_layer_sizes=(best_params['hidden1'], best_params['hidden2']),
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate'],
        max_iter=cfg.training.max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=cfg.training.early_stopping_patience,
        random_state=cfg.model.random_seed,
        verbose=True
    )

    final_clf.fit(X_trainval_vec, y_trainval)

    # === Évaluation finale ===
    y_test_pred = final_clf.predict(X_test_vec)
    test_recalls = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_mean_recall = float(np.mean(test_recalls))
    test_min_recall = float(np.min(test_recalls))
    test_acc = float(balanced_accuracy_score(y_test, y_test_pred))

    logger.info(f"\n{'='*60}")
    logger.info("FINAL TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Recall: {test_mean_recall:.4f}")
    logger.info(f"Min Recall: {test_min_recall:.4f}")
    logger.info(f"Balanced Accuracy: {test_acc:.4f}")
    logger.info(f"\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=categories,
        digits=4
    ))

    # === Export ===
    exporter = ModelExporter(
        clf=final_clf,
        params={
            **final_weights,
            'char_ngram_min': final_char_min,
            'char_ngram_max': final_char_max,
            'max_words': cfg.tokenization.max_words,
        },
        categories=categories,
        n_features=best_n_features
    )

    # Export float32
    if cfg.model.export_float32:
        exporter.export_float32(f"{cfg.paths.cpp_output_dir}/ModelWeights.h")

    # Export INT8
    if cfg.model.use_quantization:
        exporter.export_int8(f"{cfg.paths.cpp_output_dir}/ModelWeightsQ.h")

    # Export code de vérification
    test_samples = X_test[:5]
    exporter.export_verification_code(
        f"{cfg.paths.cpp_output_dir}/VerificationTests.h",
        test_samples
    )

    # Export BPE patterns si utilisé
    if bpe:
        bpe.export_cpp(f"{cfg.paths.cpp_output_dir}/BpePatterns.h")

    # Sauvegarde log des trials
    trial_df = pd.DataFrame(objective.trial_log)
    trial_df.to_csv(f"{cfg.paths.output_dir}/trials_log.csv", index=False)

    # === Sauvegarde best_results + joblib ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "best_params": best_params,
        "categories": categories,
        "test_metrics": {
            "mean_recall": test_mean_recall,
            "min_recall": test_min_recall,
            "balanced_accuracy": test_acc,
            "per_class_recall": {cat: float(rec) for cat, rec in zip(categories, test_recalls)}
        },
        "model_info": {
            "input_size": best_n_features,
            "hidden_layers": [best_params['hidden1'], best_params['hidden2']],
            "total_params": int(
                best_n_features * best_params['hidden1'] +
                best_params['hidden1'] * best_params['hidden2'] +
                best_params['hidden2'] * len(categories) +
                best_params['hidden1'] + best_params['hidden2'] + len(categories)
            )
        },
        "training_config": {
            "n_trials": cfg.training.n_trials,
            "random_seed": cfg.model.random_seed,
            "dataset_file": cfg.paths.dataset,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "timestamp": timestamp
    }

    results_file = f"{cfg.paths.output_dir}/best_results_{timestamp}.json"
    with open(results_file, "w") as f:
        import json
        json.dump(results, f, indent=2)

    model_payload = {
        "classifier": final_clf,
        "feature_params": {
            **final_weights,
            "char_ngram_min": final_char_min,
            "char_ngram_max": final_char_max,
            "max_words": cfg.tokenization.max_words
        },
        "n_features": best_n_features,
        "categories": categories,
        "bpe": bpe
    }

    model_file = f"{cfg.paths.output_dir}/final_model_{timestamp}.joblib"
    joblib.dump(model_payload, model_file)

    final_feature_params = {
        **final_weights,
        "char_ngram_min": final_char_min,
        "char_ngram_max": final_char_max,
        "max_words": cfg.tokenization.max_words,
        "n_features": best_n_features
    }

    return study, final_clf, best_params, categories, final_feature_params
