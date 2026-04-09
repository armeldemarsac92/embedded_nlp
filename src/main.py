"""
Point d'entrée principal pour l'entraînement NLP embarqué.
"""

import pandas as pd
import sys
from datetime import datetime
import logging

from config import CONFIG
from hash_utils import verify_sklearn_compatibility
from trainer import run_training


def main():
    print("=" * 70)
    print("  NLP TRAINING FOR EMBEDDED SYSTEMS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # === Logging setup ===
    logging.basicConfig(
        level=getattr(logging, CONFIG.training.log_level.upper(), logging.INFO),
        format=CONFIG.training.log_format
    )

    # === Vérification hash ===
    print("\n📋 Step 1: Verifying hash compatibility...")
    if not verify_sklearn_compatibility(CONFIG.model.input_size):
        print("❌ FATAL: Hash implementation incompatible with sklearn!")
        print("   The C++ code will NOT produce the same results.")
        sys.exit(1)

    # === Chargement données ===
    dataset_path = CONFIG.paths.dataset
    text_col = 'french_sentence'
    label_col = 'topic'

    print(f"\n📋 Step 2: Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"   Loaded {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Classes: {df[label_col].nunique()}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Vérification colonnes
    if text_col not in df.columns:
        print(f"❌ Column '{text_col}' not found in dataset")
        sys.exit(1)
    if label_col not in df.columns:
        print(f"❌ Column '{label_col}' not found in dataset")
        sys.exit(1)

    # === Entraînement ===
    print(f"\n📋 Step 3: Training with {CONFIG.training.n_trials} Optuna trials...")
    print(f"   Input size: {CONFIG.model.input_size}")
    print(f"   Quantization: {'enabled' if CONFIG.model.use_quantization else 'disabled'}")
    print(f"   BPE: {'enabled' if CONFIG.model.use_bpe else 'disabled'}")
    print()

    study, clf, best_params, categories, feature_params = run_training(
        df=df,
        text_column=text_col,
        label_column=label_col
    )

    # === Résumé final ===
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output files:")
    print(f"   - {CONFIG.paths.cpp_output_dir}/ModelWeights.h (float32)")
    if CONFIG.model.use_quantization:
        print(f"   - {CONFIG.paths.cpp_output_dir}/ModelWeightsQ.h (int8)")
    if CONFIG.model.use_bpe:
        print(f"   - {CONFIG.paths.cpp_output_dir}/BpePatterns.h")
    print(f"   - {CONFIG.paths.cpp_output_dir}/VerificationTests.h")
    print(f"   - {CONFIG.paths.output_dir}/trials_log.csv")

    print(f"\n📊 Best configuration:")
    print(f"   Fitness: {study.best_value:.4f}")
    print(f"   Hidden layers: ({best_params['hidden1']}, {best_params['hidden2']})")
    print(f"   Activation: {best_params['activation']}")

    print(f"\n📝 Next steps:")
    print(f"   1. Copy generated/*.h files to your Teensy project")
    print(f"   2. Run VerificationTests to confirm Python/C++ match")
    print(f"   3. Deploy and test!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
