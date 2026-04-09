"""
Configuration centralisée pour l'entraînement NLP embarqué.
Tous les paramètres modifiables sont ici.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, os.pardir))


@dataclass
class PathConfig:
    """Chemins des fichiers"""
    dataset: str = os.path.join(_PROJECT_ROOT, "data", "DataSetTeensyv9_ULTRA_CLEAN.csv")
    output_dir: str = os.path.join(_PROJECT_ROOT, "artifacts", "current", "training")
    cpp_output_dir: str = os.path.join(_PROJECT_ROOT, "artifacts", "current", "cpp", "generated")

    def __post_init__(self):
        self.ensure_dirs()

    def _normalize_paths(self):
        self.dataset = os.path.abspath(os.path.expanduser(self.dataset))
        self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        self.cpp_output_dir = os.path.abspath(os.path.expanduser(self.cpp_output_dir))

    def ensure_dirs(self):
        self._normalize_paths()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cpp_output_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration du modèle"""
    input_size: int = 8192
    random_seed: int = 44
    test_size: float = 0.15
    val_size: float = 0.15

    # Quantification
    use_quantization: bool = True
    export_float32: bool = True

    # BPE
    use_bpe: bool = True
    bpe_vocab_size: int = 300
    bpe_min_freq: int = 10


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    n_trials: int = 60
    max_iter: int = 500
    early_stopping_patience: int = 20
    n_jobs: int = -1
    show_progress_bar: bool = True
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    # Logging Optuna
    optuna_log_every_n: int = 1
    optuna_log_confusions: bool = True
    optuna_top_confusions: int = 5
    optuna_log_params: bool = True
    optuna_log_metrics: bool = True

    # Benchmarking
    run_benchmark: bool = False
    benchmark_samples: int = 2000
    benchmark_repeats: int = 3


@dataclass
class OptunaConfig:
    """Configuration Optuna (sampler/pruner)"""
    # TPESampler
    tpe_multivariate: bool = True
    tpe_startup_trials: int = 10

    # HyperbandPruner
    hyperband_min_resource: int = 1
    hyperband_reduction_factor: int = 3


@dataclass
class FitnessConfig:
    """Configuration de la fonction de fitness"""
    # Poids des composantes
    weight_val_mean_recall: float = 0.4
    weight_val_min_recall: float = 0.3
    weight_val_balanced_acc: float = 0.3

    # Pénalités
    overfit_margin: float = 0.09
    max_size_penalty: float = 0.05
    size_penalty_divisor: float = 5_000_000.0
    max_collision_penalty: float = 0.05
    collision_penalty_scale: float = 0.5


@dataclass
class DebugConfig:
    """Paramètres de debug/analyse"""
    collision_sample_size: int = 500


@dataclass
class FeatureSearchSpace:
    """
    Espace de recherche Optuna pour les features.
    """
    w_char: tuple[float, float] = (2.0, 4.0)
    w_word: tuple[float, float] = (8.0, 10.0)
    w_bigram: tuple[float, float] = (1.0, 4.0)
    w_trigram: tuple[float, float] = (0.0, 2.0)
    w_position: tuple[float, float] = (3.0, 5.0)
    w_bpe: tuple[float, float] = (0.0, 2.0)
    char_ngram_min: tuple[int, int] = (2, 3)
    char_ngram_max: tuple[int, int] = (4, 6)


@dataclass
class ModelSearchSpace:
    """
    Espace de recherche Optuna pour le MLP.
    """
    n_features: tuple[int, int, int] = (7168, 9216, 512)
    hidden1: tuple[int, int, int] = (64, 112, 16)
    hidden2: tuple[int, int, int] = (80, 144, 16)
    activation: tuple[str, ...] = ("tanh", "relu")
    alpha: tuple[float, float] = (1e-2, 1e-1)
    learning_rate: tuple[float, float] = (2e-4, 2e-3)


@dataclass
class TokenizationConfig:
    """
    Paramètres fixes de tokenization (non optimisés).
    """
    max_words: int = 25


@dataclass
class Config:
    """Configuration globale"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    feature_search: FeatureSearchSpace = field(default_factory=FeatureSearchSpace)
    model_search: ModelSearchSpace = field(default_factory=ModelSearchSpace)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)


# Instance globale
CONFIG = Config()
