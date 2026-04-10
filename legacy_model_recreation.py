from __future__ import annotations

import string
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "DataSetTeensyv9_ULTRA_CLEAN.csv"
OUT_DIR = ROOT / "artifacts" / "recreated_best_commit"


BEST_RUN = {
    "source_timestamp": "20260204_194813",
    "random_seed": 42,
    "n_features": 8192,
    "best_params": {
        "W_CHAR": 1,
        "W_WORD": 10,
        "W_BI": 5,
        "W_TRI": 0,
        "W_POS": 2,
        "CHAR_MIN": 2,
        "CHAR_MAX": 4,
        "alpha": 0.01,
        "hidden_1": 64,
        "hidden_2": 64,
        "activation": "relu",
        "learning_rate_init": 0.001,
    },
}


STOP_WORDS = set()


class CustomAnalyzer:
    """Legacy analyzer compatible with the archived sklearn pipeline."""

    def __init__(self, params: dict):
        self.params = params
        self.punct_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

    @staticmethod
    def normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    def __call__(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []

        text = self.normalize_text(text)
        text = text.lower().translate(self.punct_trans)
        words = [w for w in text.split()[:25] if w not in STOP_WORDS]
        if not words:
            return []

        p = self.params
        tokens: list[str] = []

        if p["W_CHAR"] > 0:
            for word in words:
                padded = f"<{word}>"
                padded_len = len(padded)
                for i in range(padded_len):
                    for n in range(p["CHAR_MIN"], p["CHAR_MAX"] + 1):
                        if i + n <= padded_len:
                            ngram = padded[i:i + n]
                            tokens.extend([f"C_{ngram}"] * p["W_CHAR"])

        if p["W_WORD"] > 0:
            tokens.extend([f"W_{w}" for w in words] * p["W_WORD"])

        if p["W_BI"] > 0 and len(words) > 1:
            tokens.extend(
                [f"B_{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)] * p["W_BI"]
            )

        if p["W_TRI"] > 0 and len(words) > 2:
            tokens.extend(
                [f"T_{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)] * p["W_TRI"]
            )

        if p["W_POS"] > 0 and len(words) > 0:
            tokens.extend([f"POS_START_{words[0]}", f"POS_END_{words[-1]}"] * p["W_POS"])

        return tokens


def build_pipeline() -> Pipeline:
    p = BEST_RUN["best_params"]
    return Pipeline(
        [
            (
                "vectorizer",
                HashingVectorizer(
                    n_features=BEST_RUN["n_features"],
                    alternate_sign=True,
                    norm=None,
                    analyzer=CustomAnalyzer(p),
                ),
            ),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(p["hidden_1"], p["hidden_2"]),
                    activation=p["activation"],
                    alpha=p["alpha"],
                    learning_rate_init=p["learning_rate_init"],
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=BEST_RUN["random_seed"],
                    verbose=False,
                ),
            ),
        ]
    )
