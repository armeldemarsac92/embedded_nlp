"""
Extracteur de features avec hashing trick.
Compatible avec le code C++ pour Teensy.
"""

import numpy as np
from typing import List, Dict

from hash_utils import hash_to_index_and_sign
from collision_tracker import CollisionTracker
from text_normalizer import tokenize_words


class FeatureExtractor:
    """
    Extracteur de features compatible C++.

    Génère un vecteur sparse via hashing trick avec:
    - Character n-grams
    - Word unigrams
    - Word bigrams/trigrams
    - Position features
    - BPE tokens (optionnel)
    """

    def __init__(
        self,
        n_features: int,
        weights: Dict[str, float],
        char_ngram_min: int,
        char_ngram_max: int,
        max_words: int,
        bpe_tokenizer=None
    ):
        self.n_features = n_features
        self.weights = weights
        self.char_ngram_min = char_ngram_min
        self.char_ngram_max = char_ngram_max
        self.max_words = max_words
        self.bpe_tokenizer = bpe_tokenizer
        self.collision_tracker = CollisionTracker(n_features)

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extrait les features d'un texte.

        Returns:
            Vecteur numpy de taille n_features
        """
        features = np.zeros(self.n_features, dtype=np.float32)
        words = tokenize_words(text, max_words=self.max_words)

        # 1. Character n-grams
        w_char = self.weights.get('w_char', 0.0)
        if w_char > 0:
            for word in words:
                padded = f"<{word}>"
                for n in range(self.char_ngram_min, self.char_ngram_max + 1):
                    for i in range(len(padded) - n + 1):
                        ngram = padded[i:i+n]
                        token = f"C_{ngram}"
                        self._add_token(token, w_char, features)

        # 2. Word unigrams
        w_word = self.weights.get('w_word', 0.0)
        if w_word > 0:
            for word in words:
                token = f"W_{word}"
                self._add_token(token, w_word, features)

        # 3. Word bigrams
        w_bigram = self.weights.get('w_bigram', 0.0)
        if w_bigram > 0 and len(words) > 1:
            for i in range(len(words) - 1):
                token = f"B_{words[i]}_{words[i+1]}"
                self._add_token(token, w_bigram, features)

        # 4. Word trigrams
        w_trigram = self.weights.get('w_trigram', 0.0)
        if w_trigram > 0 and len(words) > 2:
            for i in range(len(words) - 2):
                token = f"T_{words[i]}_{words[i+1]}_{words[i+2]}"
                self._add_token(token, w_trigram, features)

        # 5. Position features
        w_position = self.weights.get('w_position', 0.0)
        if w_position > 0 and len(words) > 0:
            # Premier mot
            token = f"POS_START_{words[0]}"
            self._add_token(token, w_position, features)
            # Dernier mot
            token = f"POS_END_{words[-1]}"
            self._add_token(token, w_position, features)

        # 6. BPE tokens (optionnel)
        w_bpe = self.weights.get('w_bpe', 0.0)
        if self.bpe_tokenizer and w_bpe > 0:
            for word in words:
                bpe_tokens = self.bpe_tokenizer.tokenize(word)
                for bpe_tok in bpe_tokens:
                    self._add_token(bpe_tok, w_bpe, features)

        return features

    def _add_token(self, token: str, weight: float, features: np.ndarray):
        """
        Ajoute un token au vecteur de features via hashing trick.
        Compatible sklearn alternate_sign=True
        """
        idx, sign = hash_to_index_and_sign(token, self.n_features)
        features[idx] += sign * weight

        # Track pour analyse
        self.collision_tracker.track(token, idx)

    def _get_words_for_benchmark(self, text: str) -> List[str]:
        """
        Internal helper for benchmarking to reuse the exact word split.
        """
        return tokenize_words(text, max_words=self.max_words)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforme une liste de textes en matrice de features.

        Args:
            texts: Liste de textes

        Returns:
            Matrice (n_samples, n_features)
        """
        n_samples = len(texts)
        X = np.zeros((n_samples, self.n_features), dtype=np.float32)

        for i, text in enumerate(texts):
            X[i] = self.extract_features(text)

        return X

    def get_feature_stats(self) -> dict:
        """Retourne des statistiques sur les features extraites"""
        return {
            'n_features': self.n_features,
            'collision_report': self.collision_tracker.get_report()
        }
