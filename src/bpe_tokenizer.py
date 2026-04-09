"""
BPE (Byte Pair Encoding) léger pour tokenization sub-word.
Compatible C++ pour embarqué.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass

from text_normalizer import tokenize_words


@dataclass
class BpeMerge:
    """Un merge BPE"""
    pair: Tuple[str, str]
    result: str
    frequency: int


class BpeTokenizer:
    """
    Tokenizer BPE simplifié pour embarqué.

    Limites volontaires pour embarqué:
    - Vocabulaire limité (300-500 tokens)
    - Pas de tokens spéciaux complexes
    - Export direct en C++
    """

    def __init__(
        self,
        vocab_size: int = 300,
        min_freq: int = 10,
        unknown_token: str = "<UNK>",
        max_words: Optional[int] = 25
    ):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.unknown_token = unknown_token
        self.max_words = max_words

        self.merges: List[BpeMerge] = []
        self.vocab: Set[str] = set()
        self.token_to_id: Dict[str, int] = {}

    def fit(self, texts: List[str]) -> 'BpeTokenizer':
        """
        Apprend le vocabulaire BPE à partir des textes.
        """
        # Initialiser avec caractères
        word_freqs = self._count_words(texts)

        # Convertir mots en séquences de caractères
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)

        # Initialiser vocabulaire avec caractères uniques
        self.vocab = set()
        for word in splits:
            for char in splits[word]:
                self.vocab.add(char)

        print(f"Initial vocab size: {len(self.vocab)} characters")

        # BPE merges
        while len(self.vocab) < self.vocab_size:
            # Compter paires
            pair_freqs = self._count_pairs(splits, word_freqs)

            if not pair_freqs:
                break

            # Meilleure paire
            best_pair = max(pair_freqs.keys(), key=lambda p: pair_freqs[p])
            best_freq = pair_freqs[best_pair]

            if best_freq < self.min_freq:
                break

            # Merge
            merged = best_pair[0] + best_pair[1]
            self.merges.append(BpeMerge(
                pair=best_pair,
                result=merged,
                frequency=best_freq
            ))
            self.vocab.add(merged)

            # Appliquer merge
            splits = self._apply_merge(splits, best_pair)

            if len(self.merges) % 50 == 0:
                print(f"  Merges: {len(self.merges)}, Vocab: {len(self.vocab)}")

        # Construire mapping
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.token_to_id[self.unknown_token] = len(self.token_to_id)

        print(f"Final vocab size: {len(self.vocab)} tokens, {len(self.merges)} merges")

        return self

    def tokenize(self, word: str) -> List[str]:
        """
        Tokenize un mot en sous-mots BPE.
        """
        if not word:
            return []

        # Commencer avec caractères
        tokens = list(word)

        # Appliquer merges dans l'ordre
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge.pair[0] and tokens[i + 1] == merge.pair[1]:
                    tokens = tokens[:i] + [merge.result] + tokens[i + 2:]
                else:
                    i += 1

        # Préfixer pour distinguer des autres tokens
        return [f"BPE_{t}" for t in tokens]

    def _count_words(self, texts: List[str]) -> Dict[str, int]:
        """Compte les mots dans les textes"""
        word_freq = Counter()
        for text in texts:
            words = tokenize_words(text, max_words=self.max_words)
            word_freq.update(words)
        return dict(word_freq)

    def _count_pairs(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """Compte les paires adjacentes"""
        pair_freqs = defaultdict(int)

        for word, split in splits.items():
            freq = word_freqs[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return dict(pair_freqs)

    def _apply_merge(
        self,
        splits: Dict[str, List[str]],
        pair: Tuple[str, str]
    ) -> Dict[str, List[str]]:
        """Applique un merge à tous les mots"""
        new_splits = {}

        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split

        return new_splits

    def export_cpp(self, filepath: str):
        """
        Exporte les règles BPE pour C++.
        """
        with open(filepath, 'w') as f:
            f.write("// BPE Tokenizer - Auto-generated\n")
            f.write("#ifndef BPE_PATTERNS_H\n#define BPE_PATTERNS_H\n\n")
            f.write("#include <Arduino.h>\n\n")

            f.write(f"const int BPE_NUM_MERGES = {len(self.merges)};\n\n")

            # Structure pour un merge
            f.write("struct BpeMerge {\n")
            f.write("    const char* first;\n")
            f.write("    const char* second;\n")
            f.write("    const char* result;\n")
            f.write("};\n\n")

            # Table des merges
            f.write("const BpeMerge BPE_MERGES[] PROGMEM = {\n")
            for merge in self.merges:
                first = merge.pair[0].replace('\\', '\\\\').replace('"', '\\"')
                second = merge.pair[1].replace('\\', '\\\\').replace('"', '\\"')
                result = merge.result.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    {{"{first}", "{second}", "{result}"}},\n')
            f.write("};\n\n")

            f.write("#endif // BPE_PATTERNS_H\n")

        print(f"✅ BPE patterns exported: {filepath}")

    def get_vocab_stats(self) -> dict:
        """Retourne des statistiques sur le vocabulaire"""
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'avg_merge_freq': sum(m.frequency for m in self.merges) / max(1, len(self.merges)),
            'top_merges': [(m.result, m.frequency) for m in self.merges[:10]]
        }
