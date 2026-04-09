"""
Shared text normalization utilities.

Must stay consistent with the C++ implementation and the original trainer.
"""

from __future__ import annotations

import string
import unicodedata
from typing import List, Optional

# Replace ASCII punctuation with spaces (same behavior as original script)
_PUNCT_TRANS = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_text(text: str) -> str:
    """
    Normalize text to match the original pipeline:
    - NFD unicode normalization
    - Remove accents
    - Lowercase
    - Replace ASCII punctuation with spaces
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower().translate(_PUNCT_TRANS)
    return " ".join(text.split())


def tokenize_words(text: str, max_words: Optional[int] = None) -> List[str]:
    """
    Normalize and split into words, optionally truncating.
    """
    words = normalize_text(text).split()
    if max_words is not None:
        return words[:max_words]
    return words
