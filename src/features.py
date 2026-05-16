"""Feature engineering helpers.

Two families of features are supported:

1. ``build_tfidf_vectorizer`` — a ready-to-use TF-IDF vectorizer with
   sensible defaults for essay-length text.
2. ``stylometric_features`` — simple hand-crafted features (sentence
   length, punctuation rates, type-token ratio, etc.) that the team can use
   as descriptive variables or as additional inputs to a classifier.
"""

from __future__ import annotations

import re
import string
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_SPLIT = re.compile(r"\b\w+\b")


def build_tfidf_vectorizer(
    *,
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Return a TfidfVectorizer with defaults tuned for essay text."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def _stylometric_row(text: str) -> dict[str, float]:
    words = _WORD_SPLIT.findall(text)
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    n_words = len(words) or 1
    n_sentences = len(sentences) or 1
    n_chars = len(text) or 1

    punct_count = sum(1 for c in text if c in string.punctuation)
    unique_words = {w.lower() for w in words}

    return {
        "n_chars": float(len(text)),
        "n_words": float(len(words)),
        "n_sentences": float(len(sentences)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_len_words": n_words / n_sentences,
        "type_token_ratio": len(unique_words) / n_words,
        "punct_rate": punct_count / n_chars,
        "comma_rate": text.count(",") / n_chars,
        "uppercase_rate": sum(1 for c in text if c.isupper()) / n_chars,
    }


def stylometric_features(texts: Iterable[str]) -> pd.DataFrame:
    """Compute a small table of hand-crafted stylometric features."""
    return pd.DataFrame([_stylometric_row(t) for t in texts])
