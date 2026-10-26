"""Classical baselines (Stage 1): TF-IDF + Logistic Regression and
averaged Word2Vec + Logistic Regression.

Follows the TN2.1 report:
- Lowercased, URL + non-ASCII stripped, whitespace collapsed.
- Input constructed as ``question [SEP] answer`` (literal separator).
- No stopword removal, no stemming.
- TF-IDF: a baseline variant and a bigrams-plus-higher-regularisation variant
  (the one the report says won validation). Exact TF-IDF hyperparameters are
  not documented in the report (only a figure); the defaults below are the
  closest match to the described behaviour.
- Word2Vec: pre-trained 300-dim GoogleNews, mean of in-vocab tokens, OOV → 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import SEED, concat_sep, set_seed


TfidfVariant = Literal["baseline", "bigrams"]


@dataclass
class TfidfConfig:
    variant: TfidfVariant = "bigrams"
    max_features: int | None = 50_000
    min_df: int = 2
    C: float = 1.0  # bigrams variant uses stronger regularisation (smaller C)

    @property
    def ngram_range(self) -> tuple[int, int]:
        return (1, 2) if self.variant == "bigrams" else (1, 1)


def tfidf_pipeline(cfg: TfidfConfig | None = None):
    """Return a fitted-on-fit scikit-learn Pipeline for TF-IDF + LR."""
    from sklearn.pipeline import Pipeline

    cfg = cfg or TfidfConfig()
    vec = TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        lowercase=False,  # already lowercased in concat_sep
        token_pattern=r"(?u)\b\w+\b",
    )
    clf = LogisticRegression(
        C=cfg.C,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def fit_tfidf(train_df: pd.DataFrame, cfg: TfidfConfig | None = None):
    set_seed(SEED)
    X = concat_sep(train_df["question"], train_df["interview_answer"])
    y = train_df["label"].values
    pipe = tfidf_pipeline(cfg)
    pipe.fit(X, y)
    return pipe


def predict_tfidf(pipe, df: pd.DataFrame) -> np.ndarray:
    X = concat_sep(df["question"], df["interview_answer"])
    return pipe.predict(X)


# ─── Word2Vec (averaged) ───────────────────────────────────────────────────────


def _load_w2v():
    """Load the pre-trained 300-dim GoogleNews Word2Vec vectors via gensim.

    First call downloads ~1.6GB via gensim.downloader to the local cache.
    """
    import gensim.downloader as api

    return api.load("word2vec-google-news-300")


def _vectorize_w2v(texts: list[str], kv, dim: int = 300) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        tokens = t.split()
        vecs = [kv[tok] for tok in tokens if tok in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def fit_word2vec(train_df: pd.DataFrame, C: float = 1.0):
    set_seed(SEED)
    kv = _load_w2v()
    X_train = _vectorize_w2v(concat_sep(train_df["question"], train_df["interview_answer"]), kv)
    y_train = train_df["label"].values
    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf, kv


def predict_word2vec(clf, kv, df: pd.DataFrame) -> np.ndarray:
    X = _vectorize_w2v(concat_sep(df["question"], df["interview_answer"]), kv)
    return clf.predict(X)
