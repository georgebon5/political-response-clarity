"""Dataset loading, stratified split, and (question, answer) input construction.

Dataset: ``ailsntua/QEvasion`` (SemEval 2026 Task 6, CLARITY).
Labels: ``Clear Reply``, ``Ambivalent``, ``Clear Non-Reply``.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

VALID_LABELS: list[str] = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(VALID_LABELS)}
ID2LABEL: dict[int, str] = {i: l for i, l in enumerate(VALID_LABELS)}

SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_qevasion() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with columns ``question``, ``interview_answer``,
    ``clarity_label`` (train only), ``index`` (test only).
    """
    from datasets import load_dataset

    ds = load_dataset("ailsntua/QEvasion")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    return _normalise(train_df), _normalise(test_df)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("question", "interview_answer"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "clarity_label" in df.columns:
        df["label"] = df["clarity_label"].map(LABEL2ID)
    return df


def stratified_val_split(train_df: pd.DataFrame, val_size: float = 0.15, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicates the TN2.2 split: 85/15 stratified by label, seed=42."""
    from sklearn.model_selection import train_test_split

    df_tr, df_val = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"], random_state=seed
    )
    return df_tr.reset_index(drop=True), df_val.reset_index(drop=True)


def stratified_val_fixed_per_class(
    train_df: pd.DataFrame, per_class: int = 100, seed: int = SEED
) -> pd.DataFrame:
    """Replicates the TN2.3 / TN2.4 validation set: 100 samples per class."""
    parts = []
    for label in VALID_LABELS:
        sub = train_df[train_df["clarity_label"] == label]
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def few_shot_pool(
    train_df: pd.DataFrame, per_class: int = 2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.Index]:
    """Pick ``per_class`` few-shot examples per class; return (pool, excluded_index).

    The excluded index should be removed from the remaining pool before building a
    validation set, to avoid leakage.
    """
    parts = []
    for label in VALID_LABELS:
        sub = train_df[train_df["clarity_label"] == label]
        parts.append(sub.sample(n=per_class, random_state=seed))
    pool = pd.concat(parts)
    return pool.reset_index(drop=True), pool.index


# ─── Input construction ────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """TN2.1 preprocessing: strip URLs + non-ASCII, collapse whitespace, lowercase."""
    if not isinstance(text, str):
        text = str(text)
    text = _URL_RE.sub("", text)
    text = _NON_ASCII_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.lower()


def concat_sep(questions: Iterable[str], answers: Iterable[str], sep: str = "[SEP]") -> list[str]:
    """TN2.1 input: ``question [SEP] answer`` after cleaning. The separator is a
    literal string here (not a tokenizer special token) — the TF-IDF vocabulary
    treats it as an ordinary word, and the Word2Vec pipeline maps it to zero.
    """
    return [f"{clean_text(q)} {sep} {clean_text(a)}" for q, a in zip(questions, answers)]


def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"
