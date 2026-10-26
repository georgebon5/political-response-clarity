"""Shared metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import ID2LABEL, LABEL2ID, VALID_LABELS


def _to_ids(labels: Iterable) -> list[int]:
    out = []
    for l in labels:
        if isinstance(l, (int, np.integer)):
            out.append(int(l))
        else:
            out.append(LABEL2ID.get(l, LABEL2ID["Ambivalent"]))
    return out


def metrics(y_true, y_pred) -> dict:
    """Compute accuracy, macro precision/recall/F1, and per-class F1."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true_ids = _to_ids(y_true)
    y_pred_ids = _to_ids(y_pred)

    p, r, f, _ = precision_recall_fscore_support(
        y_true_ids, y_pred_ids, average="macro", zero_division=0, labels=list(range(len(VALID_LABELS)))
    )
    per_class_f1 = f1_score(
        y_true_ids, y_pred_ids, average=None, zero_division=0, labels=list(range(len(VALID_LABELS)))
    )
    return {
        "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "f1_per_class": {VALID_LABELS[i]: float(per_class_f1[i]) for i in range(len(VALID_LABELS))},
    }


def classification_text_report(y_true, y_pred) -> str:
    from sklearn.metrics import classification_report

    return classification_report(
        _to_ids(y_true),
        _to_ids(y_pred),
        target_names=VALID_LABELS,
        digits=4,
        zero_division=0,
    )


def save_confusion_matrix(y_true, y_pred, out_path: str | Path, title: str = "Confusion matrix") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(_to_ids(y_true), _to_ids(y_pred), labels=list(range(len(VALID_LABELS))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=VALID_LABELS,
        yticklabels=VALID_LABELS,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_metrics_row(csv_path: str | Path, row: dict) -> None:
    """Append one row to ``results/metrics.csv``, creating the header on first write."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
