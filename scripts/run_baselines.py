#!/usr/bin/env python3
"""Stage 1 — TF-IDF or Word2Vec baseline + Logistic Regression.

Examples:
    python scripts/run_baselines.py --method tfidf
    python scripts/run_baselines.py --method tfidf --variant bigrams
    python scripts/run_baselines.py --method w2v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.baselines import (  # noqa: E402
    TfidfConfig,
    fit_tfidf,
    fit_word2vec,
    predict_tfidf,
    predict_word2vec,
)
from clarity.data import load_qevasion, stratified_val_split  # noqa: E402
from clarity.evaluate import (  # noqa: E402
    append_metrics_row,
    classification_text_report,
    metrics,
    save_confusion_matrix,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["tfidf", "w2v"], required=True)
    parser.add_argument("--variant", choices=["baseline", "bigrams"], default="bigrams",
                        help="TF-IDF variant (ignored for w2v)")
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    train_df, _ = load_qevasion()
    df_tr, df_val = stratified_val_split(train_df, val_size=args.val_size)

    if args.method == "tfidf":
        cfg = TfidfConfig(variant=args.variant)
        pipe = fit_tfidf(df_tr, cfg)
        preds = predict_tfidf(pipe, df_val)
        method_name = f"tfidf+lr({args.variant})"
    else:
        clf, kv = fit_word2vec(df_tr)
        preds = predict_word2vec(clf, kv, df_val)
        method_name = "word2vec+lr"

    print(classification_text_report(df_val["label"].values, preds))
    m = metrics(df_val["label"].values, preds)

    append_metrics_row(
        RESULTS_DIR / "metrics.csv",
        {
            "method": method_name,
            "split": "val",
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            **{f"f1_{k}": v for k, v in m["f1_per_class"].items()},
        },
    )
    save_confusion_matrix(
        df_val["label"].values, preds,
        RESULTS_DIR / f"confusion_matrix_{method_name.replace('+', '_').replace('(', '_').replace(')', '')}.png",
        title=method_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
