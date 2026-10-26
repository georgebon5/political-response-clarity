#!/usr/bin/env python3
"""Stage 2 — fine-tune one of the three sentence-pair encoders.

Examples:
    python scripts/run_finetune.py --model bert
    python scripts/run_finetune.py --model distilbert
    python scripts/run_finetune.py --model deberta --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.data import load_qevasion, stratified_val_split  # noqa: E402
from clarity.evaluate import (  # noqa: E402
    append_metrics_row,
    classification_text_report,
    metrics,
    save_confusion_matrix,
)
from clarity.finetune import PRESETS, evaluate_loader, make_loader, train  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(PRESETS), required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override preset epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    cfg = PRESETS[args.model]
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_length is not None:
        cfg.max_length = args.max_length

    train_df, _ = load_qevasion()
    df_tr, df_val = stratified_val_split(train_df)

    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = train(cfg, df_tr, df_val, device=device)
    model = result["model"]
    tokenizer = result["tokenizer"]

    val_loader = make_loader(df_val, tokenizer, cfg.max_length, cfg.batch_size, labels=True, num_workers=0)
    _, _, preds, labels = evaluate_loader(model, val_loader, nn.CrossEntropyLoss(), device)

    print(classification_text_report(labels, preds))
    m = metrics(labels, preds)
    append_metrics_row(
        RESULTS_DIR / "metrics.csv",
        {
            "method": f"finetune:{args.model}",
            "split": "val",
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            **{f"f1_{k}": v for k, v in m["f1_per_class"].items()},
        },
    )
    save_confusion_matrix(
        labels, preds,
        RESULTS_DIR / f"confusion_matrix_{args.model}.png",
        title=f"{args.model} — val",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
