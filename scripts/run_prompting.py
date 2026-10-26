#!/usr/bin/env python3
"""Stage 3 — zero-shot / CoT / few-shot prompting with Qwen2.5 instruct.

Examples:
    python scripts/run_prompting.py --model 7b --strategy few-shot
    python scripts/run_prompting.py --model 3b --strategy zero-shot
    python scripts/run_prompting.py --model 3b --strategy cot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.data import (  # noqa: E402
    few_shot_pool,
    load_qevasion,
    set_seed,
    stratified_val_fixed_per_class,
)
from clarity.evaluate import (  # noqa: E402
    append_metrics_row,
    classification_text_report,
    metrics,
    save_confusion_matrix,
)
from clarity.prompting import MODEL_IDS, load_lm, run_strategy  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODEL_IDS), required=True)
    parser.add_argument("--strategy", choices=["zero-shot", "cot", "few-shot"], required=True)
    parser.add_argument("--val-per-class", type=int, default=100,
                        help="validation samples per class (300 total by default)")
    parser.add_argument("--few-per-class", type=int, default=2)
    parser.add_argument("--use-4bit", action="store_true", help="load model in 4-bit NF4")
    args = parser.parse_args()

    set_seed()
    train_df, _ = load_qevasion()

    few_df, few_idx = few_shot_pool(train_df, per_class=args.few_per_class)
    remaining = train_df.drop(index=list(few_idx)).reset_index(drop=True)
    val_df = stratified_val_fixed_per_class(remaining, per_class=args.val_per_class)

    lm = load_lm(MODEL_IDS[args.model], use_4bit=args.use_4bit)
    preds = run_strategy(
        lm, val_df, strategy=args.strategy,
        few_shot_df=few_df if args.strategy == "few-shot" else None,
        desc=f"{args.model}/{args.strategy}",
    )

    print(classification_text_report(val_df["clarity_label"].tolist(), preds))
    m = metrics(val_df["clarity_label"].tolist(), preds)
    tag = f"prompt:{args.model}/{args.strategy}"
    append_metrics_row(
        RESULTS_DIR / "metrics.csv",
        {
            "method": tag,
            "split": "val",
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            **{f"f1_{k}": v for k, v in m["f1_per_class"].items()},
        },
    )
    save_confusion_matrix(
        val_df["clarity_label"].tolist(), preds,
        RESULTS_DIR / f"confusion_matrix_prompt_{args.model}_{args.strategy}.png",
        title=tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
