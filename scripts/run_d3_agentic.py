#!/usr/bin/env python3
"""Stage 4 — D3-Agentic pipeline on Qwen2.5-0.5B-Instruct.

Example:
    python scripts/run_d3_agentic.py
    python scripts/run_d3_agentic.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.agents import D3Orchestrator, QWEN_SMALL, load_qwen_small  # noqa: E402
from clarity.data import load_qevasion, set_seed, stratified_val_fixed_per_class  # noqa: E402
from clarity.evaluate import (  # noqa: E402
    append_metrics_row,
    classification_text_report,
    metrics,
    save_confusion_matrix,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=QWEN_SMALL,
                        help="Any Qwen2.5-Instruct model id (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--save-logs", action="store_true",
                        help="dump every agent's intermediate JSON to results/d3_logs.jsonl")
    args = parser.parse_args()

    set_seed()
    train_df, _ = load_qevasion()
    val_df = stratified_val_fixed_per_class(train_df, per_class=args.val_per_class)

    model, tokenizer = load_qwen_small(args.model)
    orch = D3Orchestrator(model, tokenizer)
    result = orch.run_dataset(val_df, desc="D3-Agentic (val)")

    print(classification_text_report(val_df["clarity_label"].tolist(), result.predictions))
    m = metrics(val_df["clarity_label"].tolist(), result.predictions)
    append_metrics_row(
        RESULTS_DIR / "metrics.csv",
        {
            "method": "d3-agentic:qwen2.5-0.5b",
            "split": "val",
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            **{f"f1_{k}": v for k, v in m["f1_per_class"].items()},
        },
    )
    save_confusion_matrix(
        val_df["clarity_label"].tolist(), result.predictions,
        RESULTS_DIR / "confusion_matrix_d3_agentic.png",
        title="D3-Agentic (Qwen2.5-0.5B) — val",
    )
    if args.save_logs:
        out = RESULTS_DIR / "d3_logs.jsonl"
        with out.open("w") as f:
            for row in result.logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved per-sample agent logs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
