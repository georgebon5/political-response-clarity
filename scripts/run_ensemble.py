#!/usr/bin/env python3
"""TN2.4 bonus — encoder ensemble (BERT, RoBERTa, DeBERTa-v3, RoBERTa-large,
DeBERTa-v3-large K-folds) with optional pseudo-labelling.

Example:
    python scripts/run_ensemble.py --out submission.csv
    python scripts/run_ensemble.py --with-pseudo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.data import load_qevasion, set_seed  # noqa: E402
from clarity.ensemble import EnsembleConfig, pseudo_label_round, run_encoder_ensemble  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="submission.csv", help="output CSV path")
    parser.add_argument("--with-pseudo", action="store_true",
                        help="run one pseudo-labelling round (threshold=0.65)")
    parser.add_argument("--folds-roberta", type=int, default=4)
    parser.add_argument("--folds-deberta", type=int, default=2)
    args = parser.parse_args()

    set_seed()
    train_df, test_df = load_qevasion()

    cfg = EnsembleConfig(n_folds_roberta=args.folds_roberta, n_folds_deberta=args.folds_deberta)
    run = run_encoder_ensemble(train_df, test_df, cfg)
    if args.with_pseudo:
        run = pseudo_label_round(run, train_df, test_df, cfg)

    assert run.final_labels is not None
    sub = pd.DataFrame({"Id": test_df["index"].tolist(), "Predicted": run.final_labels})
    sub.to_csv(args.out, index=False)
    print(f"Wrote {len(sub)} predictions → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
