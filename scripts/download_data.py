#!/usr/bin/env python3
"""Download the QEvasion dataset from Hugging Face and print basic stats.

Usage:
    python scripts/download_data.py

Dataset: https://huggingface.co/datasets/ailsntua/QEvasion
License : see the dataset card on Hugging Face.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clarity.data import load_qevasion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train_df, test_df = load_qevasion()
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")
    print("Train label distribution:")
    for label, n in Counter(train_df["clarity_label"]).most_common():
        print(f"  {label:<18s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
