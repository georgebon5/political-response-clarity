"""TN2.4 bonus — encoder ensemble that reached 2nd place on the Kaggle leaderboard.

Composition (softmax-mean averaged, matching the TN2.4 notebook):

- Base models trained once on the full train set at ``max_length=448``:
  ``bert-base-uncased``, ``roberta-base``, ``microsoft/deberta-v3-base``.
- Large models with Stratified K-Fold CV at ``max_length=512``:
  ``roberta-large`` × 4 folds (fp16, with (q,a)+(a,q) test-time augmentation);
  ``microsoft/deberta-v3-large`` × 2 folds (fp32 to avoid the fp16 class-weight bug).
- Optional pseudo-labelling round using a single DeBERTa-v3-large retrained on
  the train split extended with high-confidence (>0.65) test predictions.

Everything is implemented with the Hugging Face ``Trainer`` to keep the
original DRY structure of the notebook. Seeds are offset per model so each
fold uses a different RNG stream.
"""

from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import ID2LABEL, LABEL2ID, SEED, set_seed


@dataclass
class EnsembleConfig:
    max_len_base: int = 448
    max_len_large: int = 512
    epochs_base: int = 5
    epochs_large: int = 4
    n_folds_roberta: int = 4
    n_folds_deberta: int = 2
    pseudo_threshold: float = 0.65
    pseudo_min_samples: int = 15


def _is_deberta_v3(model_name: str) -> bool:
    n = model_name.lower()
    return "deberta-v3" in n or "deberta-v2" in n


def _class_weights(df: pd.DataFrame, num_labels: int = 3) -> torch.Tensor:
    counts = Counter(df["label_id"].values)
    return torch.tensor(
        [len(df) / (num_labels * counts.get(i, 1)) for i in range(num_labels)],
        dtype=torch.float32,
    )


class _WeightedLossTrainer:
    """Factory for a ``transformers.Trainer`` subclass with per-run class weights.

    We build the subclass lazily because ``transformers`` is heavy to import at
    module load.
    """

    _cls: type | None = None

    @classmethod
    def get(cls):
        if cls._cls is not None:
            return cls._cls

        from transformers import Trainer

        class WeightedLossTrainerCW(Trainer):
            _cw = None

            def set_cw(self, cw):
                self._cw = cw

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=self._cw.to(logits.device)
                )
                return (loss, outputs) if return_outputs else loss

        cls._cls = WeightedLossTrainerCW
        return WeightedLossTrainerCW


def _tokenize_test(test_df: pd.DataFrame, tok, max_len: int, swap: bool = False):
    from datasets import Dataset

    def fn(batch):
        if swap:
            return tok(batch["answer"], batch["question"], truncation=True, max_length=max_len, padding=False)
        return tok(batch["question"], batch["answer"], truncation=True, max_length=max_len, padding=False)

    return Dataset.from_dict(
        {
            "question": test_df["question"].tolist(),
            "answer": test_df["interview_answer"].tolist(),
        }
    ).map(fn, batched=True, remove_columns=["question", "answer"])


def train_one_and_predict(
    model_name: str,
    train_subset: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_len: int,
    seed_offset: int,
    tag: str,
    use_tta: bool = False,
) -> np.ndarray:
    """Fine-tune one encoder and return softmax probabilities on ``test_df``."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
    )

    is_deberta = _is_deberta_v3(model_name)
    use_fp16 = torch.cuda.is_available() and not is_deberta

    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch):
        return tok(batch["question"], batch["answer"], truncation=True, max_length=max_len, padding=False)

    train_ds = Dataset.from_dict(
        {
            "question": train_subset["question"].tolist(),
            "answer": train_subset["interview_answer"].tolist(),
            "labels": train_subset["label_id"].tolist(),
        }
    ).map(tok_fn, batched=True, remove_columns=["question", "answer"])

    cw_sub = _class_weights(train_subset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    effective_bs = batch_size * grad_accum
    n_steps = max(1, n_epochs * (len(train_ds) // effective_bs + 1))
    args = TrainingArguments(
        output_dir=f"./runs/{model_name.replace('/', '_')}_s{seed_offset}",
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        warmup_steps=int(0.1 * n_steps),
        weight_decay=0.01,
        fp16=use_fp16,
        seed=SEED + seed_offset,
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        disable_tqdm=False,
    )

    TrainerCW = _WeightedLossTrainer.get()
    collator = DataCollatorWithPadding(tok)
    try:
        trainer = TrainerCW(
            model=model, args=args, train_dataset=train_ds,
            processing_class=tok, data_collator=collator,
        )
    except TypeError:  # older transformers
        trainer = TrainerCW(
            model=model, args=args, train_dataset=train_ds,
            tokenizer=tok, data_collator=collator,
        )
    trainer.set_cw(cw_sub)
    trainer.train()

    test_ds_qa = _tokenize_test(test_df, tok, max_len, swap=False)
    probs_qa = torch.softmax(
        torch.tensor(trainer.predict(test_ds_qa).predictions, dtype=torch.float32), dim=-1
    ).numpy()
    probs = probs_qa
    if use_tta:
        test_ds_aq = _tokenize_test(test_df, tok, max_len, swap=True)
        probs_aq = torch.softmax(
            torch.tensor(trainer.predict(test_ds_aq).predictions, dtype=torch.float32), dim=-1
        ).numpy()
        probs = (probs_qa + probs_aq) / 2.0

    del model, trainer, tok, train_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


# ─── Public pipeline ──────────────────────────────────────────────────────────


@dataclass
class EnsembleRun:
    probs_per_model: dict[str, np.ndarray] = field(default_factory=dict)
    final_probs: np.ndarray | None = None
    final_labels: list[str] | None = None


def run_encoder_ensemble(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: EnsembleConfig | None = None
) -> EnsembleRun:
    """Full base + large K-fold ensemble (no pseudo-labelling)."""
    from sklearn.model_selection import StratifiedKFold

    cfg = cfg or EnsembleConfig()
    set_seed(SEED)
    train_df = train_df.copy()
    train_df["label_id"] = train_df["clarity_label"].map(LABEL2ID)

    run = EnsembleRun()

    base_specs = [
        ("bert-base-uncased",            0,  16, 1, 2e-5, "BERT-base",         False),
        ("roberta-base",                 0,  16, 1, 2e-5, "RoBERTa-base",      False),
        ("microsoft/deberta-v3-base",    0,  16, 1, 2e-5, "DeBERTa-v3-base",   False),
    ]
    for model_name, seed_off, bs, ga, lr, tag, tta in base_specs:
        run.probs_per_model[tag] = train_one_and_predict(
            model_name, train_df, test_df,
            n_epochs=cfg.epochs_base, batch_size=bs, grad_accum=ga, lr=lr,
            max_len=cfg.max_len_base, seed_offset=seed_off, tag=tag, use_tta=tta,
        )

    y = train_df["label_id"].values
    skf_r = StratifiedKFold(n_splits=cfg.n_folds_roberta, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, _) in enumerate(skf_r.split(np.zeros(len(y)), y)):
        tag = f"RoBERTa-large [fold {fold_idx+1}/{cfg.n_folds_roberta}]"
        fold = train_df.iloc[train_idx].reset_index(drop=True)
        run.probs_per_model[tag] = train_one_and_predict(
            "roberta-large", fold, test_df,
            n_epochs=cfg.epochs_large, batch_size=8, grad_accum=2, lr=1e-5,
            max_len=cfg.max_len_large, seed_offset=100 + fold_idx, tag=tag, use_tta=True,
        )

    skf_d = StratifiedKFold(n_splits=cfg.n_folds_deberta, shuffle=True, random_state=SEED + 1)
    for fold_idx, (train_idx, _) in enumerate(skf_d.split(np.zeros(len(y)), y)):
        tag = f"DeBERTa-v3-large [fold {fold_idx+1}/{cfg.n_folds_deberta}]"
        fold = train_df.iloc[train_idx].reset_index(drop=True)
        run.probs_per_model[tag] = train_one_and_predict(
            "microsoft/deberta-v3-large", fold, test_df,
            n_epochs=cfg.epochs_large, batch_size=8, grad_accum=2, lr=1e-5,
            max_len=cfg.max_len_large, seed_offset=150 + fold_idx, tag=tag, use_tta=False,
        )

    run.final_probs = np.mean(list(run.probs_per_model.values()), axis=0)
    run.final_labels = [ID2LABEL[i] for i in run.final_probs.argmax(axis=1)]
    return run


def pseudo_label_round(
    run: EnsembleRun,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: EnsembleConfig | None = None,
) -> EnsembleRun:
    """One pseudo-labelling round with ``DeBERTa-v3-large``."""
    cfg = cfg or EnsembleConfig()
    if run.final_probs is None:
        raise ValueError("Run has no ensemble probabilities yet.")

    max_probs = run.final_probs.max(axis=1)
    high_conf = np.where(max_probs > cfg.pseudo_threshold)[0]
    if len(high_conf) < cfg.pseudo_min_samples:
        return run

    train_df = train_df.copy()
    train_df["label_id"] = train_df["clarity_label"].map(LABEL2ID)
    pseudo_ids = run.final_probs.argmax(axis=1)[high_conf].tolist()
    ext_df = pd.DataFrame(
        {
            "question": (
                train_df["question"].tolist()
                + [test_df.iloc[int(i)]["question"] for i in high_conf]
            ),
            "interview_answer": (
                train_df["interview_answer"].tolist()
                + [test_df.iloc[int(i)]["interview_answer"] for i in high_conf]
            ),
            "label_id": train_df["label_id"].tolist() + pseudo_ids,
        }
    )
    ext_df["clarity_label"] = ext_df["label_id"].map(ID2LABEL)

    tag = f"DeBERTa-v3-large [pseudo t={cfg.pseudo_threshold}]"
    run.probs_per_model[tag] = train_one_and_predict(
        "microsoft/deberta-v3-large", ext_df, test_df,
        n_epochs=cfg.epochs_large, batch_size=8, grad_accum=2, lr=1e-5,
        max_len=cfg.max_len_large, seed_offset=200, tag=tag, use_tta=False,
    )
    run.final_probs = np.mean(list(run.probs_per_model.values()), axis=0)
    run.final_labels = [ID2LABEL[i] for i in run.final_probs.argmax(axis=1)]
    return run
