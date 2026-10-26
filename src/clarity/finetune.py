"""Stage 2 — fine-tune bert-base-uncased, distilbert-base-uncased, or
microsoft/deberta-v3-base with a custom PyTorch loop (no HF Trainer).

Faithful port of the TN2.2 notebooks. Shared choices:
- Weighted cross-entropy (inverse-frequency class weights on the train split).
- AdamW with weight decay 0.01 on non-bias / non-LayerNorm parameters, 0 on bias
  and LayerNorm.
- Linear warmup over 10% of total steps, then linear decay to zero.
- Gradient clipping at ``max_norm=1.0``.
- Early stopping on validation macro F1 with patience 2; best weights restored.
- DeBERTa-v3 is explicitly cast to float32 to avoid the fp16 class-weight bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .data import ID2LABEL, LABEL2ID, SEED, set_seed


# ─── Config presets per model (final submitted configs from TN2.2) ─────────────


@dataclass
class FinetuneConfig:
    model_name: str
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 4
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    patience: int = 2
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2


PRESETS: dict[str, FinetuneConfig] = {
    "bert": FinetuneConfig(
        model_name="bert-base-uncased", max_length=256, batch_size=16, epochs=4, lr=2e-5
    ),
    "distilbert": FinetuneConfig(
        model_name="distilbert-base-uncased", max_length=256, batch_size=32, epochs=5, lr=3e-5
    ),
    "deberta": FinetuneConfig(
        model_name="microsoft/deberta-v3-base", max_length=256, batch_size=8, epochs=5, lr=2e-5
    ),
}


# ─── Dataset ───────────────────────────────────────────────────────────────────


class ClarityDataset(Dataset):
    """Tokenises (question, answer) pairs as a sentence pair."""

    def __init__(self, questions, answers, labels, tokenizer, max_length: int):
        self.q = list(questions)
        self.a = list(answers)
        self.y = list(labels) if labels is not None else None
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tok(
            self.q[idx],
            self.a[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[idx], dtype=torch.long)
        return item


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
    labels: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    y = df["label"].tolist() if labels else None
    ds = ClarityDataset(df["question"], df["interview_answer"], y, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ─── Optimizer / scheduler ────────────────────────────────────────────────────


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(grouped, lr=lr)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup

    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )


def class_weights_from(df: pd.DataFrame, num_labels: int = 3, device=None) -> torch.Tensor:
    counts = df["label"].value_counts().sort_index().values
    w = torch.tensor(len(df) / (num_labels * counts), dtype=torch.float32)
    return w.to(device) if device is not None else w


# ─── Train / evaluate ─────────────────────────────────────────────────────────


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device)
    return out


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip: float) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    model.train()
    total_loss, preds, labels = 0.0, [], []
    for raw in loader:
        batch = _move_batch(raw, device)
        y = batch.pop("labels")
        optimizer.zero_grad()
        out = model(**batch)
        loss = criterion(out.logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / len(loader), f1_score(labels, preds, average="macro", zero_division=0)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device) -> tuple[float, float, list[int], list[int]]:
    from sklearn.metrics import f1_score

    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for raw in loader:
        batch = _move_batch(raw, device)
        y = batch.pop("labels")
        out = model(**batch)
        loss = criterion(out.logits, y)
        total_loss += loss.item()
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return (
        total_loss / len(loader),
        f1_score(labels, preds, average="macro", zero_division=0),
        preds,
        labels,
    )


# ─── Public API ───────────────────────────────────────────────────────────────


def train(
    cfg: FinetuneConfig,
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    device: torch.device | None = None,
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one model end-to-end. Returns history + best val F1 + the fitted model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(SEED)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader = make_loader(df_tr, tokenizer, cfg.max_length, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = make_loader(df_val, tokenizer, cfg.max_length, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    # TN2.2: DeBERTa-v3 must stay in float32 or weighted CE throws
    #   "expected scalar type Half but found Float".
    if "deberta" in cfg.model_name.lower():
        model = model.float()
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_from(df_tr, device=device))
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    scheduler = build_scheduler(optimizer, len(train_loader) * cfg.epochs, cfg.warmup_ratio)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg.grad_clip)
        va_loss, va_f1, _, _ = evaluate_loader(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)
        improved = " [best]" if va_f1 > best_f1 else ""
        log_fn(
            f"Epoch {epoch}/{cfg.epochs}  Train L={tr_loss:.4f} F1={tr_f1:.4f}  |  "
            f"Val L={va_loss:.4f} F1={va_f1:.4f}{improved}"
        )
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log_fn(f"Early stop at epoch {epoch} (patience={cfg.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_val_f1": best_f1,
    }


@torch.no_grad()
def predict(model, tokenizer, df: pd.DataFrame, max_length: int, batch_size: int, device=None) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = make_loader(df, tokenizer, max_length, batch_size, labels=False, num_workers=0)
    all_preds: list[int] = []
    for raw in loader:
        batch = _move_batch(raw, device)
        out = model(**batch)
        all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())
    return np.array(all_preds)
