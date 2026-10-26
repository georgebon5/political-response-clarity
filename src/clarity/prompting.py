"""Stage 3 — prompting with Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct
(zero-shot / chain-of-thought / few-shot).

Faithful port of the TN2.3 notebook. All generation is greedy
(``do_sample=False``) so results are deterministic given the seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import VALID_LABELS, truncate_words

Strategy = Literal["zero-shot", "cot", "few-shot"]

MODEL_IDS: dict[str, str] = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

MAX_ANSWER_WORDS = 350
MAX_FEW_ANS_WORDS = 150
TOKENIZER_MAX_LEN = 2048

MAX_NEW_TOKENS: dict[Strategy, int] = {
    "zero-shot": 20,
    "cot": 300,
    "few-shot": 20,
}


# ─── Prompts ──────────────────────────────────────────────────────────────────

_LABEL_DEFS = (
    "Classify the politician's response into EXACTLY ONE of these three categories:\n"
    "- Clear Reply: The politician directly and specifically answers the question asked. "
    "The response addresses the core of the question without evasion.\n"
    "- Ambivalent: The politician's response is related to the topic but does NOT directly "
    "answer the specific question. The answer may be vague, general, or only partially relevant.\n"
    "- Clear Non-Reply: The politician clearly fails to address the question. "
    "This includes explicit refusals, complete topic changes, or answers entirely unrelated to what was asked.\n"
    "\n"
    'KEY DISTINCTION: "Ambivalent" = on-topic but indirect; '
    '"Clear Non-Reply" = clearly avoids or ignores the question entirely.\n'
)

_ROLE = (
    "You are an expert analyst of political communication. "
    "Your task is to classify the clarity of a politician's response to a journalist's question.\n\n"
)

SYSTEM_ZERO_SHOT = (
    _ROLE
    + _LABEL_DEFS
    + "Output ONLY the exact category name and nothing else: "
    "Clear Reply, Ambivalent, or Clear Non-Reply."
)

SYSTEM_COT = (
    _ROLE
    + _LABEL_DEFS
    + "Reason step by step, then end your response with exactly:\n"
    '"Final answer: [Clear Reply / Ambivalent / Clear Non-Reply]"'
)

SYSTEM_FEW_SHOT = SYSTEM_ZERO_SHOT  # identical system; examples go in the user turn


def build_zero_shot_messages(question: str, answer: str) -> list[dict]:
    ans = truncate_words(answer, MAX_ANSWER_WORDS)
    return [
        {"role": "system", "content": SYSTEM_ZERO_SHOT},
        {
            "role": "user",
            "content": f"Question: {question.strip()}\n\nAnswer: {ans.strip()}\n\nCategory:",
        },
    ]


def build_cot_messages(question: str, answer: str) -> list[dict]:
    ans = truncate_words(answer, MAX_ANSWER_WORDS)
    user = (
        f"Question: {question.strip()}\n\n"
        f"Answer: {ans.strip()}\n\n"
        "Analyze step by step:\n"
        "1. What specific information is the journalist asking for?\n"
        "2. Does the politician's response directly and specifically address that question?\n"
        "3. If not direct, is the response at least on-topic and partial (Ambivalent), "
        "or does it clearly avoid/ignore the question (Clear Non-Reply)?\n\n"
        "Provide your reasoning, then write: Final answer: [category]"
    )
    return [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": user},
    ]


def build_few_shot_messages(question: str, answer: str, examples_df: pd.DataFrame) -> list[dict]:
    block = ""
    for _, row in examples_df.iterrows():
        ex_ans = truncate_words(str(row["interview_answer"]), MAX_FEW_ANS_WORDS)
        block += (
            f"Question: {str(row['question']).strip()}\n"
            f"Answer: {ex_ans.strip()}\n"
            f"Category: {row['clarity_label']}\n\n"
        )
    ans = truncate_words(answer, MAX_ANSWER_WORDS)
    user = (
        "Here are example classifications:\n\n"
        + block
        + "Now classify the following:\n"
        + f"Question: {question.strip()}\n"
        + f"Answer: {ans.strip()}\n"
        + "Category:"
    )
    return [
        {"role": "system", "content": SYSTEM_FEW_SHOT},
        {"role": "user", "content": user},
    ]


# ─── Label extraction (verbatim from TN2.3, with substring-match ordering) ─────


def extract_label(generated_text: str, is_cot: bool = False) -> str:
    """Extract one of the three valid labels from a generation.

    ``Clear Non-Reply`` is checked BEFORE ``Clear Reply`` to avoid substring
    collisions. Falls back to ``Ambivalent`` (majority class).
    """
    text = generated_text.strip()
    if is_cot:
        m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    text_lower = text.lower()
    for label in VALID_LABELS:
        if label.lower() == text_lower:
            return label
    if "clear non-reply" in text_lower or "clear non reply" in text_lower or "non-reply" in text_lower:
        return "Clear Non-Reply"
    if "clear reply" in text_lower or "clearly replies" in text_lower or "clear answer" in text_lower:
        return "Clear Reply"
    if "ambival" in text_lower or "ambiguous" in text_lower or "partially" in text_lower:
        return "Ambivalent"
    return "Ambivalent"


# ─── Model load + inference ────────────────────────────────────────────────────


@dataclass
class LoadedLM:
    model: object
    tokenizer: object


def load_lm(model_name: str, use_4bit: bool = False, device_map: str = "auto") -> LoadedLM:
    """Load a Qwen2.5 instruct causal LM. ``use_4bit=True`` for 7B on small GPUs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg, device_map=device_map, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
    model.eval()
    return LoadedLM(model=model, tokenizer=tokenizer)


@torch.no_grad()
def run_strategy(
    lm: LoadedLM,
    df: pd.DataFrame,
    strategy: Strategy,
    few_shot_df: pd.DataFrame | None = None,
    desc: str | None = None,
) -> list[str]:
    """Run one prompting strategy over ``df``; return a list of predicted labels."""
    is_cot = strategy == "cot"
    max_new = MAX_NEW_TOKENS[strategy]
    builder: Callable[[str, str], list[dict]]

    if strategy == "zero-shot":
        builder = build_zero_shot_messages
    elif strategy == "cot":
        builder = build_cot_messages
    elif strategy == "few-shot":
        if few_shot_df is None:
            raise ValueError("few_shot_df is required for the few-shot strategy")
        builder = lambda q, a: build_few_shot_messages(q, a, few_shot_df)  # noqa: E731
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    predictions: list[str] = []
    device = lm.model.device
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc or strategy):
        messages = builder(str(row["question"]).strip(), str(row["interview_answer"]).strip())
        prompt = lm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = lm.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LEN
        ).to(device)
        out_ids = lm.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=lm.tokenizer.eos_token_id,
        )
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        generated = lm.tokenizer.decode(new_ids, skip_special_tokens=True)
        predictions.append(extract_label(generated, is_cot=is_cot))
    return predictions


def majority_vote(predictions_list: list[list[str]], tie_break: str = "Ambivalent") -> list[str]:
    """Element-wise majority vote across several prediction lists."""
    from collections import Counter

    n = len(predictions_list[0])
    out: list[str] = []
    for i in range(n):
        votes = Counter(p[i] for p in predictions_list)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            out.append(tie_break)
        else:
            out.append(top[0][0])
    return out
