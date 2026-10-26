"""Stage 4 — D3-Agentic prompting pipeline on ``Qwen/Qwen2.5-0.5B-Instruct``.

Four agents chained with structured JSON passing between them:

1. :class:`QuestionIntentAgent` — what does the question require for a direct answer?
2. :class:`AnswerContentAgent`  — what does the politician actually say?
3. :class:`GapEvasionAgent`     — compares (1) and (2), estimates gap severity.
4. :class:`DecisionAgent`       — deterministic rule mapping (gap_severity, engagement_level)
                                  to one of ``Clear Reply`` / ``Ambivalent`` / ``Clear Non-Reply``.

The project spec called for ``Qwen/Qwen3.5-0.8B``; that model does not exist on
Hugging Face, so ``Qwen/Qwen2.5-0.5B-Instruct`` is the closest available
substitute.

Faithful port of the TN2.4 notebook.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import VALID_LABELS, truncate_words

QWEN_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_ANSWER_WORDS = 300
TOKENIZER_MAX_LEN = 2048
MAX_AGENT_RETRIES = 2

AGENT_TOKENS: dict[str, int] = {
    "agent1_intent": 180,
    "agent2_content": 180,
    "agent3_gap": 200,
    "agent4_decide": 80,
}


# ─── LLM helper ───────────────────────────────────────────────────────────────


@torch.no_grad()
def llm_generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LEN
    ).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─── Base agent + JSON parsing ────────────────────────────────────────────────


class BaseAgent:
    SYSTEM_PROMPT: str = ""

    def __init__(self, model, tokenizer, max_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def _user_message(self, **kwargs) -> str:
        raise NotImplementedError

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """Three fallbacks: raw / strip code fences / regex-extract largest ``{...}``."""
        for cand in [raw, re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")]:
            try:
                return json.loads(cand.strip())
            except Exception:
                pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _fallback(self) -> dict:
        return {}

    def run(self, **kwargs) -> tuple[dict, str]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(**kwargs)},
        ]
        raw = ""
        for attempt in range(MAX_AGENT_RETRIES + 1):
            raw = llm_generate(self.model, self.tokenizer, messages, self.max_tokens)
            parsed = self._parse_json(raw)
            if parsed is not None:
                return parsed, raw
            if attempt < MAX_AGENT_RETRIES:
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "Your output was not valid JSON. Output ONLY the JSON object, nothing else.",
                    },
                ]
        return self._fallback(), raw


# ─── Agent 1 — Question Intent ────────────────────────────────────────────────


class QuestionIntentAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a political-communication analyst. Identify what a journalist "
        "question requires for a CLEAR, DIRECT answer.\n\n"
        "Output ONLY a valid JSON object with these keys:\n"
        '  "question_type": one of yes_no | factual | policy_position | commitment | explanation\n'
        '  "core_intent": one-sentence description of the specific information sought\n'
        '  "required_elements": list of 2-3 short strings (facts/positions a direct answer must include)\n'
        '  "direct_answer_starts_with": example opening of a direct answer (e.g. "Yes,..." or "The plan is...")\n'
        '  "evasion_red_flags": list of 1-2 vague phrases that would indicate evasion\n'
        "No markdown, no commentary. JSON only."
    )

    def _user_message(self, question: str, **_) -> str:
        return f"Question: {question.strip()}\n\nJSON:"

    def _fallback(self) -> dict:
        return {
            "question_type": "factual",
            "core_intent": "unknown",
            "required_elements": [],
            "direct_answer_starts_with": "",
            "evasion_red_flags": [],
        }


# ─── Agent 2 — Answer Content ─────────────────────────────────────────────────


class AnswerContentAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a political-speech analyst. Extract WHAT the politician actually says. "
        "Focus on explicit claims, specific information, vague phrasing, and engagement style. "
        "Ignore what was asked — only describe what was said.\n\n"
        "Output ONLY a valid JSON object with these keys:\n"
        '  "main_claims": list of 2-3 short strings (key assertions actually made)\n'
        '  "specific_info": list of strings (concrete facts/numbers/dates/named commitments; '
        "MUST be empty list [] if none present)\n"
        '  "vague_phrases": list of up to 2 short strings (noncommittal/deflecting phrases)\n'
        '  "engagement_level": exactly one of:\n'
        '    "direct"     = specifically addresses the topic with precision\n'
        '    "partial"    = on-topic but incomplete; hedges or only partially engages\n'
        '    "deflecting" = pivots to a related but different topic; on-theme but evasive\n'
        '    "unrelated"  = complete topic change or refusal; ignores the question entirely\n'
        "No markdown, no commentary. JSON only."
    )

    def _user_message(self, answer: str, **_) -> str:
        return f"Answer: {truncate_words(answer, MAX_ANSWER_WORDS).strip()}\n\nJSON:"

    def _fallback(self) -> dict:
        return {
            "main_claims": [],
            "specific_info": [],
            "vague_phrases": [],
            "engagement_level": "partial",
        }


# ─── Agent 3 — Gap and Evasion ────────────────────────────────────────────────


class GapEvasionAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You compare the EXPECTED answer content (from Question Intent) with the ACTUAL "
        "answer content. Detect missing required elements, partial coverage, ambiguity, "
        "and evasive patterns.\n\n"
        "Output ONLY a valid JSON object with these keys:\n"
        '  "matched_elements": list of strings (required elements ACTUALLY addressed in the answer)\n'
        '  "gaps": list of strings (required elements MISSING from the answer)\n'
        '  "evasion_patterns": list of up to 2 short strings describing how the speaker evades '
        '(e.g. "pivots to talking points", "vague hedging", "refuses to specify")\n'
        '  "gap_severity": exactly one of "low" | "medium" | "high"\n'
        "    low    = at most one minor gap; core intent answered\n"
        "    medium = several gaps OR core intent only partially addressed\n"
        "    high   = most/all required elements missing; clear evasion\n"
        '  "assessment": one short sentence summarising the gap analysis\n'
        "No markdown, no commentary. JSON only."
    )

    def _user_message(self, intent_out: dict, content_out: dict, **_) -> str:
        return (
            f"QUESTION INTENT ANALYSIS:\n{json.dumps(intent_out, indent=2)}\n\n"
            f"ANSWER CONTENT ANALYSIS:\n{json.dumps(content_out, indent=2)}\n\n"
            "Identify gaps and evasion patterns and output JSON:"
        )

    def _fallback(self) -> dict:
        return {
            "matched_elements": [],
            "gaps": [],
            "evasion_patterns": [],
            "gap_severity": "medium",
            "assessment": "parse error — default medium severity",
        }


# ─── Agent 4 — Decision ───────────────────────────────────────────────────────


class DecisionAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are the final classifier in a political response clarity pipeline.\n\n"
        "Decision rule based on Gap and Evasion analysis:\n"
        '  gap_severity = "low"    AND specific_info non-empty           → "Clear Reply"\n'
        '  gap_severity = "medium"                                       → "Ambivalent"\n'
        '  gap_severity = "high"   AND engagement_level in {deflecting,unrelated} → "Clear Non-Reply"\n'
        '  gap_severity = "high"   AND engagement_level in {partial,direct}      → "Ambivalent"\n'
        '  gap_severity = "low"    BUT specific_info is empty            → "Ambivalent"\n\n'
        "Definitions:\n"
        '  "Clear Reply"      = the politician directly answers the specific question\n'
        '  "Ambivalent"       = on-topic but indirect / partial / vague\n'
        '  "Clear Non-Reply"  = evades, refuses, or completely changes topic\n\n'
        "Output ONLY a valid JSON object with these keys:\n"
        '  "label": exactly one of "Clear Reply", "Ambivalent", "Clear Non-Reply"\n'
        '  "confidence": float in [0.0, 1.0]\n'
        '  "reason": one short sentence citing the gap_severity and engagement_level\n'
        "No markdown, no commentary. JSON only."
    )

    def _user_message(
        self,
        question: str,
        answer: str,
        intent_out: dict,
        content_out: dict,
        gap_out: dict,
        **_,
    ) -> str:
        return (
            f"ORIGINAL QUESTION:\n{question.strip()}\n\n"
            f"POLITICIAN'S ANSWER:\n{truncate_words(answer, 150)}\n\n"
            f"[Agent 1] QUESTION INTENT:\n{json.dumps(intent_out, indent=2)}\n\n"
            f"[Agent 2] ANSWER CONTENT:\n{json.dumps(content_out, indent=2)}\n\n"
            f"[Agent 3] GAP AND EVASION:\n{json.dumps(gap_out, indent=2)}\n\n"
            "Apply the decision rule and output the JSON classification:"
        )

    @staticmethod
    def _normalise_label(parsed: dict | None) -> dict | None:
        if not parsed or "label" not in parsed:
            return None
        lo = str(parsed["label"]).lower()
        if "non-reply" in lo or "non reply" in lo or "nonreply" in lo:
            parsed["label"] = "Clear Non-Reply"
        elif "clear reply" in lo or lo == "reply":
            parsed["label"] = "Clear Reply"
        elif "ambival" in lo:
            parsed["label"] = "Ambivalent"
        elif parsed["label"] not in VALID_LABELS:
            parsed["label"] = "Ambivalent"
        return parsed

    def _parse_json(self, raw: str) -> dict | None:
        return self._normalise_label(super()._parse_json(raw))

    def _fallback(self) -> dict:
        return {"label": "Ambivalent", "confidence": 0.33, "reason": "parse error — fallback"}


# ─── Orchestrator ─────────────────────────────────────────────────────────────


@dataclass
class D3Result:
    predictions: list[str]
    logs: list[dict]


class D3Orchestrator:
    """Coordinates the four agents. Keeps every intermediate dict in :attr:`logs`."""

    def __init__(self, model, tokenizer):
        self.agent1 = QuestionIntentAgent(model, tokenizer, AGENT_TOKENS["agent1_intent"])
        self.agent2 = AnswerContentAgent(model, tokenizer, AGENT_TOKENS["agent2_content"])
        self.agent3 = GapEvasionAgent(model, tokenizer, AGENT_TOKENS["agent3_gap"])
        self.agent4 = DecisionAgent(model, tokenizer, AGENT_TOKENS["agent4_decide"])
        self.logs: list[dict] = []

    def run_sample(self, question: str, answer: str) -> str:
        intent_out, _ = self.agent1.run(question=question)
        content_out, _ = self.agent2.run(answer=answer)
        gap_out, _ = self.agent3.run(intent_out=intent_out, content_out=content_out)
        dec_out, _ = self.agent4.run(
            question=question,
            answer=answer,
            intent_out=intent_out,
            content_out=content_out,
            gap_out=gap_out,
        )
        label = dec_out.get("label", "Ambivalent")
        if label not in VALID_LABELS:
            label = "Ambivalent"
        self.logs.append(
            {
                "question": question,
                "answer": answer,
                "agent1": intent_out,
                "agent2": content_out,
                "agent3": gap_out,
                "agent4": dec_out,
                "label": label,
                "confidence": float(dec_out.get("confidence", 0.5)),
                "gap_severity": gap_out.get("gap_severity", "medium"),
                "engagement": content_out.get("engagement_level", "partial"),
            }
        )
        return label

    def run_dataset(self, df: pd.DataFrame, desc: str = "D3-Agentic") -> D3Result:
        preds: list[str] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            preds.append(
                self.run_sample(
                    str(row["question"]).strip(),
                    str(row["interview_answer"]).strip(),
                )
            )
        return D3Result(predictions=preds, logs=self.logs)


def load_qwen_small(model_name: str = QWEN_SMALL):
    """Load Qwen2.5-0.5B-Instruct (or any Qwen2.5 instruct variant)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer
