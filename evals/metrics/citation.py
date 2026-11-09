"""Citation recall and precision (ALCE, Gao et al. 2023) with the NLI slot as the judge (A3).

For each answer sentence, the judge reads a premise (cited chunk text) and a hypothesis (the claim):
    recall     the sentence is supported if all its cited chunks, joined in citation order, entail
               it. A sentence with no citation is not supported.
    precision  when the joined chunks entail a sentence with 2+ citations, a citation is relevant if
               its chunk entails the sentence alone, or if the other chunks no longer entail it
               without this one. Otherwise each citation counts as relevant only if the joined
               chunks entail the sentence (as in ALCE's code: unsupported sentences earn no precision).
Per answer: recall = supported sentences / sentences, precision = relevant citations / citations
(0 when there are none). Runs average these over answers that didn't abstain.

Entailed = "entailment" is the most probable of the three NLI labels (no tuned threshold).
Premise + hypothesis over 512 tokens: the premise is cut, never the hypothesis, and it's recorded.

Two hypothesis formats (reports/a03.md §9):
    A  the question followed by the answer sentence
    B  the question and sentence rewritten by MEDIUM into one standalone claim (prompts/claim_rewrite_v1.txt)
"""

import re
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import BaseModel

from copilot.infra.nli import label_id, nli_model
from copilot.llm.llm_client import chat_with_llm

MAX_TOKENS = 512
PROMPT_DIR = Path(__file__).parent / "prompts"
REWRITE_PROMPT = "v1"
REWRITE = {"slot": "MEDIUM", "thinking": False, "temperature": 0.7, "max_tokens": 200}  # Qwen3 non-thinking: 0.7
CLAIM_PREFIX = re.compile(r"^\s*claim\s*:\s*", re.IGNORECASE)


class Verdict(BaseModel):
    entailed: bool
    p_entailment: float
    truncated: bool


class SentenceVerdict(BaseModel):
    hypothesis: str
    together: Verdict | None  # None when the sentence has no citation
    alone: list[Verdict]  # per cited chunk, in citation order
    without: list[Verdict | None]  # per cited chunk: the other chunks without it (None with 1 citation)
    relevant: list[bool]  # ALCE precision, per cited chunk

    @property
    def supported(self) -> bool:
        return self.together is not None and self.together.entailed


class NLIJudge:
    def __init__(self):
        self.tokenizer, self.model = nli_model()
        self.entailment = label_id("entailment")
        self.seen: dict[tuple[str, str], Verdict] = {}
        self.calls = 0

    def __call__(self, premise: str, hypothesis: str) -> Verdict:
        if (premise, hypothesis) not in self.seen:
            self.calls += 1
            length = len(self.tokenizer(premise, hypothesis)["input_ids"])
            encoded = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=MAX_TOKENS,
                                     return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**encoded).logits[0].float(), dim=-1)
            self.seen[(premise, hypothesis)] = Verdict(
                entailed=int(probs.argmax()) == self.entailment,
                p_entailment=float(probs[self.entailment]), truncated=length > MAX_TOKENS)
        return self.seen[(premise, hypothesis)]


def hypothesis_a(question: str, sentence: str) -> str:
    return f"{question.strip()} {sentence.strip()}"


def rewrite_prompt(version: str = REWRITE_PROMPT) -> str:
    return (PROMPT_DIR / f"claim_rewrite_{version}.txt").read_text(encoding="utf-8").strip()


def hypothesis_b(question: str, sentence: str) -> str:
    response = chat_with_llm(messages=[
        {"role": "system", "content": rewrite_prompt()},
        {"role": "user", "content": f"Question: {question.strip()}\nAnswer sentence: {sentence.strip()}"},
    ], **REWRITE)
    lines = [line for line in response.text.strip().splitlines() if line.strip()]
    return CLAIM_PREFIX.sub("", lines[0]).strip() if lines else sentence.strip()


def judge_sentence(judge: NLIJudge, hypothesis: str, chunks: Sequence[str]) -> SentenceVerdict:
    """chunks: the text of each cited chunk, in citation order."""
    if not chunks:
        return SentenceVerdict(hypothesis=hypothesis, together=None, alone=[], without=[], relevant=[])
    together = judge("\n".join(chunks), hypothesis)
    alone = [judge(chunk, hypothesis) for chunk in chunks]
    without = [judge("\n".join(chunks[:i] + chunks[i + 1:]), hypothesis) if len(chunks) > 1 else None
               for i in range(len(chunks))]
    if together.entailed and len(chunks) > 1:
        relevant = [a.entailed or not w.entailed for a, w in zip(alone, without)]
    else:
        relevant = [together.entailed] * len(chunks)
    return SentenceVerdict(hypothesis=hypothesis, together=together, alone=alone, without=without,
                           relevant=relevant)


def answer_scores(verdicts: Sequence[SentenceVerdict]) -> dict:
    citations = sum(len(v.relevant) for v in verdicts)
    return {
        "sentences": len(verdicts),
        "citations": citations,
        "recall": sum(v.supported for v in verdicts) / len(verdicts) if verdicts else 0.0,
        "precision": sum(sum(v.relevant) for v in verdicts) / citations if citations else 0.0,
        "truncated": sum(v.together is not None and v.together.truncated for v in verdicts),
    }


def cohen_kappa(gold: Sequence[bool], pred: Sequence[bool]) -> float | None:
    """Agreement beyond chance; None when chance agreement is already 1 (one class only)."""
    n = len(gold)
    observed = sum(g == p for g, p in zip(gold, pred)) / n
    p_gold, p_pred = sum(gold) / n, sum(pred) / n
    chance = p_gold * p_pred + (1 - p_gold) * (1 - p_pred)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def agreement(gold: Sequence[bool], pred: Sequence[bool]) -> dict:
    pairs = list(zip(gold, pred))
    if not pairs:
        return {"n": 0}
    return {
        "n": len(pairs),
        "accuracy": sum(g == p for g, p in pairs) / len(pairs),
        "kappa": cohen_kappa(gold, pred),
        "tp": sum(g and p for g, p in pairs), "fp": sum(p and not g for g, p in pairs),
        "fn": sum(g and not p for g, p in pairs), "tn": sum(not g and not p for g, p in pairs),
    }
