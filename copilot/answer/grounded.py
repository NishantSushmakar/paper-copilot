"""Grounded answers with citations (A3): retrieve k chunks, answer from them, cite them.

The chunks go into the prompt labelled [1] to [k]. The model cites labels, and parse() maps
each label back to its chunk ID, so the stored citations refer to chunk IDs while the model
never has to copy a 50-character ID. The answer that answer F1 scores has the markers removed.

An answer is split into sentences. A sentence with no valid citation is flagged as uncited,
except "Unanswerable", which the prompt says to leave uncited.
"""

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from copilot.llm.llm_client import LLMResponse, chat_with_llm
from copilot.retrieval.chunking import SENTENCE_BREAK, Chunk
from copilot.retrieval.hit import Hit

PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSION = "v1"  # prompts/grounded_v1.txt

# [2], [2][4] and [2, 4] all count; the numbers are labels, 1 to k.
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Line breaks, and the chunker's sentence breaks (which skip "Eq.", "e.g.", "Fig." and the like),
# except after a list number such as "1." or "12.".
SENTENCE_END = re.compile(rf"\n+|(?<!\b\d\.)(?<!\b\d\d\.){SENTENCE_BREAK}")
WORD_CHAR = re.compile(r"\w")
ANSWER_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)


class CitedSentence(BaseModel):
    text: str  # without its markers
    labels: list[int]  # as the model wrote them, valid or not
    chunk_ids: list[str]  # the valid labels, mapped back


class GroundedAnswer(BaseModel):
    raw: str  # the model's answer, markers included
    answer: str  # markers removed: what answer F1 scores
    abstained: bool  # answered "Unanswerable"
    retrieved: list[str]  # chunk IDs in label order: [1] is retrieved[0]
    sentences: list[CitedSentence]
    uncited: int  # sentences with no valid citation (0 when abstained)
    invalid_labels: list[int]  # labels outside 1..k


def prompt_path(version: str = PROMPT_VERSION) -> Path:
    return PROMPT_DIR / f"grounded_{version}.txt"


def system_prompt(version: str = PROMPT_VERSION) -> str:
    return prompt_path(version).read_text(encoding="utf-8").strip()


def user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    evidence = "\n".join(f"[{label}] {chunk.text}" for label, chunk in enumerate(chunks, 1))
    return f"[Evidence]\n{evidence}\n[End of evidence]\nQuestion: {question}"


def labels_in(text: str) -> list[int]:
    return [int(n) for group in CITATION.findall(text) for n in group.split(",")]


def strip_citations(text: str) -> str:
    text = CITATION.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # "Twitter [4]." -> "Twitter."
    return re.sub(r"[ \t]+", " ", text).strip()


def is_unanswerable(answer: str) -> bool:
    return re.sub(r"[^a-z]", "", answer.lower()) == "unanswerable"


def parse(raw: str, retrieved: Sequence[str]) -> GroundedAnswer:
    raw = ANSWER_PREFIX.sub("", raw.strip())
    sentences: list[CitedSentence] = []
    invalid: list[int] = []
    for piece in SENTENCE_END.split(raw):
        labels = labels_in(piece)
        valid = [label for label in labels if 1 <= label <= len(retrieved)]
        invalid += [label for label in labels if label not in valid]
        text = strip_citations(piece)
        if not WORD_CHAR.search(text) and sentences:
            # "... Y. [2]" or "... Y. [2], [3]": markers after the full stop belong to Y
            sentences[-1].labels += labels
            sentences[-1].chunk_ids += [retrieved[label - 1] for label in valid]
        elif text:
            sentences.append(CitedSentence(text=text, labels=labels,
                                           chunk_ids=[retrieved[label - 1] for label in valid]))
    for sentence in sentences:
        sentence.chunk_ids = list(dict.fromkeys(sentence.chunk_ids))

    answer = strip_citations(raw)
    abstained = is_unanswerable(answer)
    return GroundedAnswer(raw=raw, answer=answer, abstained=abstained, retrieved=list(retrieved),
                          sentences=sentences, invalid_labels=invalid,
                          uncited=0 if abstained else sum(not s.chunk_ids for s in sentences))


def answer_question(
    question: str,
    chunks: Sequence[Chunk],
    retrieve: Callable[[str, Sequence[Chunk], int], list[Hit]],
    k: int,
    *,
    slot: str,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    prompt_version: str = PROMPT_VERSION,
    request_timeout: float = 120.0,
) -> tuple[GroundedAnswer, LLMResponse]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    top = [by_id[hit.chunk_id] for hit in retrieve(question, chunks, k)]
    messages = [
        {"role": "system", "content": system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt(question, top)},
    ]
    response = chat_with_llm(slot=slot, messages=messages, temperature=temperature,
                             max_tokens=max_tokens, thinking=thinking, request_timeout=request_timeout)
    return parse(response.text, [chunk.chunk_id for chunk in top]), response
