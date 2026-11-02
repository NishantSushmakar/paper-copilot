import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

from copilot.cards.chunking import count_tokens
from copilot.cards.schema import PaperCardModel
from copilot.llm.llm_client import LLMResponse, chat_with_llm
from copilot.llm.slots import config, get_chat_slot

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"

# The same settings for every strategy and both slots (reports/a02.md §3).
GENERATION = {"temperature": 0.6, "max_tokens": 4000, "thinking": True}

# One code fence around the whole reply: ```json ... ``` or ``` ... ```
FENCE = re.compile(r"^\s*```(?:json)?[ \t]*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)

# S2 (reports/a02.md §3)
MAX_ATTEMPTS = 3
RETRY_ON = {"format", "content"}  # a cut-off reply is a budget problem; retrying it repeats the cut
RETRY_PROMPT_VERSION = "v1"  # prompts/retry_v1.txt
TEMPLATE_MARGIN = 64  # chat-template tokens around the messages, not counted by count_tokens
MIN_RETRY_TOKENS = 1000  # below this, a retry has too little room to think and answer


# S3 (reports/a02.md §3): the card's shape only. No parsed_value (computed by the
# schema), and no patterns: a number pattern on raw_value made the model invent
# "99.99%" for a paper that said "very high". Content is left to the validators.
_ENTITY = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
    "additionalProperties": False,
}
_RESULT = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in ("dataset", "metric", "method", "raw_value")},
    "required": ["dataset", "metric", "method", "raw_value"],
    "additionalProperties": False,
}
CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {"type": "array", "items": _ENTITY},
        "datasets": {"type": "array", "items": _ENTITY},
        "methods": {"type": "array", "items": _ENTITY},
        "metrics": {"type": "array", "items": _ENTITY},
        "results": {"type": "array", "items": _RESULT},
        "code_url": {"type": ["string", "null"]},
    },
    "required": ["tasks", "datasets", "methods", "metrics", "results", "code_url"],
    "additionalProperties": False,
}


Status = Literal["valid", "format", "content", "cut_off"]


class Attempt(BaseModel):
    attempt: int
    status: Status
    error: Optional[str] = None
    failed_fields: list[str] = []
    max_tokens: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cache_hit: bool


class ChunkOutcome(BaseModel):
    paper_id: str
    chunk_index: int
    strategy: str
    slot: str
    prompt_version: str
    status: Status
    card: Optional[PaperCardModel] = None
    error: Optional[str] = None
    failed_fields: list[str] = []
    fence_stripped: bool = False
    silent_fixes: list[str] = []
    raw_text: str
    attempts: int = 1
    history: list[Attempt] = []  # one entry per model call
    # Summed over attempts for S2:
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cache_hit: bool


def load_prompt(version: str, kind: str = "extraction") -> str:
    return (PROMPT_DIR / f"{kind}_{version}.txt").read_text(encoding="utf-8")


def build_messages(system_prompt: str, chunk_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk_text},
    ]


def parse_reply(text: str) -> tuple[object, bool, Optional[str]]:
    """Return (data, fence_stripped, error). The reply is not JSON when error is set."""
    fence_stripped = False
    match = FENCE.match(text)
    if match:
        text = match.group(1)
        fence_stripped = True

    try:
        return json.loads(text), fence_stripped, None
    except json.JSONDecodeError as exc:
        return None, fence_stripped, f"The reply is not valid JSON: {exc}"


class _SilentFixCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.fixes: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Silent fix"):
            self.fixes.append(message)


@contextmanager
def collect_silent_fixes():
    """Collect the schema's "Silent fix" log messages while a card is validated."""
    schema_logger = logging.getLogger("copilot.cards.schema")
    collector = _SilentFixCollector()
    old_level = schema_logger.level
    schema_logger.addHandler(collector)
    if not schema_logger.isEnabledFor(logging.INFO):
        schema_logger.setLevel(logging.INFO)
    try:
        yield collector.fixes
    finally:
        schema_logger.removeHandler(collector)
        schema_logger.setLevel(old_level)


def short_errors(exc: ValidationError) -> str:
    """One line per error: where it is and the validator's own message.

    Drops what str(exc) adds for humans (the echoed input, the errors.pydantic.dev link),
    since S2 sends this text back to the model.
    """
    lines = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(card)"
        lines.append(f"{where}: {err['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def validate_card(data: object) -> tuple[Optional[PaperCardModel], Optional[str], list[str], list[str]]:
    """Return (card, error, failed_fields, silent_fixes). The card is None when invalid."""
    with collect_silent_fixes() as fixes:
        try:
            card = PaperCardModel.model_validate(data)
        except ValidationError as exc:
            # First part of each error location: the card field that failed ("results", "code_url", ...)
            fields = sorted({str(err["loc"][0]) if err["loc"] else "(card)" for err in exc.errors()})
            return None, short_errors(exc), fields, list(fixes)
    return card, None, [], list(fixes)


def judge(
    response: LLMResponse,
    *,
    paper_id: str,
    chunk_index: int,
    strategy: str,
    slot: str,
    prompt_version: str,
    attempts: int = 1,
) -> ChunkOutcome:
    """Turn one model reply into an outcome: cut_off, format, content or valid."""
    common = dict(
        paper_id=paper_id,
        chunk_index=chunk_index,
        strategy=strategy,
        slot=slot,
        prompt_version=prompt_version,
        raw_text=response.text,
        attempts=attempts,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=response.latency.total_seconds(),
        cache_hit=response.cache_hit,
    )

    # Checked first: a reply stopped by max_tokens is usually broken JSON too,
    # but the cause is the budget, not the model.
    if response.finish_reason == "length":
        return ChunkOutcome(
            status="cut_off",
            error=f"The reply was cut off at max_tokens after {response.completion_tokens} output tokens.",
            **common,
        )

    data, fence_stripped, error = parse_reply(response.text)
    if error is not None:
        return ChunkOutcome(status="format", error=error, fence_stripped=fence_stripped, **common)

    card, error, failed_fields, fixes = validate_card(data)
    if card is None:
        return ChunkOutcome(
            status="content",
            error=error,
            failed_fields=failed_fields,
            silent_fixes=fixes,
            fence_stripped=fence_stripped,
            **common,
        )

    return ChunkOutcome(status="valid", card=card, silent_fixes=fixes, fence_stripped=fence_stripped, **common)


def as_attempt(outcome: ChunkOutcome, attempt: int, max_tokens: int) -> Attempt:
    return Attempt(
        attempt=attempt,
        status=outcome.status,
        error=outcome.error,
        failed_fields=outcome.failed_fields,
        max_tokens=max_tokens,
        finish_reason=outcome.finish_reason,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        latency_s=outcome.latency_s,
        cache_hit=outcome.cache_hit,
    )


def _single_call(
    chunk_text: str,
    *,
    strategy: str,
    response_format: Optional[dict],
    slot: str,
    paper_id: str,
    chunk_index: int,
    prompt_version: str,
) -> ChunkOutcome:
    """One call, one judgement, no retry. S1 and S3 differ only in response_format."""
    messages = build_messages(load_prompt(prompt_version), chunk_text)
    response = chat_with_llm(slot=slot, messages=messages, response_format=response_format, **GENERATION)
    outcome = judge(
        response,
        paper_id=paper_id,
        chunk_index=chunk_index,
        strategy=strategy,
        slot=slot,
        prompt_version=prompt_version,
    )
    outcome.history = [as_attempt(outcome, 1, GENERATION["max_tokens"])]
    return outcome


def extract_s1(
    chunk_text: str,
    *,
    slot: str,
    paper_id: str,
    chunk_index: int,
    prompt_version: str = "v3",
) -> ChunkOutcome:
    """S1: prompt only. One call, one judgement, no retry."""
    return _single_call(
        chunk_text,
        strategy="S1",
        response_format=None,
        slot=slot,
        paper_id=paper_id,
        chunk_index=chunk_index,
        prompt_version=prompt_version,
    )


def extract_s3(
    chunk_text: str,
    *,
    slot: str,
    paper_id: str,
    chunk_index: int,
    prompt_version: str = "v3",
) -> ChunkOutcome:
    """S3: constrained decoding. Same prompt and settings as S1, but every generated token
    must fit CARD_SCHEMA, so the reply is always JSON of the card's shape."""
    return _single_call(
        chunk_text,
        strategy="S3",
        response_format=CARD_SCHEMA,
        slot=slot,
        paper_id=paper_id,
        chunk_index=chunk_index,
        prompt_version=prompt_version,
    )



def retry_feedback(outcome: ChunkOutcome) -> str:
    errors = "\n".join(f"- {line}" for line in outcome.error.splitlines())
    return load_prompt(RETRY_PROMPT_VERSION, kind="retry").format(errors=errors)


def retry_max_tokens(slot: str, messages: list[dict]) -> int:
    """The largest max_tokens that still fits in the slot's context with these messages."""
    context = get_chat_slot(config=config, name=slot).context_length
    used = sum(count_tokens(message["content"]) for message in messages) + TEMPLATE_MARGIN
    return min(GENERATION["max_tokens"], context - used)


def extract_s2(
    chunk_text: str,
    *,
    slot: str,
    paper_id: str,
    chunk_index: int,
    prompt_version: str = "v3",
) -> ChunkOutcome:
    """S2: validate, and on a format or content failure retry with the errors fed back.

    A retry continues the conversation: the original messages, the model's latest answer,
    and the error list as a new user turn. Only the latest answer is kept, so the context
    doesn't grow with every attempt. Tokens and latency are summed over attempts.
    """
    base = build_messages(load_prompt(prompt_version), chunk_text)
    messages = base
    max_tokens = GENERATION["max_tokens"]
    history: list[Attempt] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = chat_with_llm(
            slot=slot,
            messages=messages,
            temperature=GENERATION["temperature"],
            max_tokens=max_tokens,
            thinking=GENERATION["thinking"],
        )
        outcome = judge(
            response,
            paper_id=paper_id,
            chunk_index=chunk_index,
            strategy="S2",
            slot=slot,
            prompt_version=prompt_version,
            attempts=attempt,
        )
        history.append(as_attempt(outcome, attempt, max_tokens))

        if outcome.status not in RETRY_ON or attempt == MAX_ATTEMPTS:
            break

        messages = base + [
            {"role": "assistant", "content": outcome.raw_text},
            {"role": "user", "content": retry_feedback(outcome)},
        ]
        max_tokens = retry_max_tokens(slot, messages)
        if max_tokens < MIN_RETRY_TOKENS:
            logger.warning(
                "Not retrying %s chunk %d: only %d tokens left for the reply", paper_id, chunk_index, max_tokens
            )
            break

    return outcome.model_copy(
        update={
            "attempts": len(history),
            "history": history,
            "prompt_tokens": sum(a.prompt_tokens for a in history),
            "completion_tokens": sum(a.completion_tokens for a in history),
            "latency_s": sum(a.latency_s for a in history),
            "cache_hit": all(a.cache_hit for a in history),
        }
    )
