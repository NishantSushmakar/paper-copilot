"""Tests for the A2 extraction strategies (copilot/cards/extract.py).

The model is replaced by a fake chat_with_llm, so these tests need no Ollama. Each case
is a decision recorded in reports/a02.md §3: three failure kinds (format, content,
cut_off), one surrounding code fence stripped, failed fields and silent fixes kept per
chunk, and outcomes written as one JSON line per chunk.
"""

import json
from datetime import timedelta

import pytest

from copilot.cards import extract
from copilot.cards.schema import ResultModel
from copilot.llm.llm_client import LLMResponse

GOOD = {
    "tasks": [{"names": ["semantic segmentation"]}],
    "datasets": [{"names": ["Cityscapes"]}],
    "methods": [{"names": ["FRRN", "FRRN A"]}],
    "metrics": [{"names": ["mean IoU"]}],
    "results": [{"dataset": "Cityscapes", "metric": "mean IoU", "method": "FRRN A", "raw_value": "65.7%"}],
    "code_url": None,
}


def reply(text, finish_reason="stop"):
    return LLMResponse(
        text=text,
        thinking="...",
        prompt_tokens=4000,
        completion_tokens=900,
        finish_reason=finish_reason,
        model="qwen3:8b",
        latency=timedelta(seconds=50),
        cache_hit=False,
    )


@pytest.fixture
def fake_model(monkeypatch):
    """Make chat_with_llm return chosen replies in order, and record what it was called with.

    Pass one reply, or a list for several calls (S2).
    """
    calls = []

    def install(responses):
        queue = list(responses) if isinstance(responses, list) else [responses]

        def fake_chat_with_llm(**kwargs):
            calls.append(kwargs)
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(extract, "chat_with_llm", fake_chat_with_llm)
        return calls

    return install


def run_s1(fake_model, response):
    calls = fake_model(response)
    outcome = extract.extract_s1("passage text", slot="MEDIUM", paper_id="p1", chunk_index=2)
    return outcome, calls


def test_valid_card(fake_model):
    outcome, _ = run_s1(fake_model, reply(json.dumps(GOOD)))
    assert outcome.status == "valid"
    assert outcome.card.results[0].parsed_value == 65.7
    assert outcome.error is None
    assert (outcome.paper_id, outcome.chunk_index, outcome.strategy, outcome.slot) == ("p1", 2, "S1", "MEDIUM")


@pytest.mark.parametrize("fenced", ["```json\n{}\n```", "```\n{}\n```", "  ```JSON\n{}\n```  "])
def test_one_code_fence_is_stripped_and_counted(fake_model, fenced):
    outcome, _ = run_s1(fake_model, reply(fenced.replace("{}", json.dumps(GOOD))))
    assert outcome.status == "valid"
    assert outcome.fence_stripped is True


def test_unfenced_reply_is_not_marked(fake_model):
    outcome, _ = run_s1(fake_model, reply(json.dumps(GOOD)))
    assert outcome.fence_stripped is False


@pytest.mark.parametrize(
    "text",
    [
        "Here is the card: " + json.dumps(GOOD),  # prose around the JSON
        json.dumps(GOOD) + "\nHope this helps!",
        "not json at all",
        "",
    ],
)
def test_anything_else_is_a_format_failure(fake_model, text):
    outcome, _ = run_s1(fake_model, reply(text))
    assert outcome.status == "format"
    assert outcome.card is None
    assert outcome.error


def test_cut_off_reply(fake_model):
    outcome, _ = run_s1(fake_model, reply(json.dumps(GOOD)[:40], finish_reason="length"))
    assert outcome.status == "cut_off"  # not counted as a format failure
    assert outcome.card is None


def test_content_failure_keeps_error_and_field(fake_model):
    bad = {**GOOD, "results": [{**GOOD["results"][0], "raw_value": "competitive with full-resolution methods"}]}
    outcome, _ = run_s1(fake_model, reply(json.dumps(bad)))
    assert outcome.status == "content"
    assert outcome.failed_fields == ["results"]
    assert "competitive with full-resolution methods" in outcome.error  # S2 sends this back to the model
    # short form: where the error is, then the validator's own message, nothing for humans only
    assert outcome.error.startswith("results.0: Invalid raw_value")
    assert "errors.pydantic.dev" not in outcome.error
    assert "input_value" not in outcome.error


def test_json_that_is_not_an_object_is_a_content_failure(fake_model):
    outcome, _ = run_s1(fake_model, reply("[1, 2, 3]"))
    assert outcome.status == "content"


def test_silent_fixes_are_kept_per_chunk(fake_model):
    fixed = {**GOOD, "code_url": "github.com/foo/bar"}
    outcome, _ = run_s1(fake_model, reply(json.dumps(fixed)))
    assert outcome.status == "valid"
    assert any("Silent fix" in fix for fix in outcome.silent_fixes)


def test_messages_and_settings(fake_model):
    _, calls = run_s1(fake_model, reply(json.dumps(GOOD)))
    (call,) = calls
    assert call["slot"] == "MEDIUM"
    assert call["messages"][0] == {"role": "system", "content": extract.load_prompt("v3")}
    assert call["messages"][1] == {"role": "user", "content": "passage text"}
    assert (call["temperature"], call["max_tokens"], call["thinking"]) == (0.6, 10000, True)


def test_outcome_is_one_json_line(fake_model):
    outcome, _ = run_s1(fake_model, reply(json.dumps(GOOD)))
    row = json.loads(json.dumps(outcome.model_dump(mode="json")))
    assert row["status"] == "valid"
    assert row["latency_s"] == 50.0


def test_s1_records_one_attempt(fake_model):
    outcome, _ = run_s1(fake_model, reply(json.dumps(GOOD)))
    assert outcome.attempts == 1
    assert [a.status for a in outcome.history] == ["valid"]


# ---------- S2: validate and retry ----------

BAD = {**GOOD, "results": [{**GOOD["results"][0], "raw_value": "high"}]}


def run_s2(fake_model, responses, chunk="passage text"):
    calls = fake_model(responses)
    outcome = extract.extract_s2(chunk, slot="MEDIUM_16K", paper_id="p1", chunk_index=2)
    return outcome, calls


def test_s2_valid_first_time_makes_one_call(fake_model):
    outcome, calls = run_s2(fake_model, [reply(json.dumps(GOOD))])
    assert len(calls) == 1
    assert (outcome.status, outcome.attempts, outcome.strategy) == ("valid", 1, "S2")


def test_s2_retries_a_content_failure_with_the_errors(fake_model):
    outcome, calls = run_s2(fake_model, [reply(json.dumps(BAD)), reply(json.dumps(GOOD))])
    assert len(calls) == 2
    assert outcome.status == "valid"
    assert [a.status for a in outcome.history] == ["content", "valid"]

    first, retry = calls[0]["messages"], calls[1]["messages"]
    assert retry[:2] == first  # same system prompt and chunk
    assert retry[2] == {"role": "assistant", "content": json.dumps(BAD)}  # the model's own answer
    feedback = retry[3]
    assert feedback["role"] == "user"
    assert "results.0: Invalid raw_value: 'high'" in feedback["content"]
    assert "errors.pydantic.dev" not in feedback["content"]


def test_s2_retry_fits_in_the_context(fake_model):
    # a full-size chunk on the 16k slot (reports/a02.md §6): the retry adds the answer and the
    # errors, so its max_tokens is whatever still fits, never more than the first attempt's
    chunk = " ".join(f"The model {i} reaches 85.3 F1 on SQuAD ." for i in range(135))
    assert 2400 < extract.count_tokens(chunk) <= 2600
    outcome, calls = run_s2(fake_model, [reply(json.dumps(BAD)), reply(json.dumps(GOOD))], chunk=chunk)
    retry = calls[1]
    used = sum(extract.count_tokens(m["content"]) for m in retry["messages"])
    assert calls[0]["max_tokens"] == 10000  # the first attempt gets the full budget
    assert retry["max_tokens"] <= 10000
    assert used + extract.TEMPLATE_MARGIN + retry["max_tokens"] <= 16384
    assert [a.max_tokens for a in outcome.history] == [10000, retry["max_tokens"]]


def test_s2_retries_a_format_failure(fake_model):
    outcome, calls = run_s2(fake_model, [reply("Sure! Here it is."), reply(json.dumps(GOOD))])
    assert len(calls) == 2
    assert [a.status for a in outcome.history] == ["format", "valid"]
    assert "not valid JSON" in calls[1]["messages"][3]["content"]


def test_s2_stops_after_three_attempts(fake_model):
    outcome, calls = run_s2(fake_model, [reply(json.dumps(BAD))])  # always the same bad answer
    assert len(calls) == 3
    assert outcome.status == "content"
    assert outcome.attempts == 3
    # each retry sees only the latest answer, so the context doesn't grow
    assert len(calls[2]["messages"]) == 4


def test_s2_does_not_retry_a_cut_off(fake_model):
    outcome, calls = run_s2(fake_model, [reply(json.dumps(GOOD)[:40], finish_reason="length")])
    assert len(calls) == 1
    assert outcome.status == "cut_off"


def test_s2_sums_tokens_and_latency_over_attempts(fake_model):
    outcome, _ = run_s2(fake_model, [reply(json.dumps(BAD)), reply(json.dumps(GOOD))])
    assert outcome.prompt_tokens == 8000
    assert outcome.completion_tokens == 1800
    assert outcome.latency_s == 100.0


# ---------- S3: constrained decoding ----------

def test_s3_sends_the_card_schema_and_the_s1_prompt(fake_model):
    calls = fake_model(reply(json.dumps(GOOD)))
    outcome = extract.extract_s3("passage text", slot="MEDIUM", paper_id="p1", chunk_index=2)
    (call,) = calls
    assert call["response_format"] == extract.CARD_SCHEMA
    assert call["messages"] == extract.build_messages(extract.load_prompt("v3"), "passage text")  # same as S1
    assert (call["temperature"], call["max_tokens"], call["thinking"]) == (0.6, 10000, True)
    assert (outcome.strategy, outcome.status, outcome.attempts) == ("S3", "valid", 1)


def test_s1_sends_no_schema(fake_model):
    _, calls = run_s1(fake_model, reply(json.dumps(GOOD)))
    assert calls[0]["response_format"] is None


def test_s3_output_still_goes_through_the_validators(fake_model):
    # the decoder only enforces shape, so "high" can still arrive and is rejected as before
    calls = fake_model(reply(json.dumps(BAD)))
    outcome = extract.extract_s3("passage text", slot="MEDIUM", paper_id="p1", chunk_index=2)
    assert outcome.status == "content"
    assert outcome.failed_fields == ["results"]


def test_card_schema_matches_the_card_model():
    # decision (b): the card's shape only, without parsed_value and without patterns
    schema = extract.CARD_SCHEMA
    card_fields = set(extract.PaperCardModel.model_fields)
    assert set(schema["properties"]) == card_fields
    assert set(schema["required"]) == card_fields
    result_fields = set(ResultModel.model_fields) - {"parsed_value"}
    result_schema = schema["properties"]["results"]["items"]
    assert set(result_schema["properties"]) == result_fields
    assert "pattern" not in json.dumps(schema)
    extract.PaperCardModel.model_validate(GOOD)  # the example card fits both


def test_card_schema_accepts_what_the_schema_says():
    # every key the decoder can produce is one PaperCardModel accepts: an empty card is valid
    empty = {key: [] for key in ("tasks", "datasets", "methods", "metrics", "results")} | {"code_url": None}
    assert extract.PaperCardModel.model_validate(empty)
