"""The tool registry (copilot/tools/registry.py), on toy tools. Decisions in reports/a04.md §4."""

import json
import threading
import time
from typing import Annotated

import pytest
from langchain_core.tools import tool
from pydantic import Field

from copilot.tools import registry as reg
from copilot.tools.registry import BadArgument, Registry, ToolFailed, Unreachable


@tool
def search_library(query: str, k: Annotated[int, Field(ge=1, le=10)] = 5) -> list:
    """Search the library."""
    return [f"{query}-{i}" for i in range(k)]


@tool
def get_paper_card(paper_id: Annotated[str, Field(pattern=r"^\d{4}\.\d{5}(v\d+)?$")]) -> dict:
    """Get a paper's card."""
    if paper_id == "0000.00000":
        raise BadArgument("No paper has ID 0000.00000.", "Find the ID with search_library first.")
    return {"id": paper_id}


@tool
def broken() -> str:
    """Always fails inside."""
    raise ToolFailed("The index is missing.", "Answer without the library.")


@tool
def buggy() -> str:
    """Has a bug."""
    return 1 / 0


@tool
def slow(seconds: float) -> str:
    """Sleeps."""
    time.sleep(seconds)
    return "done"


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(reg, "RETRY_WAITS", (0.01, 0.01))


def registry(*tools, timeout=5.0):
    r = Registry()
    for t in tools:
        r.register(t, side_effect=False, timeout=timeout)
    return r


def run(r, name, budget=None, **args):
    return r.execute({"name": name, "args": args, "id": "c1"}, budget=budget)


def error_of(result):
    assert not result.ok
    assert json.loads(result.content) == result.error, "the model must see exactly the structured error"
    assert "\n" not in result.content, "errors are one-line JSON"
    assert set(result.error) == {"error", "what_was_wrong", "retry", "what_to_do"}
    return result.error


# ---------- registration ----------

def test_tools_are_offered_in_registration_order():
    assert [t.name for t in registry(get_paper_card, search_library).tools()] == ["get_paper_card", "search_library"]


def test_a_side_effect_tool_cannot_be_registered_yet():
    with pytest.raises(NotImplementedError):
        Registry().register(search_library, side_effect=True, timeout=5)


def test_a_name_can_be_registered_only_once():
    r = registry(search_library)
    with pytest.raises(ValueError):
        r.register(search_library, side_effect=False, timeout=5)


# ---------- success ----------

def test_a_valid_call_returns_the_tool_output_as_text():
    result = run(registry(search_library), "search_library", query="rag", k=2)
    assert result.ok and json.loads(result.content) == ["rag-0", "rag-1"]
    assert result.metadata["attempts"] == 1


def test_defaults_apply_when_an_optional_argument_is_left_out():
    assert len(json.loads(run(registry(search_library), "search_library", query="rag").content)) == 5


def test_to_message_keeps_the_call_id():
    message = run(registry(search_library), "search_library", query="rag").to_message("call-9")
    assert message.tool_call_id == "call-9"


# ---------- registry-level errors (schema) ----------

def test_an_unknown_tool_lists_the_real_ones():
    error = error_of(run(registry(search_library), "search_libary", query="rag"))
    assert error["error"] == "unknown_tool" and "search_library" in error["what_to_do"]


def test_a_missing_argument_is_a_bad_argument():
    error = error_of(run(registry(search_library), "search_library"))
    assert error["error"] == "bad_argument" and error["retry"] is True
    assert "`query` is required" in error["what_was_wrong"]


@pytest.mark.parametrize("k", [10.0, "5", True])
def test_wrong_types_are_not_coerced(k):
    error = error_of(run(registry(search_library), "search_library", query="rag", k=k))
    assert error["error"] == "bad_argument" and "`k`" in error["what_was_wrong"]


def test_an_extra_argument_fails():
    error = error_of(run(registry(search_library), "search_library", query="rag", top=3))
    assert "`top` is not an argument" in error["what_was_wrong"]


def test_an_explicit_null_for_an_optional_argument_means_not_given():
    assert len(json.loads(run(registry(search_library), "search_library", query="rag", k=None).content)) == 5


def test_an_explicit_null_for_a_required_argument_still_fails():
    error = error_of(run(registry(search_library), "search_library", query=None))
    assert "`query`" in error["what_was_wrong"]


def test_k_above_the_cap_fails_and_says_what_was_sent():
    error = error_of(run(registry(search_library), "search_library", query="rag", k=50))
    assert "`k`" in error["what_was_wrong"] and "50" in error["what_was_wrong"]


def test_a_paper_id_must_match_the_pattern():
    error = error_of(run(registry(get_paper_card), "get_paper_card", paper_id="transformer paper"))
    assert "'transformer paper'" in error["what_was_wrong"]


def test_the_tool_does_not_run_when_the_arguments_are_bad(monkeypatch):
    submitted = []
    monkeypatch.setattr(reg._pool, "submit", lambda *a, **k: submitted.append(a))
    run(registry(search_library), "search_library", query="rag", k=50)
    assert submitted == []


# ---------- tool-level errors ----------

def test_a_tool_bad_argument_keeps_the_tools_own_text():
    error = error_of(run(registry(get_paper_card), "get_paper_card", paper_id="0000.00000"))
    assert error == {"error": "bad_argument", "what_was_wrong": "No paper has ID 0000.00000.",
                     "retry": True, "what_to_do": "Find the ID with search_library first."}


def test_a_tool_failure_says_do_not_retry():
    error = error_of(run(registry(broken), "broken"))
    assert error["error"] == "tool_failed" and error["retry"] is False


def test_a_bug_in_a_tool_becomes_a_clean_error_and_the_cause_goes_to_metadata():
    result = run(registry(buggy), "buggy")
    assert error_of(result)["error"] == "tool_failed"
    assert "ZeroDivisionError" in result.metadata["exception"]
    assert "ZeroDivisionError" not in result.content


# ---------- unreachable: quiet retries ----------

def flaky_tool(failures, exc=Unreachable("down")):
    calls = {"n": 0}

    @tool
    def flaky() -> str:
        """Fails a few times, then works."""
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return "ok"

    return flaky, calls


def test_unreachable_is_retried_quietly_and_can_recover():
    flaky, calls = flaky_tool(failures=2)
    result = run(registry(flaky), "flaky")
    assert result.ok and result.content == "ok" and calls["n"] == 3
    assert result.metadata["attempts"] == 3


def test_a_connection_error_counts_as_unreachable():
    flaky, _ = flaky_tool(failures=1, exc=ConnectionError("refused"))
    assert run(registry(flaky), "flaky").ok


def test_still_unreachable_after_retries_tells_the_model_not_to_call_again():
    flaky, calls = flaky_tool(failures=99)
    result = run(registry(flaky), "flaky")
    error = error_of(result)
    assert error["error"] == "tool_failed" and error["retry"] is False
    assert "unavailable" in error["what_was_wrong"] and calls["n"] == 3
    assert result.metadata["unreachable"] is True


def test_other_errors_are_not_retried():
    flaky, calls = flaky_tool(failures=1, exc=ToolFailed("bad", "stop"))
    assert not run(registry(flaky), "flaky").ok and calls["n"] == 1


# ---------- time limits ----------

def test_a_call_over_the_tools_limit_times_out():
    result = run(registry(slow, timeout=0.1), "slow", seconds=1.0)
    assert error_of(result)["error"] == "tool_failed" and result.metadata["timed_out"] is True


def test_the_run_budget_caps_the_tools_limit():
    result = run(registry(slow, timeout=5.0), "slow", budget=0.1, seconds=1.0)
    assert result.metadata["timed_out"] is True and result.metadata["time_limit"] == pytest.approx(0.1)


def test_the_tools_limit_caps_a_large_budget():
    result = run(registry(slow, timeout=0.1), "slow", budget=100, seconds=1.0)
    assert result.metadata["time_limit"] == pytest.approx(0.1)


def test_no_budget_left_means_no_call():
    error = error_of(run(registry(slow), "slow", budget=0, seconds=0.0))
    assert error["error"] == "tool_failed" and "budget" in error["what_was_wrong"]


def test_a_fast_call_within_the_limits_succeeds():
    assert run(registry(slow, timeout=1.0), "slow", budget=1.0, seconds=0.01).ok


def test_the_tool_runs_off_the_calling_thread():
    seen = []

    @tool
    def where() -> str:
        """Reports its thread."""
        seen.append(threading.current_thread().name)
        return "ok"

    run(registry(where), "where")
    assert seen[0].startswith("tool")
