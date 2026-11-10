"""build_chat_model (copilot/infra/models.py): tools are bound inside the retry wrapper."""

from types import SimpleNamespace

from langchain_core.runnables.retry import RunnableRetry
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from copilot.infra.models import RetryPolicy, build_chat_model

SLOT = SimpleNamespace(model="qwen3:8b", base_url="http://localhost:11434/v1", context_length=8192)
RETRY = RetryPolicy(max_attempts=2, should_retry=lambda exc: False)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"


def build(**kwargs):
    return build_chat_model(SLOT, temperature=0.0, max_tokens=100, thinking=False, request_timeout=30.0,
                            retry=RETRY, **kwargs)


def test_without_tools_the_retry_wraps_the_plain_model():
    assert isinstance(build().bound, ChatOllama)


def test_tools_are_bound_and_calls_still_retry():
    # LangChain lifts the tool binding outside: binding(tools) -> retry -> model
    runnable = build(tools=[get_weather])
    assert [t["function"]["name"] for t in runnable.kwargs["tools"]] == ["get_weather"]
    assert isinstance(runnable.bound, RunnableRetry) and isinstance(runnable.bound.bound, ChatOllama)


def test_an_empty_tool_list_binds_nothing():
    assert isinstance(build(tools=[]).bound, ChatOllama)


def test_tool_choice_reaches_bind_tools_only_when_set():
    openai_slot = SimpleNamespace(provider="openai", model="m", base_url="http://localhost:8000/v1",
                                  context_length=8192)
    forced = build_chat_model(openai_slot, temperature=0.0, max_tokens=100, thinking=False,
                              request_timeout=30.0, retry=RETRY, tools=[get_weather], tool_choice="required")
    assert forced.kwargs["tool_choice"] == "required"
    assert "tool_choice" not in build(tools=[get_weather]).kwargs
