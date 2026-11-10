"""The LLM cache key with tools (decisions in reports/a04.md §4).

A fake model stands in for Ollama, so these run without a server.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import copilot.llm.llm_client as llm_client
from copilot.infra.cache import ResponseCache
from copilot.llm.llm_client import chat_with_llm, get_cache_key, key_messages, key_tools
from copilot.llm.slots import config, get_chat_slot


@tool
def search(query: str) -> str:
    """Search Wikipedia for a query."""
    return ""


@tool
def lookup(term: str) -> str:
    """Look up a term in the current page."""
    return ""


@tool("search")
def search_reworded(query: str) -> str:
    """Search the user's paper library for a query."""
    return ""


QUESTION = {"role": "user", "content": "Who directed the film?"}


def step_two(ids, results=("r1", "r2")):
    """One assistant turn with two parallel calls, then their results."""
    return [
        QUESTION,
        AIMessage(content="", tool_calls=[
            {"name": "search", "args": {"query": "film"}, "id": ids[0]},
            {"name": "lookup", "args": {"term": "director"}, "id": ids[1]},
        ]),
        ToolMessage(content=results[0], tool_call_id=ids[0]),
        ToolMessage(content=results[1], tool_call_id=ids[1]),
    ]


def fresh_ids():
    return [str(uuid.uuid4()), str(uuid.uuid4())]


# ---------- key_messages and key_tools ----------

def test_plain_dict_messages_come_out_unchanged():
    messages = [{"role": "system", "content": "Be brief."}, QUESTION,
                {"role": "assistant", "content": "Paris."}]
    assert key_messages(messages) == messages


def test_langchain_objects_and_dicts_give_the_same_key():
    as_dicts = [QUESTION, {"role": "assistant", "content": "Paris."}]
    as_objects = [HumanMessage(content=QUESTION["content"]), AIMessage(content="Paris.")]
    assert key_messages(as_objects) == key_messages(as_dicts)


def test_the_same_step_with_fresh_ids_gives_the_same_key():
    assert key_messages(step_two(fresh_ids())) == key_messages(step_two(fresh_ids()))


def test_ids_are_renamed_by_position_and_results_follow_their_call():
    keyed = key_messages(step_two(["abc", "xyz"]))
    assert [c["id"] for c in keyed[1]["tool_calls"]] == ["call_1", "call_2"]
    assert [m["tool_call_id"] for m in keyed[2:]] == ["call_1", "call_2"]


def test_results_in_a_different_order_give_a_different_key():
    ids = fresh_ids()
    in_order = step_two(ids)
    swapped = in_order[:2] + [in_order[3], in_order[2]]
    assert key_messages(in_order) != key_messages(swapped)


def test_tool_call_names_and_args_reach_the_key():
    a = step_two(fresh_ids())
    b = step_two(fresh_ids())
    b[1].tool_calls[0]["args"] = {"query": "another film"}
    assert key_messages(a) != key_messages(b)


def test_a_tool_result_without_its_call_raises():
    with pytest.raises(ValueError):
        key_messages([QUESTION, ToolMessage(content="r", tool_call_id="nobody-made-this")])


def test_tools_keep_their_order():
    assert key_tools([search, lookup]) != key_tools([lookup, search])


def test_a_reworded_description_changes_the_tool_json():
    assert search.name == search_reworded.name
    assert key_tools([search]) != key_tools([search_reworded])


# ---------- chat_with_llm with a fake model ----------

class FakeModel:
    """Returns one tool call with a fresh random id, as ChatOllama does."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"query": "film"}, "id": str(uuid.uuid4())}],
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            response_metadata={"done_reason": "stop", "model_name": "qwen3:8b"},
        )


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """An empty cache, a fake model and fixed server info."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    cache = ResponseCache(tmp_path)
    monkeypatch.setattr(llm_client, "llm_cache", cache)
    model = FakeModel()
    model.build_kwargs = []

    def build(slot, **kwargs):
        model.build_kwargs.append(kwargs)
        return model

    monkeypatch.setattr(llm_client, "build_chat_model", build)
    monkeypatch.setattr(llm_client, "ollama_digest", lambda base_url, tag: "sha256:fake")
    monkeypatch.setattr(llm_client, "ollama_version", lambda base_url: "0.0.0")
    monkeypatch.setattr(llm_client, "server_profile", lambda: "fake-profile")
    yield model
    cache.close()


def ask(**overrides):
    args = dict(slot="MEDIUM", messages=[QUESTION], temperature=0, max_tokens=50, thinking=False)
    return chat_with_llm(**(args | overrides))


def test_a_call_without_tools_keeps_its_old_key(fake):
    ask()
    slot = get_chat_slot(config=config, name="MEDIUM")
    old_style = {
        "messages": [QUESTION], "model_tag": slot.model, "digest": "sha256:fake",
        "API_path": getattr(slot, "provider", "ollama"), "context_length": int(slot.context_length),
        "thinking": False, "max_tokens": 50, "temperature": 0.0, "response_format": None,
        "sample_index": 0, "server_profile": "fake-profile", "version": "0.0.0",
    }
    assert get_cache_key(old_style) in llm_client.llm_cache


def test_tools_are_passed_to_the_model(fake):
    ask(tools=[search, lookup])
    assert fake.build_kwargs[0]["tools"] == [search, lookup]


def test_a_cache_hit_returns_the_tool_calls(fake):
    first = ask(tools=[search])
    second = ask(tools=[search])
    assert first.cache_hit is False and second.cache_hit is True
    assert second.tool_calls == first.tool_calls
    assert [c["name"] for c in second.tool_calls] == ["search"]
    assert fake.calls == 1


def test_offering_tools_or_not_are_different_calls(fake):
    ask()
    assert ask(tools=[search]).cache_hit is False


def test_reordered_tools_miss(fake):
    ask(tools=[search, lookup])
    assert ask(tools=[lookup, search]).cache_hit is False


def test_a_reworded_description_misses(fake):
    ask(tools=[search])
    assert ask(tools=[search_reworded]).cache_hit is False


def test_the_next_step_hits_on_replay_despite_new_ids(fake):
    """Run the second agent call twice, each time with the tool result pointing at that
    run's own random call id: the key must be the same."""
    for expected_hit in (False, True):
        call_id = str(uuid.uuid4())
        messages = [
            QUESTION,
            AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "film"}, "id": call_id}]),
            ToolMessage(content="Directed by X.", tool_call_id=call_id),
        ]
        assert ask(messages=messages, tools=[search]).cache_hit is expected_hit


def test_tool_choice_on_an_ollama_slot_raises(fake):
    with pytest.raises(ValueError, match="Ollama"):
        ask(tools=[search], tool_choice="required")
    assert fake.calls == 0


def test_tool_choice_without_tools_raises(fake):
    with pytest.raises(ValueError):
        ask(tool_choice="required")
