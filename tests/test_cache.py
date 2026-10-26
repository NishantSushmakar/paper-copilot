import copy

import httpx
import pytest

import copilot.llm.llm_client as llm_client
from copilot.infra.cache import ResponseCache
from copilot.llm.llm_client import chat_with_llm, get_cache_key


# ---------- Key tests: no server needed ----------

BASE_INPUTS = {
    "messages": [{"role": "user", "content": "What is 17*23?"}],
    "model_tag": "qwen3:8b",
    "digest": "sha256:fake-digest-0001",
    "API_path": "ollama",
    "context_length": 8192,
    "thinking": False,
    "max_tokens": 20,
    "temperature": 0.0,
    "response_format": None,
    "sample_index": 0,
    "server_profile": "fa1-kvq8",
    "version": "0.40.5",
}

# One different value per input, used to check that each input reaches the key.
CHANGED_VALUES = {
    "messages": [{"role": "user", "content": "What is 17*24?"}],
    "model_tag": "qwen3:4b-q4_K_M",
    "digest": "sha256:fake-digest-0002",
    "API_path": "openai",
    "context_length": 4096,
    "thinking": True,
    "max_tokens": 21,
    "temperature": 0.7,
    "response_format": "json",
    "sample_index": 1,
    "server_profile": "fa0-kvf16",
    "version": "0.41.0",
}


def key_of(inputs):
    return get_cache_key(copy.deepcopy(inputs))


def test_same_inputs_give_same_key():
    assert key_of(BASE_INPUTS) == key_of(BASE_INPUTS)


def test_dict_key_order_is_ignored():
    reordered = dict(reversed(list(copy.deepcopy(BASE_INPUTS).items())))
    reordered["messages"] = [{"content": "What is 17*23?", "role": "user"}]
    assert key_of(reordered) == key_of(BASE_INPUTS)


def test_message_order_matters():
    a = copy.deepcopy(BASE_INPUTS)
    a["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    b = copy.deepcopy(a)
    b["messages"].reverse()
    assert key_of(a) != key_of(b), "swapping message order should change the key"


def test_changed_values_cover_every_input():
    assert set(CHANGED_VALUES) == set(BASE_INPUTS)


@pytest.mark.parametrize("field", sorted(CHANGED_VALUES))
def test_each_input_changes_the_key(field):
    changed = copy.deepcopy(BASE_INPUTS)
    changed[field] = CHANGED_VALUES[field]
    assert key_of(changed) != key_of(BASE_INPUTS), f"changing {field} did not change the key"


# ---------- Integration tests: need Ollama ----------

def ollama_running():
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=1).is_success
    except httpx.HTTPError:
        return False


needs_ollama = pytest.mark.skipif(not ollama_running(), reason="Ollama is not running")

PROMPT = [{"role": "user", "content": "What is 17*23? Answer with just the number."}]


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Point chat_with_llm at an empty cache in a temporary folder."""
    cache = ResponseCache(tmp_path)
    monkeypatch.setattr(llm_client, "llm_cache", cache)
    yield cache
    cache.close()


def ask(**overrides):
    args = dict(slot="MEDIUM", messages=PROMPT, temperature=0, max_tokens=10, thinking=False)
    return chat_with_llm(**(args | overrides))


@needs_ollama
def test_miss_then_hit(temp_cache):
    first = ask()
    second = ask()
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert second.latency == first.latency, "a hit should keep the original call's latency"


@needs_ollama
def test_changed_temperature_misses(temp_cache):
    ask()
    assert ask(temperature=0.7).cache_hit is False


@needs_ollama
def test_int_and_float_temperature_share_an_entry(temp_cache):
    ask(temperature=0)
    assert ask(temperature=0.0).cache_hit is True


@needs_ollama
def test_cache_survives_reopening(tmp_path, monkeypatch, temp_cache):
    ask()
    temp_cache.close()
    reopened = ResponseCache(tmp_path)
    monkeypatch.setattr(llm_client, "llm_cache", reopened)
    assert ask().cache_hit is True
    reopened.close()
