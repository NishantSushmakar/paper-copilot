from copilot.llm.llm_client import get_cache_key

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

CHANGED_INPUT = {"version": "0.40.5",
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
    "server_profile": "fa1-kvq8"
    
    }

def test_sum():

    assert 1+1==2



def test_get_cache_key():
    
    assert get_cache_key(BASE_INPUTS)==get_cache_key(BASE_INPUTS)

    assert get_cache_key(BASE_INPUTS)==get_cache_key(CHANGED_INPUT)
