import httpx,ollama
from time import monotonic
from datetime import timedelta
from typing import Optional, Union
from pydantic import BaseModel
from copilot.llm.slots import get_chat_slot, config
from copilot.infra.models import build_chat_model,RetryPolicy
import hashlib,json
from copilot.infra.server_info import ollama_digest,ollama_version,server_profile
from copilot.infra.cache import ResponseCache

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
llm_cache = ResponseCache()


def should_retry(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
        ),
    ):
        return True

    if isinstance(exc, ollama.ResponseError):
        return exc.status_code in RETRYABLE_STATUSES

    return False


class LLMResponse(BaseModel):
    text: str
    thinking: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    model: str
    latency: timedelta
    cache_hit: bool

def get_cache_key(response):
    "returns a cache key for the llm request"

    payload = json.dumps(response ,sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chat_with_llm(
    slot: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    response_format:Optional[Union[str, dict]]=None,
    sample_index: int=0
) -> LLMResponse:


    slot_config = get_chat_slot(config=config, name=slot)
    base_url = str(slot_config.base_url).rstrip("/").removesuffix("/v1")


    cache_key_request = {
        "messages": messages,
        "model_tag": slot_config.model,
        "digest":ollama_digest(base_url, slot_config.model),
        "API_path": getattr(slot_config, "provider", "ollama"),
        "context_length":int(slot_config.context_length),
        "thinking":thinking,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "response_format": response_format,
        "sample_index":sample_index,
        "server_profile":server_profile(),
        "version":ollama_version(base_url)
    }

    cache_key = get_cache_key(response=cache_key_request)
    

    final_response = llm_cache.get(cache_key)

    if final_response is not None:
        final_response['cache_hit']=True

        return LLMResponse.model_validate(final_response)

    else: 

        model = build_chat_model(
            slot_config,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            request_timeout=30.0,
            retry=RetryPolicy(
                max_attempts=4,
                should_retry=should_retry,
            )
        )

        start = monotonic()
        response = model.invoke(messages)
        latency = timedelta(seconds=monotonic() - start)

        usage = response.usage_metadata or {}
        metadata = response.response_metadata or {}

        final_response = LLMResponse(
                            text=response.content,
                            thinking=(
                                getattr(response, "additional_kwargs", {}).get("reasoning")
                                or getattr(response, "additional_kwargs", {}).get("reasoning_content")
                            ),
                            prompt_tokens=usage.get("input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            finish_reason=metadata.get("done_reason", "unknown"),
                            model=metadata.get("model_name", slot_config.model),
                            latency=latency,
                            cache_hit=False,
                        )


        llm_cache.set(cache_key,final_response.model_dump())

        return final_response


if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": "tell me how many r in strawberry",
        }
    ]

    print(
        chat_with_llm(
            slot="MEDIUM",
            messages=messages,
            temperature=0,
            max_tokens=5,
            thinking=False,
        )
    )
