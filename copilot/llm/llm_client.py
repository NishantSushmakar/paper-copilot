import httpx,ollama
from time import monotonic
from datetime import timedelta
from typing import Optional
from pydantic import BaseModel
from copilot.llm.slots import get_chat_slot, config
from copilot.infra.models import build_chat_model,RetryPolicy



RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


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


def chat_with_llm(
    slot: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    thinking: bool,
) -> LLMResponse:
    slot_config = get_chat_slot(config=config, name=slot)

    model = build_chat_model(
        slot_config,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        request_timeout=30.0,
        retry=RetryPolicy(
            max_attempts=4,
            should_retry=should_retry,
        ),
    )

    start = monotonic()
    response = model.invoke(messages)
    latency = timedelta(seconds=monotonic() - start)

    usage = response.usage_metadata or {}
    metadata = response.response_metadata or {}

    return LLMResponse(
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
