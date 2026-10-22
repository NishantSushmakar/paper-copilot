
import httpx
import random
import time

from datetime import timedelta
from typing import Optional
from pydantic import BaseModel
from copilot.llm.slots import get_chat_slot, config


class LLMResponse(BaseModel):
    text: str
    thinking: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    model: str
    latency: timedelta
    cache_hit: bool


def retry_chat_delay(
    attempt: int,
    deadline: float,
    reason: str,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> bool:
    """Wait before retrying, provided the wait fits the deadline."""

    remaining = deadline - time.monotonic()

    if remaining <= 0:
        return False

    # Exponential backoff with proportional jitter.
    backoff = min(base_delay * (2 ** attempt), max_delay)
    delay_required = random.uniform(backoff, min(2 * backoff, max_delay))

    # Do not sleep if there won't be time for another attempt.
    if delay_required >= remaining:
        print(
            f"{reason}. Cannot retry: "
            f"next delay ({delay_required:.2f}s) exceeds "
            f"remaining budget ({remaining:.2f}s)."
        )
        return False

    print(
        f"{reason}. Retrying in {delay_required:.2f}s "
        f"(retry {attempt + 1})."
    )

    time.sleep(delay_required)

    return time.monotonic() < deadline


def chat_with_llm(
    slot: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    timeout_budget: float = 120,
) -> LLMResponse:

    model_config = get_chat_slot(config=config, name=slot)

    data = {
        "model": model_config.model,
        "messages": messages,
        "reasoning_effort": "high" if thinking else "none",
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }

    max_retries = 3
    endpoint_url = str(model_config.base_url) + "/chat/completions"
    deadline = time.monotonic() + timeout_budget

    retryable_statuses = {429, 500, 502, 503, 504}

    # A timeout on an individual network operation.
    operation_timeout = httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=10.0,
    )

    with httpx.Client(timeout=operation_timeout) as client:
        for attempt in range(max_retries + 1):
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TimeoutError(
                    "LLM request exceeded its total time budget"
                )

            try:
                response = client.post(
                    endpoint_url,
                    json=data,
                )

                response.raise_for_status()
                results = response.json()

                choice = results["choices"][0]
                message = choice["message"]

                # Check the overall deadline after the HTTP operation.
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "LLM request exceeded its total time budget"
                    )

                output = {
                    "text": message["content"],
                    "thinking": message.get("reasoning"),
                    "prompt_tokens": results["usage"]["prompt_tokens"],
                    "completion_tokens": results["usage"]["completion_tokens"],
                    "finish_reason": choice["finish_reason"],
                    "model": model_config.model,
                    "latency": response.elapsed,
                    "cache_hit": False,
                }

                return LLMResponse.model_validate(output)

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                reason = f"Connection failure: {type(exc).__name__}"

                if attempt == max_retries:
                    raise RuntimeError(
                        f"{reason}; exhausted all attempts"
                    ) from exc

                if not retry_chat_delay(
                    attempt, deadline, reason
                ):
                    raise TimeoutError(
                        f"{reason}; insufficient time for another retry"
                    ) from exc

            except (
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ) as exc:
                reason = f"Request timed out: {type(exc).__name__}"

                if attempt == max_retries:
                    raise RuntimeError(
                        f"{reason}; exhausted all attempts"
                    ) from exc

                if not retry_chat_delay(
                    attempt, deadline, reason
                ):
                    raise TimeoutError(
                        f"{reason}; insufficient time for another retry"
                    ) from exc

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                reason = f"HTTP {status}: {exc.response.text[:500]}"

                if status not in retryable_statuses:
                    raise RuntimeError(
                        f"Non-retryable API error: {reason}"
                    ) from exc

                if attempt == max_retries:
                    raise RuntimeError(
                        f"API failed after all attempts: {reason}"
                    ) from exc

                if not retry_chat_delay(
                    attempt, deadline, reason
                ):
                    raise TimeoutError(
                        f"{reason}; insufficient time for another retry"
                    ) from exc


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
