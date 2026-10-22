"""Build LangChain chat models from model slots.

This module only wires a slot's config into a LangChain chat model. The policy
decisions (which errors to retry, how many attempts, how long to wait) are passed
in by the caller.
"""

from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    should_retry: Callable[[BaseException], bool]
    initial_wait: float = 1.0  # seconds before the first retry; doubles each time, with jitter
    max_wait: float = 10.0     # cap on a single wait


def build_chat_model(
    slot,
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    request_timeout: float,
    retry: RetryPolicy,
) -> Runnable:
    """Return a chat model for `slot` (a validated chat slot from copilot.llm.slots).

    `request_timeout` bounds a single HTTP request. Retries multiply it, so a total
    deadline across attempts is the caller's job.

    Provider choice: "ollama" uses Ollama's native API, which returns the thinking
    text; "openai" uses any OpenAI-compatible server (for example vLLM). Slots
    without a `provider` field default to "ollama".
    """
    provider = getattr(slot, "provider", "ollama")
    base_url = str(slot.base_url).rstrip("/")

    if provider == "ollama":
        model: BaseChatModel = ChatOllama(
            model=slot.model,
            # The native API lives at the server root, not under /v1.
            base_url=base_url.removesuffix("/v1"),
            temperature=temperature,
            num_predict=max_tokens,
            # Unlike /v1, the native API honours num_ctx, so the slot's context
            # length is enforced per request instead of relying on server env vars.
            num_ctx=slot.context_length,
            reasoning=thinking,
            client_kwargs={"timeout": request_timeout},
        )
    elif provider == "openai":
        model = ChatOpenAI(
            model=slot.model,
            base_url=base_url,
            api_key="not-needed",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="high" if thinking else "none",
            timeout=request_timeout,
            # Retries come from `retry` below; disable the SDK's own so they don't stack.
            max_retries=0,
        )
    else:
        raise ValueError(f"Unknown provider {provider!r}; expected 'ollama' or 'openai'.")

    return model.with_retry(
        retry_if_exception_type=retry.should_retry,
        stop_after_attempt=retry.max_attempts,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": retry.initial_wait, "max": retry.max_wait},
    )
