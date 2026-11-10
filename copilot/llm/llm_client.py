import httpx,ollama
from time import monotonic
from datetime import timedelta
from collections.abc import Sequence
from typing import Any, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from copilot.llm.slots import get_chat_slot, config
from copilot.infra.models import build_chat_model,RetryPolicy
import hashlib,json
from copilot.infra.server_info import ollama_digest,ollama_version,server_profile
from copilot.infra.cache import ResponseCache
from langsmith import traceable, get_current_run_tree, tracing_context

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
llm_cache = ResponseCache()

def get_llm_output(output):
    return {
        "text": output.text,
        "thinking": output.thinking,
        "tool_calls": output.tool_calls,
        "finish_reason": output.finish_reason,
        "cache_hit": output.cache_hit,
        "usage_metadata": {
            "input_tokens": output.prompt_tokens,
            "output_tokens": output.completion_tokens,
            "total_tokens": output.prompt_tokens + output.completion_tokens,
        },
    }

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
    tool_calls: list[dict] = []  # each {"name", "args", "id"}; the default lets older cache entries load

def get_cache_key(response):
    "returns a cache key for the llm request"

    payload = json.dumps(response ,sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def key_messages(messages: list) -> list[dict]:
    """Messages as they enter the cache key: role, text content and, for an assistant
    message, each tool call's name and arguments. Tool-call ids are random per run, so
    they are renamed by position (call_1, call_2, ...) and each tool result points to the
    renamed id of its call. Plain {"role", "content"} dicts come out unchanged, so calls
    without tools keep their old keys."""
    renamed = {}
    keyed = []
    for message in convert_to_messages(messages):
        entry = {"role": getattr(message, "role", None) or ROLES.get(message.type, message.type),
                 "content": message.content}

        calls = getattr(message, "tool_calls", None)
        if calls:
            entry["tool_calls"] = []
            for call in calls:
                renamed[call["id"]] = f"call_{len(renamed) + 1}"
                entry["tool_calls"].append({"name": call["name"], "args": call["args"],
                                            "id": renamed[call["id"]]})

        if isinstance(message, ToolMessage):
            if message.tool_call_id not in renamed:
                raise ValueError(f"Tool result points to call {message.tool_call_id!r}, "
                                 "which no earlier assistant message made.")
            entry["tool_call_id"] = renamed[message.tool_call_id]

        keyed.append(entry)
    return keyed


def key_tools(tools: Sequence[Any]) -> list[dict]:
    """Each tool as the JSON the model receives (name, description, parameters schema),
    in the order given: the definitions are pasted into the prompt in that order."""
    return [convert_to_openai_tool(tool) for tool in tools]

@traceable(run_type="llm", name="chat_with_llm", process_outputs=get_llm_output)
def chat_with_llm(
    slot: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    response_format:Optional[Union[str, dict]]=None,
    sample_index: int=0,
    request_timeout: float=30.0,  # seconds per attempt; not in the cache key (it doesn't change the reply)
    tools: Optional[Sequence[Any]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
) -> LLMResponse:


    slot_config = get_chat_slot(config=config, name=slot)
    base_url = str(slot_config.base_url).rstrip("/").removesuffix("/v1")

    if tool_choice is not None and not tools:
        raise ValueError("tool_choice was given without any tools.")
    if tool_choice is not None and getattr(slot_config, "provider", "ollama") == "ollama":
        raise ValueError(f"tool_choice is not supported on Ollama slots ({slot}): ChatOllama "
                         "ignores it, so the model could still answer in plain text.")


    cache_key_request = {
        "messages": key_messages(messages),
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
    # Added only when set, so a call without tools hashes exactly as before.
    if tools:
        cache_key_request["tools"] = key_tools(tools)
    if tool_choice is not None:
        cache_key_request["tool_choice"] = tool_choice

    cache_key = get_cache_key(response=cache_key_request)
    

    final_response = llm_cache.get(cache_key)

    run = get_current_run_tree()

    if run is not None:
        run.add_metadata({
            "ls_provider": getattr(slot_config, "provider", "ollama"),
            "ls_model_name": slot_config.model,
            "cache_hit": final_response is not None,
        })


    if final_response is not None:
        final_response['cache_hit']=True

        return LLMResponse.model_validate(final_response)

    else: 

        model = build_chat_model(
            slot_config,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            request_timeout=request_timeout,
            retry=RetryPolicy(
                max_attempts=4,
                should_retry=should_retry,
            ),
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        start = monotonic()
        with tracing_context(enabled=False):
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
                            tool_calls=[{"name": call["name"], "args": call["args"], "id": call["id"]}
                                        for call in getattr(response, "tool_calls", [])],
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
