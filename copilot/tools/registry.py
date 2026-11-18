"""The tool registry: every tool call the agent makes goes through here.

A tool is a LangChain tool (`@tool`, with a Pydantic argument schema). The model
sees only its name, description and schema. The registry adds what the system
needs and the model doesn't (decisions in reports/a04.md §4):

* side effect: yes/no. Side-effect tools need the replay record (run ID + name +
  arguments -> stored result), which isn't built yet, so registering one raises.
* time limit: the tool's own limit. A call runs under the smaller of this and the
  run's remaining budget, which the agent loop passes in.

Errors come in two layers, and both reach the model as one-line JSON with the same
four fields (error, what_was_wrong, retry, what_to_do):

* the registry's: unknown tool, and arguments that don't fit the schema (missing,
  wrong type, extra, out of range). Types are checked strictly: 10.0 is not an int.
* the tool's: it raises `BadArgument` (the model sent something it can't use) or
  `ToolFailed` (the arguments were fine but it couldn't finish), with its own text.

The only error the model doesn't see first is a service that couldn't be reached:
that is retried quietly, and becomes `tool_failed` if the retries fail too.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, ValidationError

RETRY_WAITS = (0.5, 1.0)  # seconds before the 2nd and 3rd attempt at an unreachable service
UNREACHABLE = (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)

# Calls run in worker threads so a time limit can be enforced. Python can't stop a
# thread, so a call that times out keeps running in the background until it ends.
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


class ToolError(Exception):
    """Raised by a tool for an error only it can recognise."""

    kind = "tool_failed"
    retry = False

    def __init__(self, what_was_wrong: str, what_to_do: str):
        super().__init__(what_was_wrong)
        self.what_was_wrong = what_was_wrong
        self.what_to_do = what_to_do


class BadArgument(ToolError):
    """The model sent an argument the tool can't use; it can fix it and call again."""

    kind = "bad_argument"
    retry = True


class ToolFailed(ToolError):
    """The arguments were fine but the tool couldn't finish; calling again won't help."""


class Unreachable(Exception):
    """A service the tool needs couldn't be reached. The registry retries this quietly."""


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    side_effect: bool
    timeout: float  # seconds: this kind of call is broken if it takes longer


@dataclass
class ToolResult:
    content: str                  # the text the model receives in the tool message
    error: Optional[dict] = None  # the structured error, None on success
    metadata: dict = field(default_factory=dict)  # for the trace, never shown to the model

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_message(self, call_id: str) -> ToolMessage:
        return ToolMessage(content=self.content, tool_call_id=call_id)


def error_result(kind: str, what_was_wrong: str, retry: bool, what_to_do: str, **metadata) -> ToolResult:
    error = {"error": kind, "what_was_wrong": what_was_wrong, "retry": retry, "what_to_do": what_to_do}
    return ToolResult(content=json.dumps(error), error=error, metadata=metadata)


def schema_problems(schema: type[BaseModel], args: dict) -> list[str]:
    """Everything wrong with `args` under `schema`, one sentence each; empty if valid."""
    problems = [f"`{name}` is not an argument of this tool" for name in args if name not in schema.model_fields]
    try:
        schema.model_validate(args, strict=True)
    except ValidationError as exc:
        for err in exc.errors():
            name = ".".join(str(part) for part in err["loc"]) or "arguments"
            if err["type"] == "missing":
                problems.append(f"`{name}` is required but was not given")
            else:
                problems.append(f"`{name}`: {err['msg']}; you sent {err['input']!r}")
    return problems


def without_null_optionals(schema: type[BaseModel], args: dict) -> dict:
    """Models often send an explicit null for an optional argument; that means "not given"."""
    fields = schema.model_fields
    return {name: value for name, value in args.items()
            if not (value is None and name in fields and not fields[name].is_required())}


def stringify(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, default=str)


class Registry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool, *, side_effect: bool, timeout: float) -> None:
        if tool.name in self._specs:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")
        if side_effect:
            raise NotImplementedError(
                f"{tool.name!r} has a side effect: build the replay record before registering it.")
        if not (isinstance(tool.args_schema, type) and issubclass(tool.args_schema, BaseModel)):
            raise TypeError(f"{tool.name!r} needs a Pydantic argument schema.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self._specs[tool.name] = ToolSpec(tool=tool, side_effect=side_effect, timeout=timeout)

    def spec(self, name: str) -> ToolSpec:
        return self._specs[name]

    def tools(self) -> list[BaseTool]:
        """The tools to offer the model, in registration order (the order is in the cache key)."""
        return [spec.tool for spec in self._specs.values()]

    @traceable(run_type="tool", name="registry.execute")
    def execute(self, call: dict, budget: Optional[float] = None) -> ToolResult:
        """Run one tool call {"name", "args", "id"}. `budget` is the run's remaining
        seconds (None: no run budget). Never raises for a bad call or a failing tool."""
        result = self._execute(call, budget)
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata({"tool": call.get("name"), "ok": result.ok, **result.metadata})
        return result

    def _execute(self, call: dict, budget: Optional[float]) -> ToolResult:
        name, args = call.get("name"), call.get("args") or {}
        spec = self._specs.get(name)
        if spec is None:
            return error_result(
                "unknown_tool", f"There is no tool called {name!r}.", True,
                f"Call one of: {', '.join(self._specs)}.", layer="registry")

        args = without_null_optionals(spec.tool.args_schema, args)
        problems = schema_problems(spec.tool.args_schema, args)
        if problems:
            return error_result(
                "bad_argument", "; ".join(problems) + ".", True,
                f"Call {name} again with arguments that match its schema.", layer="registry")

        limit = spec.timeout if budget is None else min(spec.timeout, budget)
        if limit <= 0:
            return error_result(
                "tool_failed", "No time is left in this run's budget.", False,
                "Don't call any more tools; answer with what you have.", layer="registry", time_limit=limit)
        return self._run(spec, args, limit)

    def _run(self, spec: ToolSpec, args: dict, limit: float) -> ToolResult:
        name = spec.tool.name
        deadline = time.monotonic() + limit
        start = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            meta = {"attempts": attempts, "time_limit": limit}
            future = _pool.submit(spec.tool.invoke, args)
            try:
                output = future.result(timeout=max(deadline - time.monotonic(), 0))
                meta["latency"] = time.monotonic() - start
                return ToolResult(content=stringify(output), metadata=meta)
            except FutureTimeout:
                return error_result(
                    "tool_failed", f"{name} took longer than its {limit:.1f} s limit.", False,
                    "Don't call it again with the same arguments; continue without it.",
                    layer="tool", timed_out=True, **meta)
            except ToolError as exc:
                return error_result(exc.kind, exc.what_was_wrong, exc.retry, exc.what_to_do, layer="tool", **meta)
            except (Unreachable, *UNREACHABLE) as exc:
                wait = RETRY_WAITS[attempts - 1] if attempts <= len(RETRY_WAITS) else None
                if wait is None or time.monotonic() + wait >= deadline:
                    return error_result(
                        "tool_failed", f"{name} is unavailable at the moment.", False,
                        "Please don't call this tool again; continue without it.",
                        layer="tool", unreachable=True, exception=repr(exc), **meta)
                time.sleep(wait)
            except Exception as exc:  # a bug inside the tool: the model gets a clean error, the trace keeps the cause
                return error_result(
                    "tool_failed", f"{name} hit an internal error.", False,
                    "Don't call it again with the same arguments; continue without it.",
                    layer="tool", exception=repr(exc), **meta)
