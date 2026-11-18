"""Run one model slot on the tool-calling cases and score it.

    uv run python -m evals.run_tool_calls --slot SMALL --split dev
    uv run python -m evals.run_tool_calls --slot MEDIUM --thinking --split dev

Each case: the system prompt, then the question, with both library tools offered. The model
gets up to MAX_TURNS replies; every tool call it makes runs through the registry and its result
goes back as a tool message. If the model replies in text and the case has a scripted user
reply (ambiguous cases), that reply is sent once and the model continues. Scoring is
evals/metrics/tool_calls.py.

Writes evals/results/<run_id>/:
    results.jsonl   one row per case (written as the run goes)
    summary.json    pass rate, valid-call rate and right-outcome rate with 95% bootstrap CIs,
                    per-kind pass rates, tokens and seconds, provenance

The test subset needs --confirm-test: it is run once per frozen configuration.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from copilot.infra.jsonl import append_jsonl
from copilot.infra.provenance import collect_provenance
from copilot.llm.llm_client import chat_with_llm
from copilot.llm.slots import config as slot_config
from copilot.llm.slots import get_chat_slot
from copilot.tools.library import build_registry
from evals.metrics.bootstrap import bootstrap_ci
from evals.metrics.tool_calls import score_case

EVALS_DIR = Path(__file__).resolve().parent
CASES = EVALS_DIR / "data" / "tool_calls_cases.jsonl"
SUBSETS_DIR = EVALS_DIR / "subsets"
RESULTS_DIR = EVALS_DIR / "results"
SEED = 42
MAX_TURNS = 4

# Mine (2026-11-16). Says what the library is and nothing about how to pick a tool: the tool
# descriptions carry that, and they are what this eval tests.
SYSTEM_PROMPT = ("You are Research Guru, an assistant for the user's personal library of research papers. "
                 "Use the tools when a question is about papers in the library; answer general questions directly.")

GENERATION = {
    False: {"temperature": 0.0, "max_tokens": 512},
    True: {"temperature": 0.0, "max_tokens": 2048},  # room for the thinking text
}


def load_cases(split: str) -> list[dict]:
    ids = set(json.loads((SUBSETS_DIR / f"tool_calls_{split}.json").read_text())["case_ids"])
    with CASES.open() as f:
        return [case for case in map(json.loads, f) if case["id"] in ids]


def parse(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def run_case(case: dict, slot: str, thinking: bool, registry) -> dict:
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(case["question"])]
    calls, turns = [], []
    user_reply_sent = False
    tokens = seconds = 0.0
    for _ in range(MAX_TURNS):
        reply = chat_with_llm(slot=slot, messages=messages, thinking=thinking, tools=registry.tools(),
                              **GENERATION[thinking])
        tokens += reply.prompt_tokens + reply.completion_tokens
        seconds += reply.latency.total_seconds()
        turns.append({"text": reply.text, "tool_calls": reply.tool_calls, "finish_reason": reply.finish_reason,
                      "cache_hit": reply.cache_hit})
        if reply.tool_calls:
            messages.append(AIMessage(content=reply.text, tool_calls=reply.tool_calls))
            for call in reply.tool_calls:
                result = registry.execute(call)
                seconds += result.metadata.get("latency", 0.0)
                calls.append({"name": call["name"], "args": call["args"], "error": result.error,
                              "layer": result.metadata.get("layer"), "output": None if result.error else parse(result.content)})
                messages.append(result.to_message(call["id"]))
            continue
        if case.get("simulated_user") and not user_reply_sent:
            messages += [AIMessage(content=reply.text), HumanMessage(case["simulated_user"])]
            user_reply_sent = True
            continue
        break
    return {"id": case["id"], "kind": case["kind"], "question": case["question"], "needs_read": case.get("needs_read", False),
            **score_case(case, calls), "calls": calls, "turns": turns,
            "user_reply_sent": user_reply_sent, "tokens": tokens, "seconds": seconds}


def ci(values: list[bool]) -> dict:
    return bootstrap_ci([float(v) for v in values], seed=SEED).model_dump()


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    by_kind = defaultdict(list)
    for r in ok:
        by_kind[r["kind"]].append(r)
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "pass": ci([r["passed"] for r in ok]),
        "valid_calls": ci([r["valid_calls"] for r in ok]),
        "right_outcome": ci([r["right_outcome"] for r in ok]),
        "by_kind": {k: {"n": len(g), "pass": mean(r["passed"] for r in g)} for k, g in sorted(by_kind.items())},
        "tokens_per_case": mean(r["tokens"] for r in ok) if ok else None,
        "seconds_per_case": mean(r["seconds"] for r in ok) if ok else None,
        "tool_calls_per_case": mean(len(r["calls"]) for r in ok) if ok else None,
        "read_rate_when_needed": (mean(any(c["name"] == "read_paper" for c in r["calls"]) for r in ok if r.get("needs_read"))
                                  if any(r.get("needs_read") for r in ok) else None),
    }


def print_summary(s: dict) -> None:
    fmt = lambda c: f"{c['mean']:.3f}  [{c['lower']:.3f}, {c['upper']:.3f}]  n={c['n']}"
    print(f"\n  pass            {fmt(s['pass'])}")
    print(f"  valid calls     {fmt(s['valid_calls'])}")
    print(f"  right outcome   {fmt(s['right_outcome'])}")
    for kind, g in s["by_kind"].items():
        print(f"     {kind:<16} {g['pass']:.2f}  n={g['n']}")
    if s["read_rate_when_needed"] is not None:
        print(f"  read_paper called when the case needs reading: {s['read_rate_when_needed']:.2f}")
    print(f"  per case: {s['tokens_per_case']:.0f} tokens, {s['seconds_per_case']:.1f} s, "
          f"{s['tool_calls_per_case']:.1f} tool calls   errors: {s['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--confirm-test", action="store_true", help="required to run on the test subset")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("the test subset is run once per frozen configuration; add --confirm-test if that's what this is")

    mode = "think" if args.thinking else "nothink"
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_tool_calls_{args.slot}-{mode}_{args.split}"
    out_dir = RESULTS_DIR / run_id
    results_path = out_dir / "results.jsonl"

    registry = build_registry()
    rows = []
    for case in load_cases(args.split):
        try:
            row = run_case(case, args.slot, args.thinking, registry)
        except Exception as exc:  # a failed case is recorded, not fatal
            row = {"id": case["id"], "kind": case["kind"], "error": repr(exc)}
        rows.append(row)
        append_jsonl(results_path, row)
        print(f"{case['id']:<12} {'error' if 'error' in row else ('PASS' if row['passed'] else 'fail')}"
              + (f"  {row['reasons'][0][:90]}" if row.get("reasons") else ""))

    summary = summarise(rows)
    summary["provenance"] = collect_provenance(
        slots={args.slot: get_chat_slot(config=slot_config, name=args.slot)},
        settings={"thinking": args.thinking, **GENERATION[args.thinking], "max_turns": MAX_TURNS,
                  "system_prompt": SYSTEM_PROMPT},
        subset_path=SUBSETS_DIR / f"tool_calls_{args.split}.json", seed=SEED)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"\n  -> {out_dir.relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    main()
