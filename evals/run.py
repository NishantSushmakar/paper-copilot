"""Run one system on one frozen subset and save the results.

    uv run python -m evals.run --task qasper --system oracle_copy --subset dev

Writes evals/results/<run_id>/:
    results.jsonl   one row per question (written as the run goes)
    summary.json    means with 95% bootstrap CIs, overall and per question type,
                    plus provenance (git commit, models, settings, subset hash, date)

The test subset is guarded: it needs --confirm-test, so it can't be run by accident.

Systems that call an LLM describe their settings in an optional `system_settings`
dict in their systems module, e.g.
    system_settings = {"closed_book": {"slot": "MEDIUM", "temperature": 0.0, ...}}
The "slot" entry decides which model slots go into provenance; everything else is
recorded as-is. Systems without an entry (the fake ones) record no slots or settings.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from copilot.infra.provenance import collect_provenance
from copilot.llm.slots import config as slot_config
from copilot.llm.slots import get_chat_slot
from evals.adapters.qasper import load_subset_split
from evals.harness import run_all, summarise
from evals.metrics.bootstrap import bootstrap_ci
from evals.systems import qasper as qasper_systems

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
SUBSETS_DIR = EVALS_DIR / "subsets"
SEED = 42

TASKS = {
    "qasper": {
        "load": load_subset_split,
        "systems": qasper_systems.system_dict,
        "settings": getattr(qasper_systems, "system_settings", {}),
    },
}


def make_run_id(task: str, system: str, subset: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{task}_{system}_{subset}"


def add_cis(summary: dict, rows: list, seed: int) -> dict:
    """Put a bootstrap CI next to the overall mean and every per-type mean."""
    all_scores = [row.outcome.f1_score for row in rows]
    summary["mean_f1_ci"] = bootstrap_ci(all_scores, seed=seed).model_dump()
    for question_type, group in summary["by_type"].items():
        scores = [r.outcome.f1_score for r in rows if r.identity.question_type == question_type]
        group["ci"] = bootstrap_ci(scores, seed=seed).model_dump()
    return summary


def print_summary(summary: dict) -> None:
    def fmt(ci: dict) -> str:
        return f"{ci['mean']:.3f}  [{ci['lower']:.3f}, {ci['upper']:.3f}]  n={ci['n']}"

    print(f"\n  F1 overall      {fmt(summary['mean_f1_ci'])}")
    for question_type, group in summary["by_type"].items():
        print(f"     {question_type:<12} {fmt(group['ci'])}")
    print(f"  errors          {summary['errors']} {summary['error_types'] or ''}")
    if summary["n_with_response"]:
        print(f"  truncation rate {summary['truncation_rate']:.1%}")
        print(f"  cache-hit rate  {summary['cache_hit_rate']:.1%}")
        print(f"  mean latency    {summary['mean_latency_ms']:.0f} ms")
        print(f"  tokens          {summary['total_prompt_tokens']} in / {summary['total_completion_tokens']} out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--system", required=True)
    parser.add_argument("--subset", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke tests; not for reported numbers)")
    parser.add_argument("--confirm-test", action="store_true", help="required to run on the test subset")
    args = parser.parse_args()

    task = TASKS[args.task]
    if args.system not in task["systems"]:
        parser.error(f"unknown system {args.system!r}; choose from {sorted(task['systems'])}")
    if args.subset == "test" and not args.confirm_test:
        parser.error("the test subset is run once per final system; add --confirm-test if that's what this is")

    questions = task["load"](args.subset)
    if args.limit:
        questions = questions[: args.limit]

    settings = task["settings"].get(args.system, {})
    slot_names = [settings["slot"]] if "slot" in settings else []
    subset_path = SUBSETS_DIR / f"{args.task}_{args.subset}.json"
    provenance = collect_provenance(
        slots={name: get_chat_slot(slot_config, name) for name in slot_names},
        settings=settings,
        subset_path=subset_path,
        seed=SEED,
    )
    if provenance["git"]["dirty"]:
        print("warning: uncommitted changes; the commit hash alone won't reproduce this run", file=sys.stderr)

    run_id = make_run_id(args.task, args.system, args.subset)
    run_dir = RESULTS_DIR / run_id
    print(f"run {run_id}: {len(questions)} questions -> {run_dir.relative_to(EVALS_DIR.parent)}")

    start = time.monotonic()
    rows = run_all(args.system, questions, run_dir / "results.jsonl")
    wall_seconds = time.monotonic() - start

    summary = add_cis(summarise(rows), rows, SEED)
    output = {
        "run_id": run_id,
        "task": args.task,
        "system": args.system,
        "subset": args.subset,
        "limit": args.limit,
        "wall_seconds": round(wall_seconds, 2),
        "summary": summary,
        "provenance": provenance,
    }
    (run_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    print_summary(summary)
    print(f"  wall time       {wall_seconds:.1f} s")


if __name__ == "__main__":
    main()
