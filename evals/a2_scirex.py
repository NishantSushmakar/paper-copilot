"""A2 eval: paper cards on SciREX, every strategy x slot, with CIs (reports/a02.md §6).

    uv run python -m evals.a2_scirex --split dev
    uv run python -m evals.a2_scirex --split dev --limit 2 --slots MEDIUM_16K --strategies S1   # smoke test
    uv run python -m evals.a2_scirex --split dev --slots MEDIUM_16K --strategies S1 --chunk-size 4000   # ablation
    uv run python -m evals.a2_scirex --split test --confirm-test                            # once, at the end

For each slot, then each strategy (S1 before S2, so S2's first attempt is S1's cached
call), every paper is chunked, every chunk extracted, and the chunk cards merged.
Writes evals/results/<run_id>/:
    chunks.jsonl   one row per chunk (the ChunkOutcome, or an error row), written as it goes
    papers.jsonl   one row per paper and strategy x slot: the MergedPaper and its counts
    summary.json   per strategy x slot: micro P/R/F1 with 95% bootstrap CIs per field and
                   overall, the recall ceiling, validity, retries, failures, latency, plus
                   provenance

Reruns are cheap: every model call is cached, so a crashed run can simply be started
again. Latency of a cached call is the latency recorded when it was first made.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from langsmith import tracing_context
from langsmith.run_trees import get_cached_client

from copilot.cards import extract
from copilot.cards.chunking import chunk_text
from copilot.cards.merge import ChunkFailure, merge_paper
from copilot.infra.jsonl import append_jsonl
from copilot.infra.provenance import collect_provenance, file_sha256
from copilot.infra.tracing import enable_langsmith
from copilot.llm.slots import config as slot_config
from copilot.llm.slots import get_chat_slot
from evals.adapters.scirex import load_split, split_path
from evals.metrics.scirex import FIELDS, Counts, bootstrap_scores, score_paper, scores, total

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
SEED = 42
CHUNK_SIZE = 4000  # tokens (reports/a02.md §6); other sizes for an ablation with --chunk-size
CHUNK_OVERLAP = 100

STRATEGIES = {"S1": extract.extract_s1, "S2": extract.extract_s2, "S3": extract.extract_s3}

logger = logging.getLogger("evals.a2_scirex")


def run_paper(
    doc, strategy: str, slot: str, prompt_version: str, merge_rule: str, chunk_size: int, chunks_path: Path
) -> dict:
    """Chunk, extract and merge one paper; returns its papers.jsonl row."""
    chunks = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    outcomes, errors = [], []
    for chunk in chunks:
        try:
            outcome = STRATEGIES[strategy](
                chunk.text, slot=slot, paper_id=doc.doc_id, chunk_index=chunk.index, prompt_version=prompt_version
            )
        except Exception as exc:  # the call itself failed (server down after retries): not a model failure
            logger.error("%s %s %s chunk %d failed: %r", slot, strategy, doc.doc_id, chunk.index, exc)
            errors.append(ChunkFailure(chunk_index=chunk.index, status="error"))
            append_jsonl(chunks_path, {
                "paper_id": doc.doc_id, "chunk_index": chunk.index, "strategy": strategy, "slot": slot,
                "prompt_version": prompt_version, "status": "error", "error": repr(exc),
            })
            continue
        outcomes.append(outcome)
        append_jsonl(chunks_path, {**outcome.model_dump(mode="json"), "start_word": chunk.start_word,
                                   "end_word": chunk.end_word, "chunk_tokens": chunk.token_count})

    merged = merge_paper(outcomes, merge_rule=merge_rule) if outcomes else None
    card = merged.card if merged and merged.chunks_valid else None
    counts = score_paper(card, doc)
    row = {
        "paper_id": doc.doc_id,
        "strategy": strategy,
        "slot": slot,
        "chunks_total": len(chunks),
        "chunks_valid": merged.chunks_valid if merged else 0,
        "chunk_errors": len(errors),
        "merged": merged.model_dump(mode="json") if merged else None,
        "counts": {field: c.model_dump() for field, c in counts.items()},
    }
    if merged and errors:
        row["merged"]["chunks_total"] = len(chunks)
        row["merged"]["chunk_failures"] += [e.model_dump() for e in errors]
    return row


def summarise(chunk_rows: list[dict], paper_rows: list[dict]) -> dict:
    """One strategy x slot: scores with CIs, then validity, retries, failures and cost."""
    per_field = {field: [Counts(**row["counts"][field]) for row in paper_rows] for field in FIELDS}
    overall = [sum((Counts(**row["counts"][f]) for f in FIELDS), Counts()) for row in paper_rows]

    def scored(per_paper: list[Counts]) -> dict:
        intervals = bootstrap_scores(per_paper, seed=SEED)
        summed = total(per_paper)
        return {
            **{metric: interval.model_dump() for metric, interval in intervals.items()},
            "recall_ceiling": scores(summed).recall_ceiling,
            "counts": summed.model_dump(),
        }

    ok = [r for r in chunk_rows if r["status"] != "error"]
    first_failed = [r for r in ok if r["history"][0]["status"] != "valid"]
    n = len(chunk_rows)
    papers = len(paper_rows)
    return {
        "papers": papers,
        "scores": {"overall": scored(overall), **{field: scored(per_field[field]) for field in FIELDS}},
        "chunks": n,
        "first_attempt_valid_rate": sum(r["status"] != "error" and r["history"][0]["status"] == "valid" for r in chunk_rows) / n,
        "final_valid_rate": sum(r["status"] == "valid" for r in chunk_rows) / n,
        "mean_retries": sum(r["attempts"] - 1 for r in ok) / n,
        "status_counts": dict(Counter(r["status"] for r in chunk_rows)),
        "first_attempt_status_counts": dict(Counter(r["history"][0]["status"] for r in ok)),
        # Which errors did retries fix? First-attempt failures by kind and field, and how they ended.
        "first_failure_fields": dict(Counter(f for r in first_failed for f in r["history"][0]["failed_fields"])),
        "first_failure_outcomes": dict(Counter(f"{r['history'][0]['status']} -> {r['status']}" for r in first_failed)),
        "fence_stripped": sum(r.get("fence_stripped", False) for r in ok),
        "silent_fixes": sum(len(r.get("silent_fixes", [])) for r in ok),
        "papers_all_chunks_valid": sum(r["chunks_valid"] == r["chunks_total"] for r in paper_rows),
        "papers_without_card": sum(r["chunks_valid"] == 0 for r in paper_rows),
        "mean_latency_per_paper_s": sum(r.get("latency_s", 0.0) for r in ok) / papers,
        "model_calls": sum(r["attempts"] for r in ok),
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens": sum(r["completion_tokens"] for r in ok),
        "cache_hit_rate": sum(r["cache_hit"] for r in ok) / len(ok) if ok else None,
    }


def print_table(summaries: dict) -> None:
    def ci(s: dict) -> str:
        return f"{s['value']:.3f} [{s['lower']:.3f}, {s['upper']:.3f}]"

    print(f"\n  {'run':<12} {'overall F1':<22} " + " ".join(f"{f:<9}" for f in FIELDS)
          + "  1st-valid  valid  retries  cut-off  s/paper")
    for key, s in summaries.items():
        fields = " ".join(f"{s['scores'][f]['f1']['value']:<9.3f}" for f in FIELDS)
        print(f"  {key:<12} {ci(s['scores']['overall']['f1']):<22} {fields}  "
              f"{s['first_attempt_valid_rate']:<9.1%}  {s['final_valid_rate']:<5.1%}  {s['mean_retries']:<7.2f}  "
              f"{s['status_counts'].get('cut_off', 0):<7}  {s['mean_latency_per_paper_s']:.0f}")
    ceiling = next(iter(summaries.values()))["scores"]["overall"]["recall_ceiling"]
    print(f"\n  recall ceiling (gold clusters with at least one mention): {ceiling:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    # 16k slots only: GENERATION's max_tokens is sized for a 16k box and overflows an 8k one.
    parser.add_argument("--slots", nargs="+", default=["SMALL_16K", "MEDIUM_16K"], choices=["SMALL_16K", "MEDIUM_16K"])
    parser.add_argument("--strategies", nargs="+", default=["S1", "S2", "S3"], choices=sorted(STRATEGIES))
    parser.add_argument("--prompt-version", default="v3")
    parser.add_argument("--merge-rule", default="share_name", choices=["share_name", "agree2"])
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="chunk size in tokens (default %(default)s)")
    parser.add_argument("--limit", type=int, help="only the first N papers (smoke tests; not for reported numbers)")
    parser.add_argument("--confirm-test", action="store_true", help="required to run on the test split")
    parser.add_argument("--trace", action="store_true", help="send the model calls to LangSmith")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test:
        parser.error("the test split is run once per final configuration; add --confirm-test if that's what this is")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    docs = load_split(args.split)
    if args.limit:
        docs = docs[: args.limit]
    strategies = [s for s in ("S1", "S2", "S3") if s in args.strategies]  # S1 first: S2 reuses its calls

    settings = {
        "generation": extract.GENERATION,
        "s2": {"max_attempts": extract.MAX_ATTEMPTS, "retry_on": sorted(extract.RETRY_ON),
               "retry_prompt": extract.RETRY_PROMPT_VERSION, "template_margin": extract.TEMPLATE_MARGIN,
               "min_retry_tokens": extract.MIN_RETRY_TOKENS},
        "s3_schema": extract.CARD_SCHEMA,
        "chunking": {"chunk_tokens": args.chunk_size, "overlap_tokens": CHUNK_OVERLAP},
        "prompt_version": args.prompt_version,
        "prompt_sha256": file_sha256(extract.PROMPT_DIR / f"extraction_{args.prompt_version}.txt"),
        "merge_rule": args.merge_rule,
        "strategies": strategies,
    }
    provenance = collect_provenance(
        slots={name: get_chat_slot(slot_config, name) for name in args.slots},
        settings=settings,
        subset_path=split_path(args.split),
        seed=SEED,
    )
    if provenance["git"]["dirty"]:
        print("warning: uncommitted changes; the commit hash alone won't reproduce this run", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_a2_scirex_{args.split}"
    run_dir = RESULTS_DIR / run_id
    chunks_path, papers_path = run_dir / "chunks.jsonl", run_dir / "papers.jsonl"
    print(f"run {run_id}: {len(docs)} papers x {len(strategies)} strategies x {len(args.slots)} slots,"
          f" {args.chunk_size}-token chunks, max_tokens {extract.GENERATION['max_tokens']}"
          f" -> {run_dir.relative_to(EVALS_DIR.parent)}")

    if args.trace:
        enable_langsmith()
    summaries = {}
    start = time.monotonic()
    with tracing_context(enabled=args.trace, metadata={"run_id": run_id, "split": args.split}):
        # Slots outermost: one model stays loaded instead of swapping between them.
        for slot in args.slots:
            for strategy in strategies:
                key = f"{slot}/{strategy}"
                chunk_start = sum(1 for _ in chunks_path.open()) if chunks_path.exists() else 0
                paper_rows = []
                for i, doc in enumerate(docs, 1):
                    row = run_paper(doc, strategy, slot, args.prompt_version, args.merge_rule, args.chunk_size, chunks_path)
                    append_jsonl(papers_path, row)
                    paper_rows.append(row)
                    print(f"  {key} {i}/{len(docs)} {doc.doc_id[:8]}: {row['chunks_valid']}/{row['chunks_total']} chunks valid",
                          flush=True)
                with chunks_path.open() as f:
                    chunk_rows = [json.loads(line) for line in list(f)[chunk_start:]]
                summaries[key] = summarise(chunk_rows, paper_rows)
    wall_seconds = time.monotonic() - start
    if args.trace:
        get_cached_client().flush()

    output = {
        "run_id": run_id,
        "split": args.split,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "wall_seconds": round(wall_seconds, 1),
        "summaries": summaries,
        "provenance": provenance,
    }
    (run_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print_table(summaries)
    print(f"  wall time {wall_seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
