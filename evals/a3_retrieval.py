"""A3 retrieval eval: every chunker x retriever on a frozen QASPER subset (reports/a03.md §3).

    uv run python -m evals.a3_retrieval --subset dev
    uv run python -m evals.a3_retrieval --subset dev --limit 10   # smoke test, not for reporting

Retrieval is within the question's paper. Systems, for each of the two chunkers:
bm25, dense, hybrid (RRF), and two baselines: random (k random chunks, averaged over
RANDOM_DRAWS seeded draws) and first_k (the paper's first k chunks).

Questions without gold text evidence (unanswerable, or every annotator gave none) are left out
and counted. Writes evals/results/<run_id>/:
    results.jsonl   one row per question and system, with every metric
    summary.json    means with 95% bootstrap CIs for recall, MRR and coverage, means for the
                    rest, means per question type, and provenance

The test subset is guarded: it needs --confirm-test.
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from copilot.infra.provenance import collect_provenance
from copilot.ingest.qasper_source import load_qasper_papers
from copilot.retrieval import dense, hybrid
from copilot.retrieval.bm25 import bm25_search
from copilot.retrieval.chunking import CHUNKERS
from evals.adapters.qasper import load_subset_split
from evals.metrics.bootstrap import bootstrap_ci
from evals.metrics.retrieval import KS, MRR_DEPTH, THRESHOLDS, AnnotatorEvidence, best_annotator, words

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
SUBSETS_DIR = EVALS_DIR / "subsets"
SEED = 42
RANDOM_DRAWS = 20
DEPTH = max(KS)

RETRIEVERS = {
    "bm25": lambda question, chunks: [h.chunk_id for h in bm25_search(question, chunks, DEPTH)],
    "dense": lambda question, chunks: [h.chunk_id for h in dense.dense_search(question, chunks, DEPTH)],
    "hybrid": lambda question, chunks: [h.chunk_id for h in hybrid.hybrid_search(question, chunks, DEPTH)],
    "first_k": lambda question, chunks: [c.chunk_id for c in chunks[:DEPTH]],
}
CI_METRICS = ([f"recall@{k}@{t}" for t in THRESHOLDS for k in KS] + [f"mrr@{t}" for t in THRESHOLDS]
              + [f"coverage@{k}" for k in KS])


def annotators(question) -> list[list[str]]:
    """Each answering annotator's evidence strings, if any."""
    return [a.evidence for a in question.answers
            if a.answer_type != "unanswerable" and any(e.strip() for e in a.evidence)]


def score_question(question, chunks, chunker: str) -> list[dict]:
    chunk_words = {c.chunk_id: words(c.text) for c in chunks}
    tokens = {c.chunk_id: c.token_count for c in chunks}
    evidence = [AnnotatorEvidence(ev, chunk_words) for ev in annotators(question)]
    evidence = [ev for ev in evidence if len(ev)]

    def score(ranking: list[str]) -> dict:
        return best_annotator([ev.score(ranking, tokens) for ev in evidence])

    rows = []
    for name, retrieve in RETRIEVERS.items():
        start = time.monotonic()
        ranking = retrieve(question.question, chunks)
        rows.append({"retriever": name, "seconds": time.monotonic() - start, "metrics": score(ranking)})

    rng = random.Random(f"{SEED}:{question.question_id}:{chunker}")
    draws = [score(rng.sample([c.chunk_id for c in chunks], min(DEPTH, len(chunks)))) for _ in range(RANDOM_DRAWS)]
    rows.append({"retriever": "random", "seconds": 0.0,
                 "metrics": {m: statistics.mean(d[m] for d in draws) for m in draws[0]}})

    base = {"question_id": question.question_id, "paper_id": question.paper_id,
            "question_type": question.question_type, "chunker": chunker, "n_chunks": len(chunks),
            "n_annotators": len(evidence)}
    return [base | row | {"system": f"{chunker}/{row['retriever']}"} for row in rows]


def summarise(rows: list[dict]) -> dict:
    summaries = {}
    for system in dict.fromkeys(r["system"] for r in rows):
        group = [r for r in rows if r["system"] == system]
        metrics = list(group[0]["metrics"])
        summary = {"n": len(group), "mean_seconds": statistics.mean(r["seconds"] for r in group)}
        summary["metrics"] = {
            m: (bootstrap_ci([r["metrics"][m] for r in group], seed=SEED).model_dump() if m in CI_METRICS
                else {"mean": statistics.mean(r["metrics"][m] for r in group)})
            for m in metrics
        }
        summary["by_type"] = {
            qtype: {"n": len(of_type), **{m: statistics.mean(r["metrics"][m] for r in of_type) for m in metrics}}
            for qtype in sorted({r["question_type"] for r in group})
            for of_type in [[r for r in group if r["question_type"] == qtype]]
        }
        summaries[system] = summary
    return summaries


def print_table(summaries: dict) -> None:
    def cell(system: str, metric: str) -> str:
        return f"{summaries[system]['metrics'][metric]['mean']:.3f}"

    for t in THRESHOLDS:
        print(f"\nhit = one chunk holds >= {t}% of a gold string's words")
        print(f"{'system':32}" + "".join(f"{f'R@{k}':>8}" for k in KS) + f"{'MRR@' + str(MRR_DEPTH):>9}")
        for system in summaries:
            print(f"{system:32}" + "".join(f"{cell(system, f'recall@{k}@{t}'):>8}" for k in KS)
                  + f"{cell(system, f'mrr@{t}'):>9}")
    print("\ncoverage of gold words, and tokens retrieved")
    print(f"{'system':32}" + "".join(f"{f'cov@{k}':>8}" for k in KS) + "".join(f"{f'tok@{k}':>8}" for k in KS))
    for system in summaries:
        print(f"{system:32}" + "".join(f"{cell(system, f'coverage@{k}'):>8}" for k in KS)
              + "".join(f"{summaries[system]['metrics'][f'tokens@{k}']['mean']:>8.0f}" for k in KS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke tests; not for reported numbers)")
    parser.add_argument("--confirm-test", action="store_true", help="required to run on the test subset")
    args = parser.parse_args()
    if args.subset == "test" and not args.confirm_test:
        parser.error("the test subset is run once per final system; add --confirm-test if that's what this is")

    questions = load_subset_split(args.subset)
    if args.limit:
        questions = questions[: args.limit]
    papers = {p.id: p for p in load_qasper_papers(("train", "dev", "test"))}
    scored = [q for q in questions if annotators(q)]

    settings = {
        "chunkers": list(CHUNKERS), "retrievers": [*RETRIEVERS, "random"], "ks": list(KS),
        "thresholds": list(THRESHOLDS), "mrr_depth": MRR_DEPTH, "random_draws": RANDOM_DRAWS,
        "embed_model": dense.model_key(), "query_instruction": dense.QUERY_INSTRUCTION,
        "rrf_k": hybrid.RRF_K, "fusion_depth": hybrid.FUSION_DEPTH,
    }
    provenance = collect_provenance(slots={}, settings=settings,
                                    subset_path=SUBSETS_DIR / f"qasper_{args.subset}.json", seed=SEED)
    if provenance["git"]["dirty"]:
        print("warning: uncommitted changes; the commit hash alone won't reproduce this run", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_a3_retrieval_{args.subset}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True)
    print(f"run {run_id}: {len(scored)} of {len(questions)} questions have gold evidence"
          f" -> {run_dir.relative_to(EVALS_DIR.parent)}")

    rows, start = [], time.monotonic()
    with open(run_dir / "results.jsonl", "w") as f:
        for n, question in enumerate(scored, 1):
            paper = papers[question.paper_id]
            for chunker, chunk in CHUNKERS.items():
                for row in score_question(question, chunk(paper), chunker):
                    f.write(json.dumps(row) + "\n")
                    rows.append(row)
            if n % 25 == 0:
                print(f"  {n}/{len(scored)} questions, {time.monotonic() - start:.0f} s")

    summaries = summarise(rows)
    output = {"run_id": run_id, "subset": args.subset, "limit": args.limit,
              "questions": len(questions), "scored": len(scored), "left_out": len(questions) - len(scored),
              "wall_seconds": time.monotonic() - start, "summaries": summaries, "provenance": provenance}
    (run_dir / "summary.json").write_text(json.dumps(output, indent=2))
    print_table(summaries)


if __name__ == "__main__":
    main()
