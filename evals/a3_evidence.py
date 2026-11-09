"""A3 evidence F1 of a QASPER rag run: do the cited chunks point at the gold evidence paragraphs?

    uv run python -m evals.a3_evidence --run 20261107-152255_qasper_rag_dense_k5_dev
    uv run python -m evals.a3_evidence --run <a test run> --confirm-test

Each answer is re-parsed from its stored raw text; its cited chunks become predicted paragraphs
(evals/metrics/evidence.py). Every question counts, abstentions with no evidence included.
Writes evals/results/<stamp>_a3_evidence_<run id>/rows.jsonl and summary.json.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from copilot.answer.grounded import parse
from copilot.infra.provenance import collect_provenance
from copilot.ingest.qasper_source import load_qasper_papers
from copilot.retrieval.chunking import structure_chunks
from evals.adapters.qasper import QASPERQuestion, load_subset_split
from evals.metrics.bootstrap import bootstrap_ci
from evals.metrics.evidence import THRESHOLD, evidence_f1, predicted_paragraphs

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
SEED = 42


def paragraphs(question: QASPERQuestion) -> list[str]:
    """The paper's paragraphs as QASPER stores them, so gold evidence matches them exactly."""
    return [question.abstract] + [p for section in question.paper_text for p in section.paragraphs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a QASPER rag run ID under evals/results/")
    parser.add_argument("--confirm-test", action="store_true", help="required to score a test-subset run")
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    subset = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["subset"]
    if subset == "test" and not args.confirm_test:
        parser.error("this run is on the test subset; add --confirm-test")
    questions = {q.question_id: q for q in load_subset_split(subset)}
    papers = {paper.id: paper for paper in load_qasper_papers()}
    chunk_text: dict[str, str] = {}

    rows, gold_strings, gold_matched = [], 0, 0
    for line in open(run_dir / "results.jsonl", encoding="utf-8"):
        result = json.loads(line)
        question = questions[result["identity"]["question_id"]]
        details = result["details"]
        cited = []
        if details is not None and not details["abstained"]:
            if not any(c.startswith(f"qasper:{question.paper_id}v") for c in chunk_text):
                chunk_text.update({c.chunk_id: c.text for c in structure_chunks(papers[question.paper_id])})
            parsed = parse(details["raw"], details["retrieved"])
            cited = list(dict.fromkeys(c for s in parsed.sentences for c in s.chunk_ids))
        paper_paragraphs = paragraphs(question)
        predicted = predicted_paragraphs(paper_paragraphs, [chunk_text[c] for c in cited])
        annotators = [answer.evidence for answer in question.answers]
        known = set(paper_paragraphs)
        gold_strings += sum(len(a) for a in annotators)
        gold_matched += sum(g in known for a in annotators for g in a)
        rows.append({"question_id": question.question_id, "question_type": question.question_type,
                     "cited_chunks": len(cited), "predicted_paragraphs": len(predicted),
                     "evidence_f1": evidence_f1(predicted, annotators)})

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row["evidence_f1"])
    summary = {
        "run": args.run, "subset": subset, "n": len(rows), "threshold": THRESHOLD,
        "evidence_f1": bootstrap_ci([r["evidence_f1"] for r in rows], seed=SEED).model_dump(),
        "by_type": {t: bootstrap_ci(v, seed=SEED).model_dump() for t, v in sorted(by_type.items())},
        "mean_predicted_paragraphs": sum(r["predicted_paragraphs"] for r in rows) / len(rows),
        "gold_strings_that_are_paragraphs": f"{gold_matched}/{gold_strings}",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}_a3_evidence_{args.run}"
    out.mkdir()
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    summary["provenance"] = collect_provenance(slots={}, settings={"threshold": THRESHOLD, "evidence": "cited chunks"},
                                               subset_path=EVALS / "subsets" / f"qasper_{subset}.json", seed=SEED)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ci = summary["evidence_f1"]
    print(f"{args.run}: evidence F1 {ci['mean']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] n={len(rows)} -> {out}")
    for t, c in summary["by_type"].items():
        print(f"  {t:12} {c['mean']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}] n={c['n']}")
    print(f"  predicted paragraphs per answer {summary['mean_predicted_paragraphs']:.2f};"
          f" gold strings that are paper paragraphs {summary['gold_strings_that_are_paragraphs']}")


if __name__ == "__main__":
    main()
