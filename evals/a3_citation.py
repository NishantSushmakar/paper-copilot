"""A3 citation eval: validate the NLI judge on the labelled pairs, then score runs with it.

    uv run python -m evals.a3_citation --validate
    uv run python -m evals.a3_citation --run 20261107-152255_qasper_rag_dense_k5_dev [--limit N]
    uv run python -m evals.a3_citation --run <a test run> --confirm-test

--validate judges the 31 labelled sentences (54 pairs, evals/a3_citation_labels/) with hypothesis
formats A and B and compares the judge with the ground truth (labels_claude.csv, reports/a03.md §9):
"together" labels against the joined chunks, "alone" labels against each chunk, "without this"
labels against the other chunks. My partial labels (labels.csv) are compared too, as a spot check.

--run re-parses every answer of a QASPER run from its stored raw text (the current sentence
splitter), judges each cited sentence, and reports ALCE citation recall and precision per answer,
averaged over answers that didn't abstain, with bootstrap CIs.

Writes evals/results/<stamp>_a3_citation_<validate | run id>/rows.jsonl and summary.json.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from copilot.answer.grounded import parse
from copilot.infra.nli import model_key
from copilot.infra.provenance import collect_provenance
from copilot.ingest.qasper_source import load_qasper_papers
from copilot.llm.slots import config, get_chat_slot
from copilot.retrieval.chunking import structure_chunks
from evals.adapters.qasper import load_subset_split
from evals.metrics.bootstrap import bootstrap_ci
from evals.metrics.citation import (MAX_TOKENS, REWRITE, REWRITE_PROMPT, NLIJudge, agreement, answer_scores,
                                    hypothesis_a, hypothesis_b, judge_sentence)

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
LABEL_DIR = EVALS / "a3_citation_labels"
GOLD = LABEL_DIR / "labels_claude.csv"  # ground truth (my decision, reports/a03.md §9)
HUMAN = LABEL_DIR / "labels.csv"  # my partial labels: a spot check
SEED = 42
FORMATS = {"A": hypothesis_a, "B": hypothesis_b}
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # analysis only, never the rule


def yes(value: str) -> bool | None:
    return {"y": True, "n": False}.get(value)


def settings() -> dict:
    return {"nli_model": model_key(), "nli_max_tokens": MAX_TOKENS, "entailed": "argmax label",
            "rewrite_prompt": REWRITE_PROMPT, "rewrite": REWRITE, "formats": list(FORMATS)}


def write(name: str, rows: list[dict], summary: dict, subset_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}_a3_citation_{name}"
    out.mkdir(parents=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    summary["provenance"] = collect_provenance(slots={"MEDIUM": get_chat_slot(config, "MEDIUM")},
                                               settings=settings(), subset_path=subset_path, seed=SEED)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return out


def validate() -> None:
    items = json.loads((LABEL_DIR / "items.json").read_text(encoding="utf-8"))
    gold = {r["item"]: r for r in csv.DictReader(open(GOLD, encoding="utf-8"))}
    human = {r["item"]: r for r in csv.DictReader(open(HUMAN, encoding="utf-8"))}
    judge = NLIJudge()

    rows, summary = [], {"gold": GOLD.name, "sentences": len(items), "formats": {}}
    pairs = defaultdict(lambda: defaultdict(list))  # format -> kind -> [(gold, pred, type, item, p)]
    for item in items:
        texts = [chunk["text"] for chunk in item["chunks"]]
        row = {"sentence": item["sentence"], "question_type": item["question_type"], "question": item["question"],
               "text": item["text"], "gold_together": gold[item["sentence"]]["supports"], "formats": {}}
        for name, make in FORMATS.items():
            verdict = judge_sentence(judge, make(item["question"], item["text"]), texts)
            row["formats"][name] = verdict.model_dump()
            kind_rows = pairs[name]
            kind_rows["together"].append((yes(gold[item["sentence"]]["supports"]), verdict.supported,
                                          item["question_type"], item["sentence"], verdict.together.p_entailment))
            for i, chunk in enumerate(item["chunks"]):
                label = gold[chunk["item"]]
                kind_rows["alone"].append((yes(label["supports"]), verdict.alone[i].entailed,
                                           item["question_type"], chunk["item"], verdict.alone[i].p_entailment))
                if yes(label["supports_without_this"]) is not None:
                    kind_rows["without"].append((yes(label["supports_without_this"]), verdict.without[i].entailed,
                                                 item["question_type"], chunk["item"], verdict.without[i].p_entailment))
        rows.append(row)

    for name, kinds in pairs.items():
        result = {}
        for kind, entries in kinds.items():
            result[kind] = agreement([g for g, *_ in entries], [p for _, p, *_ in entries])
            by_type = defaultdict(list)
            for g, p, question_type, *_ in entries:
                by_type[question_type].append((g, p))
            result[kind]["by_type"] = {t: agreement([g for g, _ in v], [p for _, p in v]) for t, v in by_type.items()}
            result[kind]["disagreements"] = [item for g, p, _, item, _ in entries if g != p]
            result[kind]["cutoff_sweep"] = {str(t): agreement([g for g, *_ in entries], [e[4] >= t for e in entries])["accuracy"]
                                            for t in SWEEP}
        spot = [(yes(human[item]["supports"]), entries) for item, entries in
                ((item, next(e for k in ("together", "alone") for e in kinds[k] if e[3] == item))
                 for item in human if yes(human[item]["supports"]) is not None)]
        result["human_spot_check"] = agreement([h for h, _ in spot], [e[1] for _, e in spot])
        summary["formats"][name] = result
    human_vs_gold = [(yes(h["supports"]), yes(gold[i]["supports"])) for i, h in human.items() if yes(h["supports"]) is not None]
    summary["human_vs_gold"] = agreement([h for h, _ in human_vs_gold], [g for _, g in human_vs_gold])
    summary["nli_calls"] = judge.calls

    out = write("validate", rows, summary, LABEL_DIR / "items.json")
    print(f"validated on {len(items)} sentences -> {out}")
    print(f"  human spot check vs gold: {summary['human_vs_gold']['accuracy']:.2f} on {summary['human_vs_gold']['n']} items")
    for name, result in summary["formats"].items():
        for kind in ("together", "alone", "without"):
            a = result[kind]
            kappa = "n/a" if a["kappa"] is None else f"{a['kappa']:.2f}"
            print(f"  {name} {kind:9} n={a['n']:2}  accuracy {a['accuracy']:.2f}  kappa {kappa}"
                  f"  tp {a['tp']} fp {a['fp']} fn {a['fn']} tn {a['tn']}")


def score_run(run_id: str, limit: int | None, confirm_test: bool) -> None:
    run_dir = RESULTS / run_id
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    subset = run_summary["subset"]
    if subset == "test" and not confirm_test:
        raise SystemExit("this run is on the test subset; add --confirm-test once the judge is frozen")
    questions = {q.question_id: q for q in load_subset_split(subset)}
    papers = {paper.id: paper for paper in load_qasper_papers()}
    chunk_text: dict[str, str] = {}
    chunked: set[str] = set()
    judge = NLIJudge()

    rows = []
    for line in list(open(run_dir / "results.jsonl", encoding="utf-8"))[:limit]:
        result = json.loads(line)
        details = result["details"]
        if details is None or details["abstained"]:
            continue
        question = questions[result["identity"]["question_id"]]
        if question.paper_id not in chunked:
            chunked.add(question.paper_id)
            chunk_text.update({c.chunk_id: c.text for c in structure_chunks(papers[question.paper_id])})
        parsed = parse(details["raw"], details["retrieved"])
        row = {"question_id": question.question_id, "question_type": question.question_type, "formats": {}}
        for name, make in FORMATS.items():
            verdicts = [judge_sentence(judge, make(question.question, s.text) if s.chunk_ids else "",
                                       [chunk_text[c] for c in s.chunk_ids])
                        for s in parsed.sentences]
            row["formats"][name] = {**answer_scores(verdicts), "sentences": [v.model_dump() for v in verdicts]}
        rows.append(row)

    summary = {"run": run_id, "subset": subset, "answers": len(rows), "nli_calls": judge.calls, "formats": {}}
    for name in FORMATS:
        scores = [r["formats"][name] for r in rows]
        by_type = defaultdict(list)
        for r in rows:
            by_type[r["question_type"]].append(r["formats"][name])
        summary["formats"][name] = {
            "recall": bootstrap_ci([s["recall"] for s in scores], seed=SEED).model_dump(),
            "precision": bootstrap_ci([s["precision"] for s in scores], seed=SEED).model_dump(),
            "truncated_sentences": sum(s["truncated"] for s in scores),
            "by_type": {t: {"n": len(v), "recall": sum(s["recall"] for s in v) / len(v),
                            "precision": sum(s["precision"] for s in v) / len(v)} for t, v in by_type.items()},
        }
    out = write(run_id, rows, summary, EVALS / "subsets" / f"qasper_{subset}.json")
    print(f"{run_id}: {len(rows)} answers -> {out}")
    for name, s in summary["formats"].items():
        r, p = s["recall"], s["precision"]
        print(f"  {name}  recall {r['mean']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]"
              f"  precision {p['mean']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]  truncated {s['truncated_sentences']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true", help="judge the labelled pairs and report agreement")
    group.add_argument("--run", help="a QASPER run ID under evals/results/ to score")
    parser.add_argument("--limit", type=int, help="only the first N rows of the run (smoke tests)")
    parser.add_argument("--confirm-test", action="store_true", help="required to score a test-subset run")
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        score_run(args.run, args.limit, args.confirm_test)


if __name__ == "__main__":
    main()
