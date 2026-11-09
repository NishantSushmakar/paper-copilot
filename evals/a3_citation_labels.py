"""A3 citation judge check: sentence-citation pairs from the dev run, labelled by hand.

    uv run python -m evals.a3_citation_labels           # writes the sheet and the blank labels.csv

Draws from the dev dense k = 5 run (the configuration that went to test), never from test.
Only cited sentences of answers that didn't abstain are in the pool, re-parsed from the stored
raw answers so the current sentence splitter applies. Whole sentences are drawn,
stratified by question type, seed 42, until each type's pair quota is reached; all pairs from
gold-unanswerable questions are taken, since they are the likeliest unsupported ones.

Writes evals/a3_citation_labels/:
    sheet.md     per sentence: the question, the full answer, the sentence, its cited chunks
    items.json   the same, for the labelling app (ui/label_citations.py)
    labels.csv   one row per sentence and one per (sentence, chunk) pair, to fill in by hand

Labels are judged against the original question and answer (reports/a03.md §9):
    sentence row  supports = do all its cited chunks together state the claim (y/n)
    pair row      supports = does this chunk on its own state the claim (y/n); partial is n
                  supports_without_this = do the other cited chunks still state it without
                  this one (y/n), only for sentences with 3 or more citations
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from copilot.answer.grounded import parse
from copilot.ingest.qasper_source import load_qasper_papers
from copilot.retrieval.chunking import structure_chunks
from evals.adapters.qasper import load_subset_split

OUT_DIR = Path(__file__).resolve().parent / "a3_citation_labels"
SOURCE_RUN = Path(__file__).resolve().parent / "results" / "20261107-152255_qasper_rag_dense_k5_dev"
SEED = 42
QUOTA = {"extractive": 20, "abstractive": 13, "yes_no": 10, "unanswerable": 7}
COLUMNS = ["item", "level", "sentence", "question_type", "chunk_label", "n_cited",
           "supports", "supports_without_this", "notes"]


def pool() -> dict[str, list[dict]]:
    """Cited sentences of the source run, grouped by question type, in a stable order."""
    by_type = defaultdict(list)
    for line in open(SOURCE_RUN / "results.jsonl"):
        row = json.loads(line)
        details = row["details"]
        if details["abstained"]:
            continue
        # re-parsed from the stored raw answer, so the current sentence splitter applies
        for index, sentence in enumerate(parse(details["raw"], details["retrieved"]).sentences):
            if sentence.chunk_ids:
                by_type[row["identity"]["question_type"]].append({
                    "question_id": row["identity"]["question_id"], "index": index,
                    "raw": details["raw"], "retrieved": details["retrieved"], **sentence.model_dump()})
    return {t: sorted(s, key=lambda x: (x["question_id"], x["index"])) for t, s in by_type.items()}


def draw(by_type: dict[str, list[dict]]) -> list[dict]:
    rng = random.Random(SEED)
    picked = []
    for question_type, quota in QUOTA.items():
        candidates = by_type.get(question_type, [])
        rng.shuffle(candidates)
        pairs = 0
        for sentence in candidates:
            if pairs >= quota:
                break
            picked.append({**sentence, "question_type": question_type})
            pairs += len(sentence["chunk_ids"])
    return picked


def write(picked: list[dict]) -> None:
    questions = {q.question_id: q for q in load_subset_split("dev")}
    papers = {paper.id: paper for paper in load_qasper_papers()}
    chunk_text = {}
    for paper_id in {questions[s["question_id"]].paper_id for s in picked}:
        chunk_text.update({c.chunk_id: c.text for c in structure_chunks(papers[paper_id])})

    sheet = ["# A3 citation labels (dev, dense k = 5)", "",
             "Label against the original question and answer: a chunk supports a sentence only if it",
             "**states** the claim (a paraphrase counts, an inference doesn't; partial support is n).",
             "Fill in labels.csv; the item IDs below match its rows.", ""]
    rows, items = [], []
    for number, s in enumerate(picked, 1):
        sid = f"S{number:02d}"
        question = questions[s["question_id"]]
        n = len(s["chunk_ids"])
        sheet += [f"## {sid} ({s['question_type']}, {n} cited)", "",
                  f"**Question:** {question.question}", "",
                  f"**Full answer:** {s['raw']}", "",
                  f"**Sentence to judge:** {s['text']}", ""]
        rows.append({"item": sid, "level": "sentence", "sentence": sid, "question_type": s["question_type"],
                     "chunk_label": "all", "n_cited": n, "supports_without_this": "-"})
        items.append({"sentence": sid, "question_type": s["question_type"], "question_id": s["question_id"],
                      "question": question.question, "answer": s["raw"], "text": s["text"], "chunks": []})
        for chunk_id in s["chunk_ids"]:
            label = s["retrieved"].index(chunk_id) + 1
            item = f"{sid}-c{label}"
            items[-1]["chunks"].append({"item": item, "label": label, "chunk_id": chunk_id, "text": chunk_text[chunk_id]})
            sheet += [f"### {item} (chunk [{label}])", "", "> " + chunk_text[chunk_id].replace("\n", "\n> "), ""]
            rows.append({"item": item, "level": "pair", "sentence": sid, "question_type": s["question_type"],
                         "chunk_label": label, "n_cited": n,
                         "supports_without_this": "" if n >= 3 else "-"})
        sheet.append("---\n")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    (OUT_DIR / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    pairs = [r for r in rows if r["level"] == "pair"]
    print(f"{len(picked)} sentences, {len(pairs)} pairs -> {OUT_DIR}")
    for question_type in QUOTA:
        print(f"  {question_type:12} sentences {sum(s['question_type'] == question_type for s in picked):2}"
              f"  pairs {sum(r['question_type'] == question_type for r in pairs):2}")
    print(f"  sentences with 3+ citations: {sum(len(s['chunk_ids']) >= 3 for s in picked)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing labels.csv")
    args = parser.parse_args()
    if (OUT_DIR / "labels.csv").exists() and not args.force:
        parser.error(f"{OUT_DIR / 'labels.csv'} exists and may hold your labels; add --force to overwrite")
    write(draw(pool()))


if __name__ == "__main__":
    main()
