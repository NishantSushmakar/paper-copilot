"""A3 written question: how much of the gap to the oracle is retrieval, and how much is reading?

    uv run python -m evals.a3_gap

Dev only, from runs that already exist: per question, the oracle's answer F1 (A0), dense k = 5's
answer F1, and whether dense put gold evidence in its top 5 (the retrieval eval's recall@5@80 for
structure-aware chunks, best annotator). Questions are bucketed by that hit:
    missed     no gold evidence paragraph in the top 5: the gap here is retrieval's share
    partial    some of it: both play a part
    all        all of it: the gap here is reading's share (with 4 distractor chunks around it)
    no gold evidence   questions the retrieval eval leaves out (gold unanswerable, mostly)
Each bucket's share of the total gap is the sum of its per-question gaps over the sum of all of them.
Writes evals/results/<stamp>_a3_gap_dev/summary.json.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from copilot.retrieval.chunking import STRUCTURE
from evals.metrics.bootstrap import bootstrap_ci

RESULTS = Path(__file__).resolve().parent / "results"
RAG = "20261107-152255_qasper_rag_dense_k5_dev"
ORACLE = "20261027-171055_qasper_oracle_dev"
RETRIEVAL = "20261107-135633_a3_retrieval_dev"
HIT = "recall@5@80"
SEED = 42
BUCKETS = ("missed", "partial", "all", "no gold evidence")


def answer_f1(run: str) -> dict[str, tuple[str, float]]:
    rows = (json.loads(line) for line in open(RESULTS / run / "results.jsonl", encoding="utf-8"))
    return {r["identity"]["question_id"]: (r["identity"]["question_type"], r["outcome"]["f1_score"]) for r in rows}


def main() -> None:
    rag, oracle = answer_f1(RAG), answer_f1(ORACLE)
    recall = {}
    for line in open(RESULTS / RETRIEVAL / "results.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["chunker"] == STRUCTURE and row["retriever"] == "dense":
            recall[row["question_id"]] = row["metrics"][HIT]

    buckets = defaultdict(list)  # bucket -> [(qid, type, oracle, rag)]
    for qid, (question_type, rag_f1) in rag.items():
        r = recall.get(qid)
        bucket = "no gold evidence" if r is None else "missed" if r == 0 else "all" if r == 1 else "partial"
        buckets[bucket].append((qid, question_type, oracle[qid][1], rag_f1))

    total_gap = sum(o - g for rows in buckets.values() for _, _, o, g in rows)
    print(f"dev, n = {len(rag)}: oracle {sum(o for _, o in oracle.values()) / len(oracle):.3f}, "
          f"dense k5 {sum(f for _, f in rag.values()) / len(rag):.3f}, total gap {total_gap / len(rag):+.3f} per question")
    print(f"{'bucket':17} {'n':>3}  {'oracle':>6}  {'rag':>6}  {'gap [95% CI]':>24}  share of gap")
    summary = {}
    for bucket in BUCKETS:
        rows = buckets[bucket]
        gaps = [o - g for _, _, o, g in rows]
        ci = bootstrap_ci(gaps, seed=SEED)
        share = sum(gaps) / total_gap
        types = defaultdict(int)
        for _, t, _, _ in rows:
            types[t] += 1
        summary[bucket] = {"n": len(rows), "oracle": sum(o for _, _, o, _ in rows) / len(rows),
                           "rag": sum(g for _, _, _, g in rows) / len(rows), "gap": ci.model_dump(),
                           "share_of_gap": share, "types": dict(types)}
        s = summary[bucket]
        print(f"{bucket:17} {len(rows):3}  {s['oracle']:.3f}   {s['rag']:.3f}   "
              f"{ci.mean:+.3f} [{ci.lower:+.3f}, {ci.upper:+.3f}]  {share:6.0%}   {dict(types)}")
    out = RESULTS / f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_a3_gap_dev"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"rag": RAG, "oracle": ORACLE, "retrieval": RETRIEVAL, "hit": HIT,
                                                  "buckets": summary}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
