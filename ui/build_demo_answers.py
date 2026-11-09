"""Pre-compute the demo's suggested answers with the real pipeline (ui/answers.py).

    uv run python -m ui.build_demo_answers

Three questions per paper in ui/data/sample_papers.json, answered by dense k = 5 + MEDIUM, as the
model wrote them. Needs Ollama. Writes ui/data/demo_answers.json after every answer and skips
questions it already has, so an interrupted run resumes; repeated calls come from the LLM cache.
"""

import json
import time
from pathlib import Path

from ui.answers import answer

UI_DIR = Path(__file__).resolve().parent
PAPERS_PATH = UI_DIR / "data" / "sample_papers.json"
OUT_PATH = UI_DIR / "data" / "demo_answers.json"

QUESTIONS = {
    "2610.37657": ["How does HoS stop injected instructions?",
                   "How does it compare with SecAlign on adaptive attacks?",
                   "Does HoS hurt the model's normal performance?"],
    "2610.37651": ["Why not learn from every failed rollout?",
                   "What are the two components of GEPA-Curator?",
                   "How much does curation improve results?"],
    "2610.37649": ["How are the altered worlds created?",
                   "How well do the models discover the hidden laws?",
                   "Which models were evaluated?"],
    "2610.37645": ["What does MERLIN add to multi-state linear attention?",
                   "How much does it improve on LongBench?",
                   "Does it make inference slower?"],
    "2610.37642": ["What does the controller decide during generation?",
                   "How much compute does it save?",
                   "Which benchmarks were used?"],
    "2610.37624": ["Why does relevance-based memory fail?",
                   "How was the CrossProfile benchmark built?",
                   "Which models were tested?"],
    "2610.37609": ["How does EnvSched choose which tasks to run?",
                   "Which benchmarks show the gains?",
                   "Does it reduce the number of rollouts needed?"],
    "2610.37494": ["What does semanticfa do without survey responses?",
                   "How well do the semantic factors match human data?",
                   "Which embedding model was used?"],
    "2610.37484": ["What is in the coupled welfare corpus?",
                   "What is the pressure ladder?",
                   "Did the corpus reduce harmful decisions?"],
    "2610.37481": ["How does SimpleLoRIF find harmful training data?",
                   "How well does it detect emergent misalignment?",
                   "Does filtering hurt the model's helpfulness?"],
}


def main() -> None:
    ids = [item["paper"]["id"] for item in json.loads(PAPERS_PATH.read_text(encoding="utf-8"))["papers"]]
    assert set(ids) == set(QUESTIONS), "every demo paper needs its questions"
    done = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    for paper_id in ids:
        for question in QUESTIONS[paper_id]:
            if any(a["q"] == question for a in done.get(paper_id, [])):
                continue
            start = time.perf_counter()
            result = answer(paper_id, question)
            done.setdefault(paper_id, []).append(result)
            OUT_PATH.write_text(json.dumps(done, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            cited = sum(bool(labels) for _, labels in result["sentences"])
            print(f"{paper_id} {time.perf_counter() - start:5.1f}s  {cited}/{len(result['sentences'])} cited"
                  f"{'  ABSTAINED' if result['abstained'] else ''}  {question}", flush=True)


if __name__ == "__main__":
    main()
