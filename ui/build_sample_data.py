"""Build ui/data/sample_papers.json: the 10 latest arXiv papers in the paper store, with sample cards.

    uv run python -m ui.build_sample_data

The paper metadata (title, authors, abstract, sections) is read from the paper store as it is.
The cards are hand-written placeholders until card extraction runs on these papers: each
lists only entities named in the paper's abstract, and only result figures the abstract
states word for word. Every card is validated against PaperCardModel, so the real cards
(MergedPaper rows) can replace these without changing the UI.
"""

import json
import sqlite3
from pathlib import Path

from copilot.cards.schema import PaperCardModel

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "ingest" / "queue.db"
OUT_PATH = Path(__file__).resolve().parent / "data" / "sample_papers.json"
LATEST = 10


def entities(*groups: list[str] | str) -> list[dict]:
    """Each argument is one entity: a name, or a list of names (scoring name first)."""
    return [{"names": [g] if isinstance(g, str) else list(g)} for g in groups]


def result(dataset: str, metric: str, method: str, raw_value: str) -> dict:
    return {"dataset": dataset, "metric": metric, "method": method, "raw_value": raw_value}


# Keyed by arXiv ID. Written from each abstract only; see the module docstring.
CARDS = {
    "2610.37657": {
        "tasks": entities("prompt injection defense", "instruction-data separation"),
        "datasets": [],
        "methods": entities(["Heads of Security", "HoS"], "SecAlign++", "ASIDE", "RL-Hammer", "Claudini",
                            "Llama-3.3-70B-Instruct", "GPT-OSS-20B"),
        "metrics": entities(["attack success rate", "ASR"], "utility"),
        "results": [
            result("RL-Hammer attack", "ASR", "HoS", "10%"),
            result("RL-Hammer attack", "ASR", "SecAlign", "85%"),
            result("RL-Hammer attack", "ASR", "GPT-OSS-20B", "97%"),
        ],
    },
    "2610.37651": {
        "tasks": entities("prompt optimization", "self-improving agents"),
        "datasets": entities("GoBrowse", "ALFWorld", "AppWorld", "τ-bench"),
        "methods": entities("GEPA-Curator", "GEPA", "ACE", "task curation", "prior success retrieval",
                            "Gemini 3.1 Pro", "Claude Opus 5"),
        "metrics": entities("mean performance"),
        "results": [],  # the abstract gives only "gains of up to 20.7%", with no benchmark named
    },
    "2610.37649": {
        "tasks": entities("scientific discovery", "physical law discovery"),
        "datasets": entities("LongChang"),
        "methods": entities("GPT-6-Astra"),
        "metrics": entities("discovery rate"),
        "results": [result("LongChang", "discovery rate", "GPT-6-Astra", "16.7%")],
    },
    "2610.37645": {
        "tasks": entities("long-context modeling", "needle-in-a-haystack retrieval", "long-context QA"),
        "datasets": entities("RULER", "LongBench"),
        "methods": entities(["MERLIN", "MomEnt Routing for multi-state LINear attention"],
                            ["Multi-State Linear Attention", "MSLA"], "Gated DeltaNet-1.3B", "Mamba-2-780M"),
        "metrics": entities("improvement over baselines"),
        "results": [
            result("LongBench", "improvement over baselines", "MERLIN (Gated DeltaNet-1.3B)", "13.5%"),
            result("LongBench", "improvement over baselines", "MERLIN (Mamba-2-780M)", "12.0%"),
            result("RULER and long-context QA", "improvement over baselines", "MERLIN (Gated DeltaNet-1.3B)", "8.3%"),
            result("RULER and long-context QA", "improvement over baselines", "MERLIN (Mamba-2-780M)", "7.0%"),
        ],
    },
    "2610.37642": {
        "tasks": entities("efficient inference", "language agents"),
        "datasets": [],
        "methods": entities("metacognitive controller", "diffusion language models"),
        "metrics": entities("inference FLOPs", "task accuracy"),
        "results": [result("five agent benchmarks, average", "FLOPs reduction vs. most efficient baseline",
                           "metacognitive controller", "20.6%")],
    },
    "2610.37624": {
        "tasks": entities("group decision making", "memory for multi-agent discussion"),
        "datasets": entities("CrossProfile"),
        "methods": entities("rarest-first selection", "relevance selection", "Qwen2.5-7B", "Llama-3.1-8B"),
        "metrics": entities("accuracy"),
        "results": [],  # the abstract reports ranges ("23 to 31 points"), not single figures
    },
    "2610.37609": {
        "tasks": entities("agentic reinforcement learning", "task scheduling"),
        "datasets": entities("BFCL-v3", ["τ-Bench", "tau-Bench"], "ACEBench-Agent"),
        "methods": entities(["EnvSched", "Environment-aware Scheduling"], "random task selection", "Qwen3-8B"),
        "metrics": entities("average score"),
        "results": [result("BFCL-v3, τ-Bench, ACEBench-Agent", "average score after 40 steps",
                           "EnvSched (Qwen3-8B)", "48.81")],
    },
    "2610.37494": {
        "tasks": entities("semantic factor analysis", "psychometric scale analysis"),
        "datasets": entities(["IPIP Big-Five Factor Markers",
                              "International Personality Item Pool Big-Five Factor Markers"]),
        "methods": entities("semanticfa", "Qwen3-Embedding-8B", "parallel analysis"),
        "metrics": entities("factor congruence", ["correlation with raw responses", "r"]),
        "results": [
            result("IPIP Big-Five Factor Markers", "factor congruence, Openness", "semanticfa", "0.92"),
            result("IPIP Big-Five Factor Markers", "factor congruence, Agreeableness", "semanticfa", "0.52"),
            result("IPIP Big-Five Factor Markers", "correlation with raw responses", "semanticfa", "0.46"),
        ],
    },
    "2610.37484": {
        "tasks": entities("AI alignment", "ecological reasoning"),
        "datasets": entities("coupled welfare corpus", "MMLU"),
        "methods": entities("coupled-welfare continued pretraining", "QLoRA", "Llama-3.3-70B", "Qwen3.6-27B"),
        "metrics": entities("harmful decisions out of 300", "MMLU accuracy"),
        "results": [
            result("pressure ladder, highest pressure", "harmful decisions out of 300", "Llama-3.3-70B", "70"),
            result("pressure ladder, highest pressure", "harmful decisions out of 300",
                   "Llama-3.3-70B + coupled welfare", "0"),
        ],
    },
    "2610.37481": {
        "tasks": entities("training data attribution", "safety degradation detection", "emergent misalignment"),
        "datasets": entities("Tulu 3", "SafeTulu"),
        "methods": entities("SimpleLoRIF", "LLM-as-a-judge", "GCG"),
        "metrics": entities("AUC", ["attack success rate", "ASR"]),
        "results": [
            result("emergent misalignment", "AUC", "SimpleLoRIF", "0.865"),
            result("SafeTulu", "GCG attack success rate reduction", "SimpleLoRIF filtering", "22%"),
        ],
    },
}


def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT paper_json FROM papers WHERE source = 'arxiv' ORDER BY id DESC LIMIT ?", (LATEST,)
    ).fetchall()
    papers = []
    for (paper_json,) in rows:
        paper = json.loads(paper_json)
        card = PaperCardModel.model_validate({**CARDS[paper["id"]], "code_url": None})
        papers.append({
            "paper": {key: paper[key] for key in ("source", "id", "version", "title", "authors", "abstract",
                                                  "fetched_at")}
                     | {"sections": len(paper["sections"])},
            "card_source": "sample",  # "sample": hand-written from the abstract; "model": card extraction
            "card": card.model_dump(mode="json"),
        })
    store_size = conn.execute("SELECT COUNT(*) FROM papers WHERE source = 'arxiv'").fetchone()[0]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"store_size": store_size, "papers": papers}
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(papers)} papers to {OUT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
