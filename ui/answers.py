"""Cited answers for the Research Guru demo, from the real pipeline.

An ingested arXiv paper is split into structure-aware chunks, the 5 chunks closest to the
question are retrieved (dense), and MEDIUM writes an answer citing them: the same
configuration as the QASPER test run (`rag_dense_k5` in evals/systems/qasper.py).
Nothing here edits the model's answer.
"""

from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_DB = ROOT / "data" / "ingest" / "queue.db"
SYSTEM = "rag_dense_k5"


@cache
def papers() -> dict:
    """The newest stored version of every ingested paper, by arXiv ID."""
    from copilot.cards.backfill import stored_papers

    newest = {}
    for paper in stored_papers(QUEUE_DB):  # newest version first
        newest.setdefault(paper.id, paper)
    return newest


@cache
def chunks(paper_id: str) -> tuple:
    from copilot.retrieval.chunking import structure_chunks

    return tuple(structure_chunks(papers()[paper_id]))


def passage_text(chunk) -> str:
    """A chunk's text without the section heading it starts with."""
    text = chunk.text
    if chunk.heading and text.startswith(chunk.heading):
        text = text[len(chunk.heading):]
    return text.strip()


def answer(paper_id: str, question: str) -> dict:
    """{"q", "abstained", "sentences": [[text, [labels]]], "passages": [{"label", "section", "text"}]}"""
    from copilot.answer.grounded import answer_question
    from copilot.retrieval.dense import dense_search
    from evals.systems.qasper import system_settings

    settings = system_settings[SYSTEM]
    paper_chunks = chunks(paper_id)
    grounded, _ = answer_question(
        question, paper_chunks, dense_search, settings["k"],
        slot=settings["slot"], temperature=settings["temperature"], max_tokens=settings["max_tokens"],
        thinking=settings["thinking"], prompt_version=settings["prompt_version"],
        request_timeout=settings["request_timeout_s"],
    )
    by_id = {chunk.chunk_id: chunk for chunk in paper_chunks}
    k = len(grounded.retrieved)
    return {
        "q": question,
        "abstained": grounded.abstained,
        "sentences": [[s.text, [n for n in s.labels if 1 <= n <= k]] for s in grounded.sentences],
        "passages": [{"label": i + 1, "section": by_id[cid].heading or "", "text": passage_text(by_id[cid])}
                     for i, cid in enumerate(grounded.retrieved)],
    }
