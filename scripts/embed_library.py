"""Embed every library paper's structure-aware chunks once, so `read_paper` never pays for it.

Usage: uv run python scripts/embed_library.py

Resumable: vectors go into the embedding store (evals/cache/embeddings/), keyed by model and
chunk ID, and chunks already stored are skipped. Prints progress per 20 papers.
"""

import time

from copilot.retrieval.chunking import structure_chunks
from copilot.retrieval.dense import embed_chunks
from copilot.tools.library import library


def main() -> None:
    papers = library().papers
    start = time.monotonic()
    total = 0
    for i, paper in enumerate(papers, 1):
        chunks = structure_chunks(paper)
        embed_chunks(chunks)
        total += len(chunks)
        if i % 20 == 0 or i == len(papers):
            print(f"{i}/{len(papers)} papers, {total} chunks, {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
