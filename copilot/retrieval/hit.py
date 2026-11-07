"""What every retriever returns: the top-k chunks of a paper, best first."""

from collections.abc import Sequence
from dataclasses import dataclass

from copilot.retrieval.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int  # 1 = best


def ranked(chunks: Sequence[Chunk], scores: Sequence[float], k: int) -> list[Hit]:
    """The k best chunks by score. Ties keep the paper's order, so a run is reproducible and a
    question that matches nothing falls back to the paper's first k chunks."""
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:k]
    return [Hit(chunks[i].chunk_id, float(scores[i]), rank) for rank, i in enumerate(order, 1)]

