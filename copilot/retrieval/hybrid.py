"""Hybrid retrieval: reciprocal rank fusion of the dense and BM25 rankings (Cormack et al., 2009).

A chunk scores 1 / (RRF_K + rank) for each list it is in, and 0 for a list it isn't in. Only
ranks are used, so BM25's and cosine's different scales don't matter. How deep the input lists
go is the caller's choice: a chunk below the cut of one list gets nothing from that list.
"""

from collections.abc import Sequence

from copilot.retrieval.bm25 import bm25_search
from copilot.retrieval.chunking import Chunk
from copilot.retrieval.dense import dense_search
from copilot.retrieval.hit import Hit

RRF_K = 60
# Each retriever hands over its top 50, whatever the final k, and only the fused list is cut at k.
# That is the whole paper for nearly every question (on dev, 5 of 200 questions are on a paper
# with more than 50 fixed-size chunks and 11 of 200 with more than 50 structure-aware ones).
FUSION_DEPTH = 50


def rrf_score(semantic_rank: int | None, bm25_rank: int | None) -> float:

    score = 0

    if semantic_rank is not None:
        score += 1/(RRF_K+semantic_rank)

    if bm25_rank is not None:
        score += 1/(RRF_K+bm25_rank)

    return score


def rrf_ranker(semantic_top_k: Sequence[Hit], bm25_top_k: Sequence[Hit], k: int) -> list[Hit]:

    chunk_dict = {}

    for chunk in semantic_top_k:
            chunk_dict[chunk.chunk_id] = {
                "semantic_rank": chunk.rank,
                "bm25_rank": None
            }


    for chunk in bm25_top_k:

        if chunk.chunk_id not in chunk_dict.keys():

            chunk_dict[chunk.chunk_id] = {
                "bm25_rank": chunk.rank,
                "semantic_rank": None
            }

        else:
            chunk_dict[chunk.chunk_id]["bm25_rank"] = chunk.rank

    rrf_chunk_score ={}

    for key ,values in chunk_dict.items():

        rrf_chunk_score[key] = rrf_score(semantic_rank=values["semantic_rank"],bm25_rank=values["bm25_rank"])

    # Best fused score first. sorted() is stable, so ties keep the order the chunks were first
    # seen in: the dense list's order, then BM25-only chunks in BM25's order.
    order = sorted(rrf_chunk_score, key=lambda chunk_id: -rrf_chunk_score[chunk_id])[:k]

    return [Hit(chunk_id, rrf_chunk_score[chunk_id], rank) for rank, chunk_id in enumerate(order, 1)]


def hybrid_search(question: str, chunks: Sequence[Chunk], k: int = 10) -> list[Hit]:
    return rrf_ranker(semantic_top_k=dense_search(question, chunks, FUSION_DEPTH),
                      bm25_top_k=bm25_search(question, chunks, FUSION_DEPTH), k=k)
