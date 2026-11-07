"""Reciprocal rank fusion (copilot/retrieval/hybrid.py), on the worked example from reports/a03.md §5."""

import pytest

from copilot.retrieval.hit import Hit
from copilot.retrieval.hybrid import RRF_K, rrf_ranker, rrf_score


def hits(*chunk_ids: str) -> list[Hit]:
    """A ranked list; the scores are deliberately on wildly different scales and must be ignored."""
    return [Hit(chunk_id, 1000.0 / rank, rank) for rank, chunk_id in enumerate(chunk_ids, 1)]


def test_the_worked_example():
    # BM25 ranks A, B, C; dense ranks B, D, A.
    fused = rrf_ranker(semantic_top_k=hits("B", "D", "A"), bm25_top_k=hits("A", "B", "C"), k=10)
    assert [h.chunk_id for h in fused] == ["B", "A", "D", "C"]
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert [h.score for h in fused] == pytest.approx([1/61 + 1/62, 1/61 + 1/63, 1/62, 1/63])


def test_k_cuts_the_fused_list():
    fused = rrf_ranker(hits("B", "D", "A"), hits("A", "B", "C"), k=2)
    assert [h.chunk_id for h in fused] == ["B", "A"]


def test_a_chunk_missing_from_one_list_gets_nothing_from_it():
    assert rrf_score(semantic_rank=1, bm25_rank=None) == pytest.approx(1 / (RRF_K + 1))
    assert rrf_score(semantic_rank=None, bm25_rank=1) == pytest.approx(1 / (RRF_K + 1))
    assert rrf_score(semantic_rank=None, bm25_rank=None) == 0


def test_only_ranks_matter_not_scores():
    a = rrf_ranker(hits("X", "Y"), hits("Y", "X"), k=2)
    b = rrf_ranker([Hit("X", 0.9, 1), Hit("Y", 0.1, 2)], [Hit("Y", 1e6, 1), Hit("X", -5, 2)], k=2)
    assert [(h.chunk_id, h.score) for h in a] == [(h.chunk_id, h.score) for h in b]


def test_ties_keep_the_dense_order():
    # X and Y each get 1/61 + 1/62: the dense list put X first.
    fused = rrf_ranker(hits("X", "Y"), hits("Y", "X"), k=2)
    assert [h.chunk_id for h in fused] == ["X", "Y"]
    assert fused[0].score == fused[1].score


def test_hybrid_search_fuses_full_depth_lists_and_cuts_only_the_result(tmp_path, monkeypatch):
    from copilot.infra.embedding_store import EmbeddingStore
    from copilot.retrieval import dense, hybrid
    from copilot.retrieval.chunking import Chunk

    monkeypatch.setattr(dense, "default_store", lambda: EmbeddingStore(tmp_path))
    texts = ["1 Data\nWe train on the WMT14 corpus.", "2 Model\nA six-layer Transformer.",
             "3 Results\nThe model reaches 28.4 BLEU.", "4 Analysis\nErrors on rare named entities."]
    chunks = [Chunk(f"arxiv:x:test:{i}:{i:08x}", i, None, 0, t) for i, t in enumerate(texts)]
    question = "What BLEU score does the model reach?"

    top1 = hybrid.hybrid_search(question, chunks, k=1)
    full = hybrid.rrf_ranker(dense.dense_search(question, chunks, 50), hybrid.bm25_search(question, chunks, 50), k=4)
    assert len(top1) == 1 and top1[0] == full[0]  # k cuts the fused list, not the inputs
    assert len(hybrid.hybrid_search(question, chunks, k=10)) == len(chunks)


def test_empty_lists_give_nothing():
    assert rrf_ranker([], [], k=5) == []
    assert [h.chunk_id for h in rrf_ranker([], hits("A", "B"), k=5)] == ["A", "B"]
