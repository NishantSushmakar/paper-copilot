"""BM25 and dense retrieval over one paper's chunks (reports/a03.md §3)."""

import subprocess
import sys

import numpy as np
import pytest

from copilot.infra.embedding_store import EmbeddingStore
from copilot.retrieval import dense
from copilot.retrieval.bm25 import bm25_search, tokenize
from copilot.retrieval.chunking import Chunk
from copilot.retrieval.hit import ranked

INSTRUCTION = "Given a question about a research paper, retrieve the passage that answers it"

TEXTS = [
    "1 Introduction\nWe study machine translation between English and German.",
    "2 Data\nWe train on the WMT14 corpus with 4.5 million sentence pairs.",
    "3 Model\nOur encoder is a six-layer Transformer with relative position embeddings.",
    "4 Results\nThe model reaches 28.4 BLEU on newstest2014, above the baseline.",
    "5 Analysis\nErrors concentrate on rare named entities and long sentences.",
]


def make_chunks(texts=TEXTS) -> list[Chunk]:
    return [Chunk(f"arxiv:2611.00001v1:test:{i}:{i:08x}", i, None, 0, text) for i, text in enumerate(texts)]


def test_ranked_keeps_paper_order_on_ties_and_cuts_at_k():
    chunks = make_chunks()
    hits = ranked(chunks, [0.0, 2.0, 0.0, 2.0, 1.0], k=4)
    assert [h.chunk_id for h in hits] == [chunks[i].chunk_id for i in (1, 3, 4, 0)]
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_tokenize_lowercases_and_splits_on_anything_but_letters_and_digits():
    assert tokenize("BERT-base's F1") == ["bert", "base", "s", "f1"]
    assert tokenize("snake_case état 28.4") == ["snake", "case", "état", "28", "4"]


def test_bm25_ranks_the_chunk_with_the_rare_question_words_first():
    hits = bm25_search("How many sentence pairs are in the corpus?", make_chunks(), k=3)
    assert hits[0].chunk_id == make_chunks()[1].chunk_id
    assert len(hits) == 3 and hits[0].score > hits[1].score


def test_bm25_with_no_word_in_common_falls_back_to_paper_order():
    hits = bm25_search("zzz qqq", make_chunks(), k=2)
    assert [h.chunk_id for h in hits] == [c.chunk_id for c in make_chunks()[:2]]
    assert all(h.score == 0 for h in hits)


@pytest.mark.parametrize("search", [bm25_search, lambda q, c, k: dense.dense_search(q, c, k, INSTRUCTION)])
def test_k_larger_than_the_paper_returns_every_chunk_and_no_chunks_returns_nothing(search, tmp_path, monkeypatch):
    monkeypatch.setattr(dense, "default_store", lambda: EmbeddingStore(tmp_path))
    assert len(search("translation", make_chunks(), 50)) == len(TEXTS)
    assert search("translation", [], 5) == []


def test_dense_without_an_instruction_refuses():
    with pytest.raises(ValueError, match="QUERY_INSTRUCTION"):
        dense.embed_query("What is the BLEU score?", instruction="")


def test_chunk_vectors_are_unit_length_and_cached(tmp_path, monkeypatch):
    store = EmbeddingStore(tmp_path)
    first = dense.embed_chunks(make_chunks(), store)
    assert first.shape == (len(TEXTS), 1024)
    assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-5)

    calls = []
    real_encode = dense.model().encode
    monkeypatch.setattr(dense.model(), "encode", lambda *a, **kw: calls.append(a) or real_encode(*a, **kw))
    again = dense.embed_chunks(make_chunks(), store)
    assert calls == []  # nothing re-embedded
    assert np.array_equal(first, again)


def test_embedding_the_same_text_twice_gives_the_same_vector(tmp_path):
    a = dense.embed_chunks(make_chunks(), EmbeddingStore(tmp_path / "a"))
    b = dense.embed_chunks(make_chunks(), EmbeddingStore(tmp_path / "b"))
    assert np.allclose(a, b, atol=1e-4)  # mps is not bit-exact


def test_a_chunks_own_text_as_the_query_ranks_it_first(tmp_path):
    chunks = make_chunks()
    store = EmbeddingStore(tmp_path)
    for chunk in chunks:
        hits = dense.dense_search(chunk.text, chunks, k=1, instruction=INSTRUCTION, store=store)
        assert hits[0].chunk_id == chunk.chunk_id


def test_dense_ranks_the_answering_chunk_first(tmp_path):
    hits = dense.dense_search("What BLEU score does the model get?", make_chunks(), k=5,
                              instruction=INSTRUCTION, store=EmbeddingStore(tmp_path))
    assert hits[0].chunk_id == make_chunks()[3].chunk_id
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_loads_the_model_in_a_fresh_process():
    """Importing dense first must not segfault (torch before faiss: see copilot/retrieval/dense.py)."""
    code = "from copilot.retrieval import dense; dense.model(); print('ok')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_the_model_key_names_the_model_its_revision_and_dtype():
    name, rest = dense.model_key().split("@")
    revision, dtype = rest.split("/")
    assert name == "Qwen/Qwen3-Embedding-0.6B" and len(revision) == 12 and dtype == "float32"
