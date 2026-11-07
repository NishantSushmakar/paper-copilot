"""Tests for the A1 card backfill (copilot/cards/backfill.py).

Each case is a decision recorded in reports/a02.md §8: a card is keyed by the paper and
the extractor; a paper is "complete", "partial" (the card merges its valid chunks) or
"none"; each chunk's outcome is stored with the card; a paper that raises gets an error
row and no card, so the next run retries it, and the batch goes on; papers with a card
are skipped. The model and the chunker are replaced by fakes, so no model is called.
"""

from datetime import datetime, timezone

import pytest

from copilot.cards import backfill, extract
from copilot.cards.chunking import Chunk
from copilot.cards.extract import ChunkOutcome
from copilot.cards.schema import PaperCardModel
from copilot.ingest.paper import PaperModel, SectionModel

CARD = {"tasks": [{"names": ["question answering"]}], "datasets": [], "methods": [], "metrics": [],
        "results": [], "code_url": None}


def paper(id_):
    return PaperModel(
        source="arxiv", id=id_, version=1, title="T", abstract="A",
        sections=[SectionModel(heading="1 Introduction", paragraphs=["Body."])],
        fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def fake_model(monkeypatch):
    """Two chunks per paper; plan[paper id] is the chunk statuses, or "raise"."""
    plan, calls = {}, []

    def fake_chunk_text(text, chunk_size, chunk_overlap):
        return [Chunk(index=i, start_char=0, end_char=1, start_word=0, end_word=1, token_count=10, text=f"c{i}")
                for i in range(2)]

    def fake_s2(chunk_text, *, slot, paper_id, chunk_index, prompt_version):
        calls.append((paper_id, chunk_index))
        statuses = plan.get(paper_id, ["valid", "valid"])
        if statuses == "raise":
            raise ConnectionError("server down")
        status = statuses[chunk_index]
        return ChunkOutcome(
            paper_id=paper_id, chunk_index=chunk_index, strategy="S2", slot=slot, prompt_version=prompt_version,
            status=status, card=PaperCardModel.model_validate(CARD) if status == "valid" else None,
            failed_fields=["results"] if status == "content" else [], raw_text="{}", finish_reason="stop",
            prompt_tokens=100, completion_tokens=50, latency_s=10.0, cache_hit=False,
        )

    monkeypatch.setattr(backfill, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(extract, "extract_s2", fake_s2)
    return plan, calls


@pytest.fixture
def conn(tmp_path):
    conn = backfill.connect(tmp_path / "cards.db")
    yield conn
    conn.close()


def card_row(conn, id_):
    return conn.execute("SELECT status, chunks_total, chunks_valid FROM cards WHERE id = ?", (id_,)).fetchone()


def test_complete_card_is_stored_with_its_chunks(conn, fake_model):
    counts = backfill.backfill(conn, [paper("p1")])
    assert counts["complete"] == 1
    assert card_row(conn, "p1") == ("complete", 2, 2)
    assert conn.execute("SELECT COUNT(*) FROM chunk_outcomes WHERE id = 'p1'").fetchone()[0] == 2


def test_partial_card_merges_the_valid_chunks(conn, fake_model):
    plan, _ = fake_model
    plan["p1v1"] = ["valid", "content"]
    backfill.backfill(conn, [paper("p1")])
    assert card_row(conn, "p1") == ("partial", 2, 1)
    statuses = conn.execute("SELECT status FROM chunk_outcomes WHERE id = 'p1' ORDER BY chunk_index").fetchall()
    assert statuses == [("valid",), ("content",)]


def test_no_valid_chunk_is_stored_as_none(conn, fake_model):
    plan, _ = fake_model
    plan["p1v1"] = ["cut_off", "format"]
    backfill.backfill(conn, [paper("p1")])
    assert card_row(conn, "p1") == ("none", 2, 0)


def test_papers_with_a_card_are_skipped(conn, fake_model):
    _, calls = fake_model
    backfill.backfill(conn, [paper("p1")])
    backfill.backfill(conn, [paper("p1"), paper("p2")])
    assert [paper_id for paper_id, _ in calls] == ["p1v1", "p1v1", "p2v1", "p2v1"]


def test_a_card_from_another_extractor_does_not_count(conn, fake_model, monkeypatch):
    backfill.backfill(conn, [paper("p1")])
    monkeypatch.setattr(backfill, "EXTRACTOR", "v4/SMALL_16K/S2/4000+100/share_name")
    assert not backfill.has_card(conn, paper("p1"))


def test_an_error_leaves_no_card_and_the_batch_goes_on(conn, fake_model):
    plan, _ = fake_model
    plan["p1v1"] = "raise"
    counts = backfill.backfill(conn, [paper("p1"), paper("p2")])
    assert counts == {"error": 1, "complete": 1}
    assert card_row(conn, "p1") is None
    assert conn.execute("SELECT id FROM card_errors").fetchall() == [("p1",)]


def test_a_paper_that_errored_is_retried_on_the_next_run(conn, fake_model):
    plan, _ = fake_model
    plan["p1v1"] = "raise"
    backfill.backfill(conn, [paper("p1")])
    del plan["p1v1"]
    backfill.backfill(conn, [paper("p1")])
    assert card_row(conn, "p1") == ("complete", 2, 2)


def test_errors_in_a_row_stop_the_batch(conn, fake_model):
    plan, _ = fake_model
    papers = [paper(f"p{i}") for i in range(backfill.MAX_CONSECUTIVE_ERRORS + 2)]
    for p in papers:
        plan[f"{p.id}v1"] = "raise"
    counts = backfill.backfill(conn, papers)
    assert counts["error"] == backfill.MAX_CONSECUTIVE_ERRORS
