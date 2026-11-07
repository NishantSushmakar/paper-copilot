"""Backfill: a card for every stored A1 paper that doesn't have one yet (reports/a02.md §8).

    uv run python -m copilot.cards.backfill --limit 10   # the 10 newest papers: time it first
    uv run python -m copilot.cards.backfill              # every paper still without a card
    uv run python -m copilot.cards.backfill --summary    # counts so far, no model calls

One extractor, chosen on SciREX dev (reports/a02.md §7): SMALL_16K + S2, prompt v3,
4,000-token chunks, merge share_name. Cards go to data/cards/cards.db, keyed by the paper
(source, id, version) and the extractor, so a later extractor adds its own cards next to
these instead of overwriting them.

A card is "complete" (every chunk valid), "partial" (some valid; the card merges those) or
"none" (no valid chunk; the card is empty). Every chunk's outcome is stored with it,
failures included. A paper whose processing raises (the model server is down, a bug) gets
an error row and no card, so the next run tries it again. One bad paper never stops the
batch, but MAX_CONSECUTIVE_ERRORS in a row do, since that means the server is down.

Restarting is cheap: papers with a card are skipped, and the chunks a crashed run had
finished replay from the LLM cache.
"""

import argparse
import json
import logging
import sqlite3
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from copilot.cards import extract
from copilot.cards.chunking import chunk_text
from copilot.cards.merge import MergedPaper, merge_paper
from copilot.cards.paper_text import is_back_matter, paper_text
from copilot.infra.ingest_config import ROOT_DIR, load_ingest_config
from copilot.ingest.paper import PaperModel

logger = logging.getLogger(__name__)

SLOT = "SMALL_16K"
STRATEGY = "S2"  # extract.extract_s2 in card_paper
PROMPT_VERSION = "v3"
CHUNK_SIZE = 4000  # tokens
CHUNK_OVERLAP = 100
MERGE_RULE = "share_name"
EXTRACTOR = f"{PROMPT_VERSION}/{SLOT}/{STRATEGY}/{CHUNK_SIZE}+{CHUNK_OVERLAP}/{MERGE_RULE}"

CARDS_DB = ROOT_DIR / "data" / "cards" / "cards.db"
MAX_CONSECUTIVE_ERRORS = 3

SCHEMA = """
-- One row per paper and extractor. A row means "done": the backfill skips it.
CREATE TABLE IF NOT EXISTS cards (
    source           TEXT    NOT NULL,
    id               TEXT    NOT NULL,
    version          INTEGER NOT NULL,
    extractor        TEXT    NOT NULL,
    status           TEXT    NOT NULL CHECK (status IN ('complete', 'partial', 'none')),
    chunks_total     INTEGER NOT NULL,
    chunks_valid     INTEGER NOT NULL,
    dropped_sections TEXT    NOT NULL,   -- JSON list of the back-matter headings left out
    merged           TEXT    NOT NULL,   -- the MergedPaper; its card is empty when status is 'none'
    created_at       TEXT    NOT NULL,
    PRIMARY KEY (source, id, version, extractor)
) STRICT;

-- Every chunk's ChunkOutcome, with all its attempts: the failure log.
CREATE TABLE IF NOT EXISTS chunk_outcomes (
    source       TEXT    NOT NULL,
    id           TEXT    NOT NULL,
    version      INTEGER NOT NULL,
    extractor    TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    status       TEXT    NOT NULL,
    chunk_tokens INTEGER NOT NULL,
    outcome      TEXT    NOT NULL,
    PRIMARY KEY (source, id, version, extractor, chunk_index)
) STRICT;

-- Papers whose processing raised. They have no card, so the next run retries them.
CREATE TABLE IF NOT EXISTS card_errors (
    row_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    source    TEXT    NOT NULL,
    id        TEXT    NOT NULL,
    version   INTEGER NOT NULL,
    extractor TEXT    NOT NULL,
    error     TEXT    NOT NULL,
    failed_at TEXT    NOT NULL
) STRICT;
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path = CARDS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def stored_papers(queue_db: Path) -> list[PaperModel]:
    """Every paper in the A1 store, newest arXiv ID first (so the UI's 10 papers come first)."""
    store = sqlite3.connect(f"file:{queue_db}?mode=ro", uri=True)
    try:
        rows = store.execute("SELECT paper_json FROM papers ORDER BY id DESC, version DESC").fetchall()
    finally:
        store.close()
    return [PaperModel.model_validate_json(paper_json) for (paper_json,) in rows]


def has_card(conn: sqlite3.Connection, paper: PaperModel) -> bool:
    row = conn.execute(
        "SELECT 1 FROM cards WHERE source = ? AND id = ? AND version = ? AND extractor = ?",
        (paper.source, paper.id, paper.version, EXTRACTOR),
    ).fetchone()
    return row is not None


def card_status(chunks_valid: int, chunks_total: int) -> str:
    if chunks_valid == chunks_total:
        return "complete"
    return "partial" if chunks_valid else "none"


def card_paper(conn: sqlite3.Connection, paper: PaperModel) -> tuple[str, MergedPaper]:
    """Chunk, extract and merge one paper, then store its card and chunk outcomes together."""
    chunks = chunk_text(paper_text(paper), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    paper_id = f"{paper.id}v{paper.version}"
    outcomes = [
        extract.extract_s2(chunk.text, slot=SLOT, paper_id=paper_id, chunk_index=chunk.index,
                           prompt_version=PROMPT_VERSION)
        for chunk in chunks
    ]
    merged = merge_paper(outcomes, merge_rule=MERGE_RULE)
    status = card_status(merged.chunks_valid, merged.chunks_total)
    key = (paper.source, paper.id, paper.version, EXTRACTOR)
    dropped = [section.heading for section in paper.sections if is_back_matter(section.heading)]

    with conn:  # one transaction: a paper is stored whole or not at all
        conn.executemany(
            "INSERT OR REPLACE INTO chunk_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(*key, o.chunk_index, o.status, c.token_count, o.model_dump_json()) for c, o in zip(chunks, outcomes)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, status, merged.chunks_total, merged.chunks_valid, json.dumps(dropped),
             merged.model_dump_json(), now()),
        )
    return status, merged


def record_error(conn: sqlite3.Connection, paper: PaperModel, exc: Exception) -> None:
    with conn:
        conn.execute(
            "INSERT INTO card_errors (source, id, version, extractor, error, failed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (paper.source, paper.id, paper.version, EXTRACTOR, repr(exc), now()),
        )


def backfill(conn: sqlite3.Connection, papers: list[PaperModel]) -> Counter:
    """Card every paper that has no card from EXTRACTOR; returns this run's counts by outcome."""
    todo = [paper for paper in papers if not has_card(conn, paper)]
    print(f"{EXTRACTOR}: {len(papers)} papers, {len(papers) - len(todo)} already have a card, {len(todo)} to do")
    counts, errors_in_row, start = Counter(), 0, time.monotonic()

    for n, paper in enumerate(todo, 1):
        try:
            status, merged = card_paper(conn, paper)
        except Exception as exc:  # anything: one paper must not stop the batch
            logger.exception("%sv%d failed", paper.id, paper.version)
            record_error(conn, paper, exc)
            counts["error"] += 1
            errors_in_row += 1
            if errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                print(f"stopping: {errors_in_row} errors in a row (is the model server up?)")
                break
            continue
        errors_in_row = 0
        counts[status] += 1
        print(f"[{n}/{len(todo)}] {paper.id}v{paper.version} {status}: {merged.chunks_valid}/{merged.chunks_total}"
              f" chunks valid, {merged.latency_s:.0f} s ({(time.monotonic() - start) / n:.0f} s/paper so far)")
    return counts


def summary(conn: sqlite3.Connection, store_size: int) -> None:
    """Counts for EXTRACTOR over everything stored so far. No model calls."""
    statuses = Counter(dict(conn.execute(
        "SELECT status, COUNT(*) FROM cards WHERE extractor = ? GROUP BY status", (EXTRACTOR,))))
    latencies = [json.loads(m)["latency_s"]
                 for (m,) in conn.execute("SELECT merged FROM cards WHERE extractor = ?", (EXTRACTOR,))]
    outcomes = [json.loads(o)
                for (o,) in conn.execute("SELECT outcome FROM chunk_outcomes WHERE extractor = ?", (EXTRACTOR,))]
    open_errors = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT source, id, version FROM card_errors e WHERE extractor = ?"
        " AND NOT EXISTS (SELECT 1 FROM cards c WHERE c.source = e.source AND c.id = e.id"
        " AND c.version = e.version AND c.extractor = e.extractor))",
        (EXTRACTOR,),
    ).fetchone()[0]

    print(f"\n{EXTRACTOR}: {sum(statuses.values())} of {store_size} papers have a card"
          f" (complete {statuses['complete']}, partial {statuses['partial']}, none {statuses['none']});"
          f" {open_errors} errored and wait for a retry")
    if outcomes:
        n = len(outcomes)
        first_valid = sum(o["history"][0]["status"] == "valid" for o in outcomes)
        final = Counter(o["status"] for o in outcomes)
        retries = sum(o["attempts"] - 1 for o in outcomes)
        print(f"chunks: {n}; first-attempt valid {first_valid / n:.1%}, final valid {final['valid'] / n:.1%},"
              f" mean retries {retries / n:.2f}; final failures {dict(final - Counter(valid=final['valid']))}")
    if latencies:
        print(f"seconds per paper (recorded model time): mean {statistics.mean(latencies):.0f},"
              f" median {statistics.median(latencies):.0f}, max {max(latencies):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the N newest papers")
    parser.add_argument("--summary", action="store_true", help="print the counts so far and exit; no model calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    papers = stored_papers(load_ingest_config().queue.db_file)
    store_size = len(papers)
    if args.limit:
        papers = papers[: args.limit]

    conn = connect()
    try:
        if not args.summary:
            print("this run:", dict(backfill(conn, papers)))
        summary(conn, store_size)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
