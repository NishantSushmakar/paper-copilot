import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from copilot.ingest.events import ArxivEventModel

EnqueueResult = Literal["enqueued", "duplicate", "malformed"]

SCHEMA = """
-- One row per event key, kept for ever: this is where "have we seen this?" is answered.
CREATE TABLE IF NOT EXISTS events (
    source           TEXT    NOT NULL,
    id               TEXT    NOT NULL,
    version          INTEGER NOT NULL,
    payload          TEXT    NOT NULL,   -- the event JSON as received
    status           TEXT    NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'processing', 'done', 'dead')),
    attempt_count    INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  TEXT    NOT NULL,   -- not tried before this; arrived_at for a new event
    arrived_at       TEXT    NOT NULL,
    claimed_at       TEXT,               -- lease: set when a worker claims the row
    first_claimed_at TEXT,               -- first time any worker picked it up
    last_error       TEXT,
    done_at          TEXT,
    PRIMARY KEY (source, id, version)
) STRICT;

-- The worker's lookup: pending rows that are due, oldest first.
CREATE INDEX IF NOT EXISTS events_due ON events (status, next_attempt_at);

-- Why events failed. Malformed events may have no key, so the key columns can be empty
-- and each row gets its own number instead.
CREATE TABLE IF NOT EXISTS dead_letters (
    row_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT,
    id               TEXT,
    version          INTEGER,
    raw_payload      TEXT    NOT NULL,   -- exactly as received, so replay can resend it
    failure_kind     TEXT    NOT NULL
                     CHECK (failure_kind IN ('malformed', 'exhausted', 'permanent')),
    failure_reason   TEXT    NOT NULL,
    attempt_count    INTEGER NOT NULL,
    dead_lettered_at TEXT    NOT NULL
        

) STRICT;
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the queue database, creating the tables on first use."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _salvage_key(line: str) -> tuple[str | None, str | None, int | None]:
    """Best-effort key for a malformed line. Only values of the right type are kept,
    because the STRICT dead_letters table would reject anything else."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None
    source, paper_id, version = data.get("source"), data.get("id"), data.get("version")
    return (
        source if isinstance(source, str) else None,
        paper_id if isinstance(paper_id, str) else None,
        # bool is a subclass of int in Python: don't let `true` pass as version 1.
        version if isinstance(version, int) and not isinstance(version, bool) else None,
    )


def enqueue(conn: sqlite3.Connection, line: str) -> EnqueueResult:
    """The front door: one raw event line in, one committed row out.

    Valid events become `pending` rows (raw line kept as the payload); a key that is
    already in the queue changes nothing; anything the event model rejects goes to
    dead_letters as `malformed`.
    """
    with conn:  # commits this one event before returning
        return insert_event(conn, line)


def insert_event(conn: sqlite3.Connection, line: str) -> EnqueueResult:
    """`enqueue` without the commit, for callers that need it inside a larger transaction (replay)."""
    try:
        event = ArxivEventModel.model_validate_json(line)  # parses and validates in one step
    except ValidationError as error:
        source, paper_id, version = _salvage_key(line)
        conn.execute(
            "INSERT INTO dead_letters (source, id, version, raw_payload, failure_kind,"
            " failure_reason, attempt_count, dead_lettered_at) VALUES (?, ?, ?, ?, 'malformed', ?, 0, ?)",
            (source, paper_id, version, line, str(error), now_utc()),
        )
        return "malformed"

    arrived_at = now_utc()
    cursor = conn.execute(
        "INSERT INTO events (source, id, version, payload, next_attempt_at, arrived_at)"
        " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        (event.source, event.id, event.version, line, arrived_at, arrived_at),
    )
    return "enqueued" if cursor.rowcount == 1 else "duplicate"


def enqueue_file(conn: sqlite3.Connection, path: Path) -> Counter:
    """Enqueue every line of a JSONL file; returns how many lines got each result."""
    results = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                results[enqueue(conn, line)] += 1
    return results
