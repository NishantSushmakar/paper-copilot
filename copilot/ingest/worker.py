import random
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Callable

from copilot.infra.ingest_config import QueueConfig, load_ingest_config
from copilot.ingest.event_queue import now_utc
from copilot.ingest.events import ArxivEventModel


class TransientError(Exception):
    """Waiting might fix it (server down, timeout): retry with backoff."""


class PermanentError(Exception):
    """Retrying can't fix it (e.g. an unparseable PDF): dead-letter straight away."""


Handler = Callable[[ArxivEventModel], None]


def _iso_in(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def backoff_seconds(failed_attempts: int, config: QueueConfig) -> float:
    """Wait before the next try: base, doubling per failed attempt, scaled by +/- jitter."""
    wait = config.backoff_base_seconds * 2 ** (failed_attempts - 1)
    return wait * random.uniform(1 - config.backoff_jitter, 1 + config.backoff_jitter)


def claim_next(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Take the oldest due pending event. One UPDATE statement, so two workers can't both get it."""
    now = now_utc()
    with conn:
        rows = conn.execute(
            """
            UPDATE events
            SET status = 'processing', claimed_at = :now,
                first_claimed_at = COALESCE(first_claimed_at, :now)
            WHERE rowid = (
                SELECT rowid FROM events
                WHERE status = 'pending' AND next_attempt_at <= :now
                ORDER BY next_attempt_at
                LIMIT 1
            )
            RETURNING *
            """,
            {"now": now},
        ).fetchall()
    return rows[0] if rows else None


# Every update below only applies while the row is still ours: still processing, under the
# same claim. If our lease expired and someone else took the event, the update matches nothing.
_OURS = "source = ? AND id = ? AND version = ? AND status = 'processing' AND claimed_at = ?"


def _ours(row: sqlite3.Row) -> tuple:
    return (row["source"], row["id"], row["version"], row["claimed_at"])


def mark_done(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    with conn:
        cursor = conn.execute(
            f"UPDATE events SET status = 'done', done_at = ?, claimed_at = NULL WHERE {_OURS}",
            (now_utc(), *_ours(row)),
        )
    return "done" if cursor.rowcount == 1 else "lease_lost"


def record_failure(
    conn: sqlite3.Connection, row: sqlite3.Row, reason: str, permanent: bool, config: QueueConfig
) -> str:
    """Count a failed attempt: back to pending with a backoff, or dead after the last attempt."""
    attempts = row["attempt_count"] + 1
    if not permanent and attempts < config.max_attempts:
        with conn:
            cursor = conn.execute(
                "UPDATE events SET status = 'pending', attempt_count = ?, last_error = ?,"
                f" next_attempt_at = ?, claimed_at = NULL WHERE {_OURS}",
                (attempts, reason, _iso_in(backoff_seconds(attempts, config)), *_ours(row)),
            )
        return "retry" if cursor.rowcount == 1 else "lease_lost"

    # The queue row stays (status dead) so a re-delivered duplicate is still recognised;
    # the dead-letter row says why. Both in one transaction.
    with conn:
        cursor = conn.execute(
            f"UPDATE events SET status = 'dead', attempt_count = ?, last_error = ?, claimed_at = NULL WHERE {_OURS}",
            (attempts, reason, *_ours(row)),
        )
        if cursor.rowcount == 0:
            return "lease_lost"
        conn.execute(
            "INSERT INTO dead_letters (source, id, version, raw_payload, failure_kind,"
            " failure_reason, attempt_count, dead_lettered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["source"], row["id"], row["version"], row["payload"],
             "permanent" if permanent else "exhausted", reason, attempts, now_utc()),
        )
    return "dead"


def expire_leases(conn: sqlite3.Connection, config: QueueConfig) -> Counter:
    """Events stuck in processing past the lease: the worker presumably died. Counts as a failed attempt."""
    cutoff = _iso_in(-config.lease_seconds)
    stuck = conn.execute(
        "SELECT * FROM events WHERE status = 'processing' AND claimed_at < ?", (cutoff,)
    ).fetchall()
    outcomes = Counter()
    for row in stuck:
        outcome = record_failure(conn, row, "lease expired: worker presumed dead", False, config)
        outcomes[f"lease_expired_{outcome}"] += 1
    return outcomes


def _unfinished(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM events WHERE status IN ('pending', 'processing')"
    ).fetchone()[0]


def run_worker(
    conn: sqlite3.Connection,
    handler: Handler,
    config: QueueConfig | None = None,
    stop_when_idle: bool = True,
) -> Counter:
    """Process events until none are left (or forever if stop_when_idle is False)."""
    config = config or load_ingest_config().queue
    outcomes = Counter()
    while True:
        outcomes.update(expire_leases(conn, config))
        row = claim_next(conn)
        if row is None:
            if stop_when_idle and _unfinished(conn) == 0:
                return outcomes
            time.sleep(config.poll_interval_seconds)  # waiting for a backoff or a lease to run out
            continue

        try:
            handler(ArxivEventModel.model_validate_json(row["payload"]))
        except PermanentError as error:
            outcomes[record_failure(conn, row, repr(error), True, config)] += 1
        except Exception as error:
            # TransientError, and anything unexpected: retried, but bounded by max_attempts.
            outcomes[record_failure(conn, row, repr(error), False, config)] += 1
        else:
            outcomes[mark_done(conn, row)] += 1
