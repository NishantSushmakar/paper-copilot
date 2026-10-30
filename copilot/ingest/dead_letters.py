import argparse
import sqlite3
from collections import Counter

from copilot.infra.ingest_config import load_ingest_config
from copilot.ingest.event_queue import connect, insert_event, now_utc

KINDS = ("exhausted", "permanent", "malformed")


def _short(reason: str, width: int = 90) -> str:
    one_line = " ".join(reason.split())
    return one_line if len(one_line) <= width else one_line[: width - 3] + "..."


def inspect(conn: sqlite3.Connection, kind: str | None = None) -> None:
    """Print a count per kind, then one line per dead letter, oldest first."""
    counts = Counter(r[0] for r in conn.execute("SELECT failure_kind FROM dead_letters"))
    print("dead letters:", ", ".join(f"{k} {counts[k]}" for k in KINDS), f"(total {sum(counts.values())})")
    query = "SELECT * FROM dead_letters" + (" WHERE failure_kind = ?" if kind else "") + " ORDER BY row_id"
    for row in conn.execute(query, (kind,) if kind else ()):
        paper = f"{row['id']}v{row['version']}" if row["id"] is not None else "(no key)"
        print(
            f"{row['row_id']:>5}  {row['failure_kind']:<9}  {paper:<16}  "
            f"attempts {row['attempt_count']}  {row['dead_lettered_at'][:19]}  {_short(row['failure_reason'])}"
        )


def replay_row(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    """Give one dead letter another chance. The dead-letter row is removed in the same
    transaction, so a crash can neither lose the event nor replay it twice."""
    with conn:
        if row["failure_kind"] == "malformed":
            # Never had a queue row: send the raw payload through the front door again.
            # If it is still broken, insert_event writes a new dead-letter row.
            outcome = insert_event(conn, row["raw_payload"])
        else:
            cursor = conn.execute(
                "UPDATE events SET status = 'pending', attempt_count = 0, next_attempt_at = ?,"
                " claimed_at = NULL WHERE source = ? AND id = ? AND version = ? AND status = 'dead'",
                (now_utc(), row["source"], row["id"], row["version"]),
            )
            if cursor.rowcount == 0:
                return "skipped"  # the queue row is no longer dead; leave the record alone
            outcome = "requeued"
        conn.execute("DELETE FROM dead_letters WHERE row_id = ?", (row["row_id"],))
    return outcome


def replay(conn: sqlite3.Connection, kind: str | None = None, row_ids: list[int] | None = None) -> Counter:
    """Replay the chosen rows: by row number, by kind, or (default) every exhausted one."""
    if row_ids:
        marks = ", ".join("?" for _ in row_ids)
        rows = conn.execute(f"SELECT * FROM dead_letters WHERE row_id IN ({marks}) ORDER BY row_id", row_ids)
    else:
        rows = conn.execute(
            "SELECT * FROM dead_letters WHERE failure_kind = ? ORDER BY row_id", (kind or "exhausted",)
        )
    outcomes = Counter()
    for row in rows.fetchall():
        outcomes[replay_row(conn, row)] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and replay dead-lettered events.")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("inspect", help="list dead letters")
    show.add_argument("--kind", choices=KINDS)
    again = sub.add_parser("replay", help="put dead letters back into the pipeline")
    choice = again.add_mutually_exclusive_group()
    choice.add_argument("--kind", choices=KINDS, help="default: exhausted")
    choice.add_argument("--row", type=int, action="append", dest="rows", help="row number (repeatable)")
    args = parser.parse_args()

    conn = connect(load_ingest_config().queue.db_file)
    if args.command == "inspect":
        inspect(conn, args.kind)
    else:
        print(dict(replay(conn, args.kind, args.rows)))


if __name__ == "__main__":
    main()
