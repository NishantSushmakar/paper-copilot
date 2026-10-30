"""A1 replay eval: push the recorded arXiv fixture through the real pipeline under faults.

    uv run python -m evals.a1_replay

What it does, all offline (papers come from the data/arxiv/raw cache that the real run
filled; nothing is fetched from arXiv):
  1. Builds a stream from the 300 fixture events plus 20% injected duplicates and 5%
     malformed payloads, shuffled with a fixed seed.
  2. Enqueues the stream into a fresh queue database inside the run folder.
  3. Runs three worker processes in turn, each with the real handler wrapped so 10% of
     calls fail with a TransientError:
       a. "crash_after_store" exits on its own right after storing one paper, before the
          event is marked done: the at-least-once window, which forces a redelivery;
       b. "before_kill" is killed with SIGKILL mid-batch;
       c. "after_restart" finishes the job.
     The lease is shortened to 10 s so the restarts don't wait 5 min.
  4. Measures: redeliveries, duplicate processing rate, lost events, eventual success for
     transient failures, dead-letter precision and recall, throughput.

Writes evals/results/<run_id>/: stream.jsonl, labels.jsonl, handled.jsonl, queue.db, summary.json.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from copilot.infra.ingest_config import load_ingest_config
from copilot.infra.provenance import file_sha256, git_info
from copilot.ingest.event_queue import connect, enqueue
from copilot.ingest.events import ArxivEventModel
from copilot.ingest.handler import RAW_DIR, ArxivHandler
from copilot.ingest.worker import TransientError, run_worker

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
FIXTURE = EVALS_DIR.parent / "data" / "arxiv" / "events.jsonl"

SEED = 42
DUPLICATE_RATE = 0.20
MALFORMED_RATE = 0.05
TRANSIENT_RATE = 0.10
EVAL_LEASE_SECONDS = 10
KILL_AFTER_FRACTION = 0.5  # of the valid unique events
CRASH_AFTER_STORES = 25  # the first worker exits right after its 25th stored paper


# ---------------------------------------------------------------- the stream


def _malform(line: str, rng: random.Random, kind: str) -> str:
    """One broken payload of a given kind, made from a real event."""
    event = json.loads(line)
    if kind == "truncated_json":
        return line[: rng.randint(10, len(line) // 2)]
    if kind == "missing_field":
        event.pop(rng.choice(["id", "version", "title", "authors", "pdf_url"]))
    elif kind == "version_zero":
        event["version"] = 0
    elif kind == "wrong_type":
        event["version"] = str(event["version"])
    elif kind == "type_mismatch":  # paper_published with a version above 1, or updated with 1
        event["event_type"], event["version"] = (
            ("paper_published", 3) if event["version"] == 1 else ("paper_updated", 1)
        )
    elif kind == "naive_time":
        event["polled_at"] = event["polled_at"].replace("Z", "").split("+")[0]
    return json.dumps(event, ensure_ascii=False)


MALFORMED_KINDS = ["truncated_json", "missing_field", "version_zero", "wrong_type", "type_mismatch", "naive_time"]


def build_stream(lines: list[str], rng: random.Random) -> list[tuple[str, str]]:
    """(line, label) pairs: label is original, duplicate or malformed:<kind>."""
    stream = [(line, "original") for line in lines]
    stream += [(line, "duplicate") for line in rng.sample(lines, round(DUPLICATE_RATE * len(lines)))]
    for i, line in enumerate(rng.sample(lines, round(MALFORMED_RATE * len(lines)))):
        kind = MALFORMED_KINDS[i % len(MALFORMED_KINDS)]
        stream.append((_malform(line, rng, kind), f"malformed:{kind}"))
    rng.shuffle(stream)
    return stream


# ---------------------------------------------------------------- the worker process


class OfflineHandler(ArxivHandler):
    """The real handler, but it must never reach arXiv: every document has to be cached."""

    def _download(self, url, cache):
        if not cache.exists() and not cache.with_name(cache.name + ".404").exists():
            raise RuntimeError(f"{cache.name} is not cached; run the real pipeline first")
        return super()._download(url, cache)


def worker_main(run_dir: Path, phase: str) -> None:
    """One worker process: the real (offline) handler behind a 10% transient-failure switch.
    Every call is logged to handled.jsonl and flushed, so a SIGKILL loses nothing written."""
    db = run_dir / "queue.db"
    conn = connect(db)
    config = load_ingest_config().queue.model_copy(update={"lease_seconds": EVAL_LEASE_SECONDS})
    handler = OfflineHandler(db_path=db)
    rng = random.Random(f"{SEED}-{phase}")
    log = (run_dir / "handled.jsonl").open("a", encoding="utf-8")

    def write(record: dict) -> None:
        log.write(json.dumps({**record, "phase": phase}) + "\n")
        log.flush()

    stores = 0

    def flaky(event: ArxivEventModel) -> None:
        nonlocal stores
        if rng.random() < TRANSIENT_RATE:
            write({"id": event.id, "version": event.version, "outcome": "injected_failure"})
            raise TransientError("injected transient failure")
        handler(event)
        write({"id": event.id, "version": event.version, "outcome": handler.last_outcome["stored"],
               "parsed_from": handler.last_outcome["parsed_from"]})
        stores += handler.last_outcome["stored"] == "stored"
        if phase == "crash_after_store" and stores == CRASH_AFTER_STORES:
            os._exit(1)  # the paper is stored, the event is still "processing": no cleanup, no mark-done

    outcomes = run_worker(conn, flaky, config)
    (run_dir / f"worker_{phase}.json").write_text(json.dumps(dict(outcomes), indent=2))


def _start_worker(run_dir: Path, phase: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "evals.a1_replay", "--worker", str(run_dir), phase])


def _completed_calls(run_dir: Path) -> int:
    path = run_dir / "handled.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if '"injected_failure"' not in line)


# ---------------------------------------------------------------- metrics


def measure(run_dir: Path, stream: list[tuple[str, str]], enqueue_results: Counter, timings: dict) -> dict:
    conn = connect(run_dir / "queue.db")
    handled = [json.loads(line) for line in (run_dir / "handled.jsonl").read_text().splitlines()]
    completions = Counter((h["id"], h["version"]) for h in handled if h["outcome"] != "injected_failure")
    stored = Counter((h["id"], h["version"]) for h in handled if h["outcome"] == "stored")
    failed_once = {(h["id"], h["version"]) for h in handled if h["outcome"] == "injected_failure"}

    valid_keys = {(r["id"], r["version"]) for r in (json.loads(line) for line, label in stream if label == "original")}
    status = {(r["id"], r["version"]): r["status"] for r in conn.execute("SELECT id, version, status FROM events")}
    papers = {(r["id"], r["version"]) for r in conn.execute("SELECT id, version FROM papers")}

    injected_malformed = {line for line, label in stream if label.startswith("malformed")}
    dead_rows = conn.execute("SELECT failure_kind, raw_payload, failure_reason, id, version FROM dead_letters").fetchall()
    dead_injected = sum(1 for r in dead_rows if r["raw_payload"] in injected_malformed)
    not_malformed = [
        {"kind": r["failure_kind"], "paper": f"{r['id']}v{r['version']}", "reason": r["failure_reason"][:120]}
        for r in dead_rows if r["raw_payload"] not in injected_malformed
    ]

    done = [k for k in valid_keys if status.get(k) == "done"]
    lost = [k for k in valid_keys if status.get(k) not in ("done", "dead")]
    transient_done = [k for k in failed_once if status.get(k) == "done"]
    work_seconds = sum(timings.values())

    return {
        "stream": dict(Counter(label.split(":")[0] for _, label in stream)),
        "malformed_kinds": dict(Counter(label.split(":")[1] for _, label in stream if label.startswith("malformed"))),
        "enqueue": dict(enqueue_results),
        "final_status": dict(Counter(status.get(k) for k in valid_keys)),
        "lost_events": len(lost),
        "papers_rows": len(papers),
        "done_without_paper": len([k for k in done if k not in papers]),
        "redeliveries": sum(c - 1 for c in completions.values() if c > 1),
        "redelivered_keys": [f"{i}v{v}" for (i, v), c in completions.items() if c > 1],
        "duplicate_processing": {
            "keys_stored_more_than_once": sum(1 for c in stored.values() if c > 1),
            "rate": sum(1 for c in stored.values() if c > 1) / max(len(stored), 1),
        },
        "transient": {
            "injected_failures": sum(1 for h in handled if h["outcome"] == "injected_failure"),
            "events_with_a_failure": len(failed_once),
            "eventually_done": len(transient_done),
            "eventual_success_rate": len(transient_done) / max(len(failed_once), 1),
            # An event dies of injected failures only if all 3 attempts fail: P = 0.1^3 per event,
            # i.e. 0.01 for an event that has already failed once.
            "expected_exhausted": round(len(failed_once) * TRANSIENT_RATE ** 2, 2),
        },
        "dead_letters": {
            "rows": len(dead_rows),
            "by_kind": dict(Counter(r["failure_kind"] for r in dead_rows)),
            "precision": dead_injected / max(len(dead_rows), 1),
            "recall": dead_injected / max(len(injected_malformed), 1),
            "not_injected_malformed": not_malformed,
        },
        "parsed_from": dict(Counter(r[0] for r in conn.execute("SELECT parsed_from FROM papers"))),
        # The crashed and killed workers never write theirs; the later ones show the lease expiries.
        "worker_outcomes": {
            p.stem.removeprefix("worker_"): json.loads(p.read_text()) for p in sorted(run_dir.glob("worker_*.json"))
        },
        "timings": timings,
        "throughput_events_per_second": round(len(done) / work_seconds, 2) if work_seconds else None,
    }


# ---------------------------------------------------------------- the orchestrator


def main(limit: int | None) -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    missing = [
        line for line in lines
        if not any((RAW_DIR / f"{e.id.replace('/', '_')}v{e.version}{ext}").exists()
                   for e in [ArxivEventModel.model_validate_json(line)]
                   for ext in (".html", ".pdf", ".pdf.404"))
    ]
    if missing:
        sys.exit(f"{len(missing)} fixture papers are not cached yet; finish the real run first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}_a1_replay{f'_limit{limit}' if limit else ''}"
    run_dir.mkdir(parents=True)
    rng = random.Random(SEED)
    stream = build_stream(lines, rng)
    (run_dir / "stream.jsonl").write_text("".join(line + "\n" for line, _ in stream), encoding="utf-8")
    (run_dir / "labels.jsonl").write_text("".join(json.dumps({"label": lab}) + "\n" for _, lab in stream))

    conn = connect(run_dir / "queue.db")
    enqueue_results = Counter(enqueue(conn, line) for line, _ in stream)
    conn.close()
    print("enqueue:", dict(enqueue_results))

    timings = {}
    start = time.monotonic()
    crash_exit = _start_worker(run_dir, "crash_after_store").wait()
    timings["crash_after_store_seconds"] = round(time.monotonic() - start, 1)
    print(f"first worker exited with code {crash_exit} right after storing paper {CRASH_AFTER_STORES}")

    kill_after = int(KILL_AFTER_FRACTION * enqueue_results["enqueued"])
    start = time.monotonic()
    victim = _start_worker(run_dir, "before_kill")
    while _completed_calls(run_dir) < kill_after and victim.poll() is None:
        time.sleep(0.02)
    victim.send_signal(signal.SIGKILL)
    victim.wait()
    timings["before_kill_seconds"] = round(time.monotonic() - start, 1)
    print(f"killed the second worker after {_completed_calls(run_dir)} completed calls")

    start = time.monotonic()
    _start_worker(run_dir, "after_restart").wait()
    timings["after_restart_seconds"] = round(time.monotonic() - start, 1)

    summary = {
        "run_id": run_dir.name,
        "provenance": {
            "git": git_info(),
            "date": datetime.now(timezone.utc).isoformat(),
            "fixture_sha256": file_sha256(FIXTURE),
            "fixture_lines_used": len(lines),
            "seed": SEED,
            "rates": {"duplicate": DUPLICATE_RATE, "malformed": MALFORMED_RATE, "transient": TRANSIENT_RATE},
            "queue_config": {**load_ingest_config().queue.model_dump(mode="json"), "lease_seconds": EVAL_LEASE_SECONDS},
            "crash_after_stores": CRASH_AFTER_STORES,
            "crash_exit_code": crash_exit,
            "kill_after_completed_calls": kill_after,
        },
        **measure(run_dir, stream, enqueue_results, timings),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A1 replay eval (offline, from the raw cache).")
    parser.add_argument("--limit", type=int, help="use only the first N fixture events (for a quick check)")
    parser.add_argument("--worker", nargs=2, metavar=("RUN_DIR", "PHASE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker_main(Path(args.worker[0]), args.worker[1])
    else:
        main(args.limit)
