

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from copilot.infra.ingest_config import load_ingest_config
from copilot.ingest.arxiv_source import RETRYABLE_STATUS
from copilot.ingest.event_queue import connect, enqueue_file, now_utc
from copilot.ingest.events import ArxivEventModel
from copilot.ingest.paper import PaperModel, SectionModel
from copilot.ingest.parse import clean_text, html_sections, pdf_sections
from copilot.ingest.worker import PermanentError, TransientError, run_worker

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "arxiv" / "raw"
TIMEOUT_SECONDS = 60  # PDFs can be tens of MB

PAPERS_SCHEMA = """
-- One row per paper: only the highest version is kept.
CREATE TABLE IF NOT EXISTS papers (
    source      TEXT    NOT NULL,
    id          TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    paper_json  TEXT    NOT NULL,   -- the PaperModel
    parsed_from TEXT    NOT NULL CHECK (parsed_from IN ('html', 'pdf', 'qasper')),
    stored_at   TEXT    NOT NULL,
    PRIMARY KEY (source, id)
) STRICT;
"""


def store_paper(conn: sqlite3.Connection, paper: PaperModel, parsed_from: str) -> str:
    """Insert, or replace only if this version is higher. Returns 'stored' or 'stale'."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO papers (source, id, version, paper_json, parsed_from, stored_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (source, id) DO UPDATE SET version = excluded.version,"
            " paper_json = excluded.paper_json, parsed_from = excluded.parsed_from,"
            " stored_at = excluded.stored_at"
            " WHERE excluded.version > papers.version",
            (paper.source, paper.id, paper.version, paper.model_dump_json(), parsed_from, now_utc()),
        )
    return "stored" if cursor.rowcount == 1 else "stale"


def _has_text(sections: list[SectionModel]) -> bool:
    return any(section.paragraphs for section in sections)


class ArxivHandler:
    """Callable handler for run_worker. Raw downloads are cached under data/arxiv/raw,
    so re-running the pipeline (or the replay eval) doesn't fetch the same paper twice."""

    def __init__(self, db_path: Path | None = None, raw_dir: Path = RAW_DIR):
        config = load_ingest_config()
        self.interval = config.arxiv.request_interval_seconds
        self.conn = connect(db_path or config.queue.db_file)
        self.conn.executescript(PAPERS_SCHEMA)
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self.last_outcome: dict = {}  # what the most recent call did, for logging and evals

    def _download(self, url: str, cache: Path) -> bytes | None:
        """The document's bytes, or None if arXiv has no such document (404)."""
        missing = cache.with_name(cache.name + ".404")
        if cache.exists():
            return cache.read_bytes()
        if missing.exists():
            return None
        wait = self._last_request + self.interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = httpx.get(url, timeout=TIMEOUT_SECONDS, follow_redirects=True)
        except httpx.TransportError as error:
            raise TransientError(f"{url}: {error!r}") from error
        finally:
            self._last_request = time.monotonic()
        if response.status_code == 404:
            missing.touch()
            return None
        if response.status_code in RETRYABLE_STATUS:
            raise TransientError(f"{url}: HTTP {response.status_code}")
        if response.is_error:
            raise PermanentError(f"{url}: HTTP {response.status_code}")
        cache.write_bytes(response.content)
        return response.content

    def _sections(self, event: ArxivEventModel) -> tuple[list[SectionModel], str]:
        name = f"{event.id.replace('/', '_')}v{event.version}"
        html = self._download(f"https://arxiv.org/html/{event.id}v{event.version}", self.raw_dir / f"{name}.html")
        if html is not None:
            sections = html_sections(html.decode("utf-8", errors="replace"))
            if _has_text(sections):
                return sections, "html"

        pdf = self._download(event.pdf_url, self.raw_dir / f"{name}.pdf")
        if pdf is None:
            raise PermanentError("no HTML with text, and the PDF is missing (404)")
        try:
            sections = pdf_sections(pdf)
        except Exception as error:
            raise PermanentError(f"PDF could not be parsed: {error!r}") from error
        if not _has_text(sections):
            raise PermanentError("no text found in HTML or PDF")
        return sections, "pdf"

    def __call__(self, event: ArxivEventModel) -> None:
        start = time.monotonic()
        sections, parsed_from = self._sections(event)
        paper = PaperModel(
            source="arxiv",
            id=event.id,
            version=event.version,
            title=clean_text(event.title),
            authors=event.authors,
            abstract=clean_text(event.abstract),
            sections=sections,
            fetched_at=datetime.now(timezone.utc),
        )
        self.last_outcome = {
            "id": event.id,
            "version": event.version,
            "parsed_from": parsed_from,
            "stored": store_paper(self.conn, paper, parsed_from),
            "seconds": round(time.monotonic() - start, 2),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one worker with the real arXiv handler.")
    parser.add_argument("--enqueue", type=Path, help="an events JSONL file to enqueue first")
    args = parser.parse_args()

    db_path = load_ingest_config().queue.db_file
    conn = connect(db_path)
    if args.enqueue:
        print("enqueue:", dict(enqueue_file(conn, args.enqueue)))
    handler = ArxivHandler(db_path)

    def handle_and_log(event: ArxivEventModel) -> None:
        handler(event)
        print(handler.last_outcome, flush=True)

    print("worker:", dict(run_worker(conn, handle_and_log)))
