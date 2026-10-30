"""Poll the arXiv API for new and updated papers and record them as events (A1).

Run: uv run python -m copilot.ingest.arxiv_source --n 300
"""

import argparse
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from copilot.infra.ingest_config import load_ingest_config
from copilot.infra.jsonl import append_jsonl
from copilot.ingest.events import ArxivEventModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT_DIR / "data" / "arxiv" / "events.jsonl"

API_URL = "https://export.arxiv.org/api/query"
TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3
# Fixed wait; longer than arXiv's 3 s interval, so a retry never breaks the rate limit.
RETRY_DELAY_SECONDS = 5
# Waiting can fix these: rate limiting (429) and trouble on arXiv's side (5xx).
RETRYABLE_STATUS = {429, 500, 502, 503}

ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_page(categories: list[str], start: int, page_size: int) -> str:
    """One request, no retries: return the raw Atom XML or raise."""
    params = {
        "search_query": " OR ".join(f"cat:{category}" for category in categories),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": page_size,
    }
    response = httpx.get(API_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _is_transient(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS
    # Timeouts and dropped connections never got a status code.
    return isinstance(error, httpx.TransportError)


def fetch_page_with_retries(categories: list[str], start: int, page_size: int) -> str:
    """Retry transient failures up to MAX_ATTEMPTS requests in total; raise anything else at once."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetch_page(categories, start, page_size)
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            if not _is_transient(error) or attempt == MAX_ATTEMPTS:
                raise
            print(f"attempt {attempt} failed ({error!r}); retrying in {RETRY_DELAY_SECONDS} s")
            time.sleep(RETRY_DELAY_SECONDS)


def _text(entry: ET.Element, tag: str) -> str:
    element = entry.find(ATOM + tag)
    if element is None or element.text is None:
        raise ValueError(f"entry has no <{tag}>")
    return element.text


def _split_id(abs_url: str) -> tuple[str, int]:
    # <id> looks like http://arxiv.org/abs/2510.12345v2 (old-style IDs: .../abs/cs/0112017v1).
    if "/abs/" not in abs_url:
        raise ValueError(f"unexpected <id>: {abs_url}")
    paper_id, version = abs_url.split("/abs/", 1)[1].rsplit("v", 1)
    return paper_id, int(version)


def _entry_to_event(entry: ET.Element, polled_at: datetime) -> ArxivEventModel:
    paper_id, version = _split_id(_text(entry, "id"))
    pdf_links = [link.get("href") for link in entry.findall(ATOM + "link") if link.get("title") == "pdf"]
    if not pdf_links:
        raise ValueError("entry has no pdf link")
    return ArxivEventModel(
        event_type="paper_published" if version == 1 else "paper_updated",
        source="arxiv",
        id=paper_id,
        version=version,
        # Raw text, line breaks included: cleaning belongs to the parse step, not the event.
        title=_text(entry, "title"),
        abstract=_text(entry, "summary"),
        authors=[_text(author, "name") for author in entry.findall(ATOM + "author")],
        categories=[c.get("term") for c in entry.findall(ATOM + "category")],
        published=datetime.fromisoformat(_text(entry, "published")),
        updated=datetime.fromisoformat(_text(entry, "updated")),
        pdf_url=pdf_links[0],
        polled_at=polled_at,
    )


def parse_feed(xml_text: str, polled_at: datetime) -> tuple[list[ArxivEventModel], int]:
    """Turn one page into events. Entries that don't fit the model are skipped and counted."""
    root = ET.fromstring(xml_text)
    events, skipped = [], 0
    for entry in root.findall(ATOM + "entry"):
        try:
            events.append(_entry_to_event(entry, polled_at))
        except (ValueError, ValidationError) as error:
            skipped += 1
            print(f"skipped an entry: {error}")
    return events, skipped


def poll(n_events: int, out_path: Path = DEFAULT_OUT) -> dict:
    """Fetch the most recently updated papers until n_events are written to out_path.

    Each raw page is saved next to the events file, so the parser can be re-run offline.
    """
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; pick a new path rather than mixing two polls")
    config = load_ingest_config().arxiv
    pages_dir = out_path.parent / f"{out_path.stem}_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, start = 0, 0, 0
    while written < n_events:
        if start > 0:
            time.sleep(config.request_interval_seconds)
        xml_text = fetch_page_with_retries(config.categories, start, config.page_size)
        polled_at = datetime.now(timezone.utc)
        (pages_dir / f"page_{start:05d}.xml").write_text(xml_text, encoding="utf-8")

        events, n_bad = parse_feed(xml_text, polled_at)
        skipped += n_bad
        if not events and n_bad == 0:
            break  # past the last result
        for event in events[: n_events - written]:
            append_jsonl(out_path, event)
            written += 1
        start += config.page_size

    return {"written": written, "skipped": skipped, "pages": start // config.page_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record arXiv paper events to JSONL.")
    parser.add_argument("--n", type=int, default=300, help="number of events to record")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="events file to create")
    args = parser.parse_args()
    print(poll(args.n, args.out))
