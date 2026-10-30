"""A1 hand check: section structure of 20 papers, HTML vs PDF, judged by hand.

    uv run python -m evals.a1_handcheck            # writes the checklist (fetches missing PDFs)
    uv run python -m evals.a1_handcheck --score    # summarises the filled-in scores.csv

Picks 20 papers (seed 42) among those the real run parsed from HTML, so both versions
exist for the same paper. Writes evals/a1_handcheck/:
    checklist.md   per paper: links to the real paper, and the headings each parser found
    scores.csv     one row per paper and parser, to fill in by hand
"""

import argparse
import csv
import json
import random
import sqlite3
import time
from pathlib import Path

import httpx

from copilot.infra.ingest_config import load_ingest_config
from copilot.ingest.handler import RAW_DIR
from copilot.ingest.parse import html_sections, pdf_sections

OUT_DIR = Path(__file__).resolve().parent / "a1_handcheck"
SEED = 42
N_PAPERS = 20
COLUMNS = ["paper", "parser", "true_sections", "missed", "extra", "garbled_headings", "nesting_ok", "paragraphs_ok", "notes"]


def _pick_papers() -> list[tuple[str, int]]:
    db = load_ingest_config().queue.db_file
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT id, version FROM papers WHERE parsed_from = 'html' ORDER BY id").fetchall()
    return sorted(random.Random(SEED).sample(rows, N_PAPERS))


def _pdf_bytes(paper_id: str, version: int, interval: float) -> bytes:
    path = RAW_DIR / f"{paper_id.replace('/', '_')}v{version}.pdf"
    if not path.exists():
        time.sleep(interval)  # arXiv's 3 s rule
        response = httpx.get(f"https://arxiv.org/pdf/{paper_id}v{version}", timeout=60, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def _heading_lines(sections) -> list[str]:
    return [f"{s.heading or '(no heading)'}  [{len(s.paragraphs)} para]" for s in sections]


def write_checklist() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    interval = load_ingest_config().arxiv.request_interval_seconds
    papers = _pick_papers()
    parts = [
        "# A1 hand check: section structure, HTML vs PDF",
        "",
        "For each paper, open the real paper (the PDF link) and look at its actual section headings.",
        "Then compare both lists below and fill in **two rows per paper** in `scores.csv`:",
        "",
        "| column | what to write |",
        "|---|---|",
        "| `true_sections` | how many headings the real paper has (sections + subsections + appendix sections; skip Abstract and References) |",
        "| `missed` | real headings this parser did not find |",
        "| `extra` | headings this parser invented (e.g. a table row or sentence taken as a heading) |",
        "| `garbled_headings` | headings found but with wrong text (lost spaces, broken words) |",
        "| `nesting_ok` | `y` if subsections sit under the right parent, else `n` |",
        "| `paragraphs_ok` | open one middle section: `y` if its paragraphs are whole and in order, else `n` |",
        "| `notes` | anything odd |",
        "",
        "Fill in `true_sections` once per paper and copy it into both rows.",
        "",
    ]
    for i, (paper_id, version) in enumerate(papers, 1):
        html = (RAW_DIR / f"{paper_id.replace('/', '_')}v{version}.html").read_text(errors="replace")
        html_secs = html_sections(html)
        pdf_secs = pdf_sections(_pdf_bytes(paper_id, version, interval))
        title = json.loads(
            sqlite3.connect(f"file:{load_ingest_config().queue.db_file}?mode=ro", uri=True)
            .execute("SELECT paper_json FROM papers WHERE id = ?", (paper_id,)).fetchone()[0]
        )["title"]
        parts += [
            f"## {i}. {paper_id}v{version}: {title}",
            "",
            f"[PDF](https://arxiv.org/pdf/{paper_id}v{version}) · [HTML](https://arxiv.org/html/{paper_id}v{version})",
            "",
            f"**HTML parser** ({len(html_secs)} sections)",
            "```",
            *_heading_lines(html_secs),
            "```",
            f"**PDF parser** ({len(pdf_secs)} sections)",
            "```",
            *_heading_lines(pdf_secs),
            "```",
            "",
        ]
        print(f"{i:2}. {paper_id}v{version}: HTML {len(html_secs)} sections, PDF {len(pdf_secs)}")
    (OUT_DIR / "checklist.md").write_text("\n".join(parts), encoding="utf-8")

    scores = OUT_DIR / "scores.csv"
    if scores.exists():
        print(f"{scores} already exists; not overwritten")
    else:
        with scores.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for paper_id, version in papers:
                for parser in ("html", "pdf"):
                    writer.writerow([f"{paper_id}v{version}", parser] + [""] * (len(COLUMNS) - 2))
    print(f"wrote {OUT_DIR}")


def score() -> None:
    rows = list(csv.DictReader((OUT_DIR / "scores.csv").open()))
    blank = [r["paper"] for r in rows if not r["true_sections"].strip()]
    if blank:
        raise SystemExit(f"{len(blank)} rows not filled in yet, e.g. {blank[0]}")
    summary = {}
    for parser in ("html", "pdf"):
        mine = [r for r in rows if r["parser"] == parser]
        true = sum(int(r["true_sections"]) for r in mine)
        missed = sum(int(r["missed"]) for r in mine)
        extra = sum(int(r["extra"]) for r in mine)
        found = true - missed
        summary[parser] = {
            "papers": len(mine),
            "heading_recall": round(found / true, 3),
            "heading_precision": round(found / (found + extra), 3) if found + extra else None,
            "garbled_headings": sum(int(r["garbled_headings"]) for r in mine),
            "nesting_ok": f"{sum(r['nesting_ok'].strip().lower() == 'y' for r in mine)}/{len(mine)}",
            "paragraphs_ok": f"{sum(r['paragraphs_ok'].strip().lower() == 'y' for r in mine)}/{len(mine)}",
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A1 hand check of section structure.")
    parser.add_argument("--score", action="store_true", help="summarise the filled-in scores.csv")
    args = parser.parse_args()
    score() if args.score else write_checklist()
