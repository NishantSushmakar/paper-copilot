"""Research Guru: the Streamlit front end of the Paper Copilot project.

    uv run --with streamlit streamlit run ui/app.py

Two screens, picked by the `paper` query parameter:
    /                  the latest papers, one card each, with search, views and a reading list
    /?paper=<arXiv ID> one paper: its results, entities and abstract

Each card is built for a quick keep-or-skip decision (topic first, then the paper's own claim,
then its first reported figure); the paper screen is for reading. Everything shown is quoted
from the paper or its card; nothing is summarised by a model on the fly.

The data is ui/data/sample_papers.json (built by ui/build_sample_data.py): real arXiv metadata
from the A1 store, with sample cards until the A2 extraction runs on these papers. The reading
list is kept in ui/data/reading_list.json (local, not committed).
"""

import html
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

UI_DIR = Path(__file__).resolve().parent
DATA_PATH = UI_DIR / "data" / "sample_papers.json"
READING_LIST_PATH = UI_DIR / "data" / "reading_list.json"
STYLE_PATH = UI_DIR / "style.css"
CARD_TOPICS = 2  # task chips at the top of a card
CARD_CHIPS = 3  # dataset and method chips per line on a card
VIEWS = {"all": "All papers", "results": "With results", "saved": "Reading list"}
# The sentence where a paper states its contribution, quoted on the card.
CLAIM = re.compile(r"\b(we|this paper|this work)\b.{0,40}?\b(introduce|propose|present|develop|construct|build|tackle)",
                   re.IGNORECASE)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


@st.cache_data
def load_data() -> dict:
    """{"store_size": arXiv papers in the A1 store, "papers": the latest ones with their cards}"""
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


# ---------- reading list ----------

def reading_list() -> list[str]:
    try:
        return json.loads(READING_LIST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def toggle_saved(paper_id: str) -> None:
    saved = reading_list()
    saved = [p for p in saved if p != paper_id] if paper_id in saved else saved + [paper_id]
    READING_LIST_PATH.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")


def open_paper(paper_id: str) -> None:
    st.query_params["paper"] = paper_id


def close_paper() -> None:
    st.query_params.clear()


# ---------- text helpers ----------

def fetched_date(paper: dict) -> str:
    return datetime.fromisoformat(paper["fetched_at"].replace("Z", "+00:00")).strftime("%-d %b %Y")


def author_line(authors: list[str], shown: int = 3) -> str:
    if len(authors) <= shown:
        return ", ".join(authors)
    return f"{', '.join(authors[:shown])} +{len(authors) - shown}"


def claim_sentence(abstract: str) -> str:
    """The abstract's own "we introduce..." sentence, verbatim; the first sentence if there is none."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z$])", abstract.strip())
    return next((s for s in sentences if CLAIM.search(s)), sentences[0])


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def matches(item: dict, query: str) -> bool:
    card = item["card"]
    haystack = [item["paper"]["title"], item["paper"]["id"], *item["paper"]["authors"]]
    for field in ("tasks", "datasets", "methods", "metrics"):
        haystack += [name for entity in card[field] for name in entity["names"]]
    return any(query in text.casefold() for text in haystack)


# ---------- pieces ----------

def top_bar(store_size: int, saved: int) -> str:
    return (
        '<nav class="rg-bar" aria-label="Research Guru">'
        '<a class="rg-brand" href="./" target="_self"><span class="rg-mark" aria-hidden="true"></span>'
        "<b>Research Guru</b></a>"
        f'<span class="rg-scope">arXiv cs.CL · <b>{store_size}</b> papers read · <b>{saved}</b> on your reading list</span>'
        "</nav>"
    )


def topics(card: dict, limit: int | None = None) -> str:
    names = [e["names"][0] for e in card["tasks"]][:limit]
    return '<div class="rg-topics">' + "".join(f'<span class="rg-topic">{esc(n)}</span>' for n in names) + "</div>"


def chip_line(label: str, entities: list[dict]) -> str:
    if not entities:
        return f'<dt>{label}</dt><dd><span class="rg-dash">none named</span></dd>'
    chips = "".join(f'<span class="rg-chip">{esc(e["names"][0])}</span>' for e in entities[:CARD_CHIPS])
    more = len(entities) - CARD_CHIPS
    if more > 0:
        chips += f'<span class="rg-chip more">+{more}</span>'
    return f"<dt>{label}</dt><dd>{chips}</dd>"


def result_block(results: list[dict]) -> str:
    if not results:
        return '<div class="rg-result none">No single figure reported in the abstract.</div>'
    first = results[0]
    more = (f'<div class="more">+{plural(len(results) - 1, "more result")} on the card</div>'
            if len(results) > 1 else "")
    return (
        '<div class="rg-result">'
        f'<div class="fig">{esc(first["raw_value"])}</div>'
        f'<div class="what"><b>{esc(first["method"])}</b>{esc(first["metric"])} · {esc(first["dataset"])}</div>'
        f"{more}</div>"
    )


def card_html(item: dict, is_saved: bool) -> str:
    paper, card = item["paper"], item["card"]
    return (
        '<article class="rg-card">'
        + ('<span class="rg-saved-flag" hidden></span>' if is_saved else "")
        + '<div class="rg-card-top">' + topics(card, CARD_TOPICS)
        + f'<span class="rg-id">{esc(paper["id"])}</span></div>'
        f'<h3><a href="?paper={esc(paper["id"])}" target="_self">{esc(paper["title"])}</a></h3>'
        f'<p class="rg-authors">{esc(author_line(paper["authors"]))}</p>'
        f'<p class="rg-quote"><small>In the authors’ words</small>{esc(claim_sentence(paper["abstract"]))}</p>'
        + result_block(card["results"])
        + '<dl class="rg-chips">' + chip_line("Datasets", card["datasets"]) + chip_line("Methods", card["methods"])
        + "</dl></article>"
    )


def paper_card(item: dict, saved: list[str]) -> None:
    paper_id = item["paper"]["id"]
    key = paper_id.replace(".", "_")
    is_saved = paper_id in saved
    with st.container(key=f"card_{key}"):
        st.html(card_html(item, is_saved))
        with st.container(horizontal=True, key=f"actions_{key}"):
            st.button("Open card", key=f"open_{key}", type="primary", on_click=open_paper, args=(paper_id,))
            st.link_button("arXiv ↗", f"https://arxiv.org/abs/{paper_id}")
            st.button("Saved ✓" if is_saved else "Save", key=f"save_{key}", on_click=toggle_saved, args=(paper_id,),
                      help="Remove from your reading list" if is_saved else "Add to your reading list")


def sample_notice(items: list[dict]) -> None:
    if any(item["card_source"] == "sample" for item in items):
        st.html('<div class="rg-notice"><b>Sample cards.</b> Written by hand from each abstract, with every figure '
                "quoted from it. The model-extracted cards replace them once the A2 extraction runs on these papers.</div>")


# ---------- screens ----------

def list_screen(data: dict, saved: list[str]) -> None:
    items = data["papers"]
    with_results = sum(bool(i["card"]["results"]) for i in items)
    with_code = sum(bool(i["card"]["code_url"]) for i in items)
    newest = max(fetched_date(i["paper"]) for i in items)

    with st.container(key="band"):
        st.html(top_bar(data["store_size"], len(saved)))
        st.html(
            '<div class="rg-hero"><h1>New in cs.CL</h1>'
            f"<p>The {len(items)} newest papers, fetched {newest}. Each card shows what the paper is about, "
            "what the authors say they did, and the first figure they report.</p>"
            '<div class="rg-stats">'
            f'<span class="rg-stat"><b>{len(items)}</b> papers</span>'
            f'<span class="rg-stat"><b>{with_results}</b> report figures</span>'
            f'<span class="rg-stat"><b>{with_code}</b> link to code</span>'
            "</div></div>"
        )
        search_col, view_col = st.columns([3, 2], vertical_alignment="bottom")
        with search_col:
            query = st.text_input("Search", placeholder="Title, author, task, dataset or method")
        with view_col:
            counts = {"all": len(items), "results": with_results, "saved": len(saved)}
            view = st.segmented_control(
                "Show", options=list(VIEWS), default="all", key="view",
                format_func=lambda v: f"{VIEWS[v]} · {counts[v]}",
            ) or "all"

    sample_notice(items)

    shown = items
    if view == "results":
        shown = [i for i in shown if i["card"]["results"]]
    elif view == "saved":
        shown = [i for i in shown if i["paper"]["id"] in saved]
    query = query.strip().casefold()
    if query:
        shown = [i for i in shown if matches(i, query)]

    st.html(f'<p class="rg-count">Showing {plural(len(shown), "paper")}, newest first</p>')
    if not shown:
        message = ("Your reading list is empty. Save a paper from its card to keep it here."
                   if view == "saved" and not query else "No paper matches this search.")
        st.html(f'<div class="rg-empty">{message}</div>')
        return
    for pair in (shown[i:i + 2] for i in range(0, len(shown), 2)):
        columns = st.columns(2)
        for column, item in zip(columns, pair):
            with column:
                paper_card(item, saved)


def detail_screen(data: dict, item: dict, saved: list[str]) -> None:
    paper, card = item["paper"], item["card"]
    paper_id = paper["id"]
    is_saved = paper_id in saved
    card_from = "sample card from the abstract" if item["card_source"] == "sample" else "card extracted by the model"

    with st.container(key="band"):
        st.html(top_bar(data["store_size"], len(saved)))
        st.button("← All papers", type="tertiary", on_click=close_paper)
        st.html(
            '<div class="rg-paper-hero">' + topics(card)
            + f'<h1>{esc(paper["title"])}</h1>'
            f'<p class="rg-authors">{esc(", ".join(paper["authors"]))}</p>'
            f'<p class="meta">arXiv {esc(paper_id)} v{paper["version"]} · fetched {fetched_date(paper)} · '
            f'{plural(paper["sections"], "section")} · {card_from}</p></div>'
        )
        with st.container(horizontal=True, key="paper_actions"):
            st.link_button("Read on arXiv ↗", f"https://arxiv.org/abs/{paper_id}", type="primary")
            st.link_button("PDF ↗", f"https://arxiv.org/pdf/{paper_id}")
            if card["code_url"]:
                st.link_button("Code ↗", card["code_url"])
            st.button("Saved ✓" if is_saved else "Save to reading list", on_click=toggle_saved, args=(paper_id,))

    sample_notice([item])
    main_col, side_col = st.columns([2, 1], gap="medium")
    with main_col:
        if card["results"]:
            rows = "".join(
                f'<tr><td>{esc(r["method"])}</td><td class="ctx">{esc(r["dataset"])}</td>'
                f'<td class="ctx">{esc(r["metric"])}</td><td class="num">{esc(r["raw_value"])}</td></tr>'
                for r in card["results"]
            )
            results = ('<table class="rg-table"><thead><tr><th>Method</th><th>Dataset</th><th>Metric</th>'
                       f'<th class="num">Value</th></tr></thead><tbody>{rows}</tbody></table>')
        else:
            results = '<p class="rg-dash">No single figure reported in the abstract.</p>'
        st.html(f'<section class="rg-panel"><h2>Reported results<span>{plural(len(card["results"]), "result")}'
                f"</span></h2>{results}</section>")
        st.html(f'<section class="rg-panel"><h2>Abstract</h2><p class="rg-abstract">{esc(paper["abstract"])}</p></section>')
    with side_col:
        def group(title: str, entities: list[dict], css: str = "") -> str:
            if not entities:
                return f'<div><h3>{title}</h3><span class="rg-dash">none named</span></div>'
            chips = "".join(
                f'<li class="{css}">{esc(e["names"][0])}'
                + (f' <small>· {esc(", ".join(e["names"][1:]))}</small>' if len(e["names"]) > 1 else "")
                + "</li>"
                for e in entities
            )
            return f"<div><h3>{title} · {len(entities)}</h3><ul>{chips}</ul></div>"

        st.html(
            '<section class="rg-panel"><h2>On this card</h2><div class="rg-ents">'
            + group("Tasks", card["tasks"], "topic") + group("Datasets", card["datasets"])
            + group("Methods", card["methods"]) + group("Metrics", card["metrics"])
            + "</div></section>"
        )


def missing_screen(data: dict, paper_id: str, saved: list[str]) -> None:
    with st.container(key="band"):
        st.html(top_bar(data["store_size"], len(saved)))
        st.button("← All papers", type="tertiary", on_click=close_paper)
        st.html(f'<div class="rg-paper-hero"><h1>No paper {esc(paper_id)}</h1>'
                '<p class="meta">It isn’t among the latest papers.</p></div>')


def main() -> None:
    st.set_page_config(page_title="Research Guru", layout="wide")
    st.html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")
    data = load_data()
    saved = reading_list()

    paper_id = st.query_params.get("paper")
    if paper_id is None:
        list_screen(data, saved)
        return
    item = next((i for i in data["papers"] if i["paper"]["id"] == paper_id), None)
    if item:
        detail_screen(data, item, saved)
    else:
        missing_screen(data, paper_id, saved)


main()
