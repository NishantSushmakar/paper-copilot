"""Research Guru demo: swipe through new papers, then chat with the ones you saved.

    uv run --with streamlit streamlit run ui/demo.py

Three screens, kept in session state (nothing is saved to disk):
    welcome   the greeting and today's count
    discover  one card at a time; Skip (← key) or Save (→ key)
    chats     the saved papers as chats; each paper greets you and answers with [n] citations

The papers are the 10 newest ingested arXiv papers (ui/data/sample_papers.json, with cards
written by hand from each abstract until model cards are ready). Every answer comes from the
real pipeline (ui/answers.py): the suggested questions were answered ahead of time by
ui/build_demo_answers.py, and any other question runs live, which needs Ollama.
"""

import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

UI_DIR = Path(__file__).resolve().parent
PAPERS_PATH = UI_DIR / "data" / "sample_papers.json"
ANSWERS_PATH = UI_DIR / "data" / "demo_answers.json"
STYLES = (UI_DIR / "style.css", UI_DIR / "demo.css")
TYPING_SECONDS = 0.7  # the "typing…" pause before a stored reply streams in
WORD_SECONDS = 0.03  # per word while a reply streams
MATCH_OVERLAP = 0.6  # share of a suggested question's words a typed question must contain to reuse its answer
STOPWORDS = {"a", "an", "the", "is", "it", "of", "to", "in", "on", "do", "does", "did", "what", "how", "which",
             "who", "why", "you", "your", "this", "that", "for", "and", "or", "are", "was", "were", "be", "they"}
NOT_COVERED = "My pages don't answer that, so I won't guess."
NO_MODEL = "I can't reach the model right now, so I can't read my pages for this one. Try a suggested question."
# The sentence where a paper states its contribution, quoted on the card.
CLAIM = re.compile(r"\b(we|this paper|this work)\b.{0,40}?\b(introduce|propose|present|develop|construct|build|tackle)",
                   re.IGNORECASE)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def claim_sentence(abstract: str) -> str:
    """The abstract's own "we introduce..." sentence, verbatim; the first sentence if there is none."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z$])", abstract.strip())
    return next((s for s in sentences if CLAIM.search(s)), sentences[0])


def demo_paper(item: dict) -> dict:
    paper, card = item["paper"], item["card"]
    abs_url = f"https://arxiv.org/abs/{paper['id']}v{paper['version']}"
    return {
        "id": paper["id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "date": datetime.fromisoformat(paper["fetched_at"].replace("Z", "+00:00")).strftime("%-d %b %Y"),
        "topics": [e["names"][0] for e in card["tasks"]][:2],
        "claim": claim_sentence(paper["abstract"]),
        "result": card["results"][0] if card["results"] else None,
        "datasets": [e["names"][0] for e in card["datasets"]],
        "methods": [e["names"][0] for e in card["methods"]],
        "abs_url": abs_url,
        "pdf_url": f"https://arxiv.org/pdf/{paper['id']}v{paper['version']}",
        "html_url": f"https://arxiv.org/html/{paper['id']}v{paper['version']}",
    }


@st.cache_data
def load_papers() -> list[dict]:
    return [demo_paper(item) for item in json.loads(PAPERS_PATH.read_text(encoding="utf-8"))["papers"]]


@st.cache_data
def load_answers() -> dict[str, list[dict]]:
    return json.loads(ANSWERS_PATH.read_text(encoding="utf-8")) if ANSWERS_PATH.exists() else {}


def state() -> dict:
    defaults = {"screen": "welcome", "index": 0, "saved": [], "last": None, "chats": {}, "open_chat": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return st.session_state


def short_title(paper: dict) -> str:
    return paper["title"].split(":")[0]


# ---------- actions ----------

def go(screen: str) -> None:
    st.session_state.screen = screen


def swipe(paper: dict, direction: str) -> None:
    s = st.session_state
    s.last = {"paper": paper, "direction": direction}
    s.index += 1
    if direction == "right":
        s.saved.append(paper["id"])
        s.toast = f"It's a match! “{short_title(paper)}” is in your chats."


def restart() -> None:
    for key in ("screen", "index", "saved", "last", "chats", "open_chat"):
        st.session_state.pop(key, None)


def open_chat(paper_id: str) -> None:
    s = st.session_state
    s.screen, s.open_chat = "chats", paper_id
    if paper_id not in s.chats:
        s.chats[paper_id] = [{"role": "paper", "kind": "greeting", "new": True}]


def ask(paper: dict, question: str) -> None:
    st.session_state.chats[paper["id"]] += [{"role": "user", "text": question},
                                            {"role": "paper", "kind": "answer", "question": question, "new": True}]


# ---------- answers ----------

def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOPWORDS


def stored_answer(paper: dict, question: str) -> dict | None:
    """The pre-computed answer whose question the typed one matches, if any."""
    asked = words(question)
    best, best_overlap = None, 0.0
    for item in load_answers().get(paper["id"], []):
        target = words(item["q"])
        overlap = len(asked & target) / len(target) if target else 0.0
        if overlap > best_overlap:
            best, best_overlap = item, overlap
    return best if best_overlap >= MATCH_OVERLAP else None


def live_answer(paper: dict, question: str) -> dict | None:
    try:
        if str(UI_DIR.parent) not in sys.path:  # Streamlit puts ui/ on the path, not the repo root
            sys.path.insert(0, str(UI_DIR.parent))
        from ui.answers import answer

        return answer(paper["id"], question)
    except Exception:  # the model server is down, or the paper isn't in this machine's store
        return None


def badge(n: int) -> str:
    return f":blue-badge[{n}]"


def md(text: str) -> str:
    """Model text as literal Markdown: "$" would start LaTeX."""
    return text.replace("$", r"\$")


def reply_markdown(paper: dict, message: dict) -> str:
    if message["kind"] == "greeting":
        return (f"Hi! I'm **{md(short_title(paper))}**. Thanks for the match. Ask me anything about me: I answer only "
                "from my own pages, and every answer shows the passage it came from.")
    result = message.get("answer")
    if result is None:
        return NO_MODEL
    if result["abstained"]:
        return NOT_COVERED
    # a label the model repeats ([1, 1, 2]) gets one badge
    return " ".join(md(text) + " " + "".join(badge(n) for n in dict.fromkeys(labels)) for text, labels in result["sentences"])


def stream(markdown: str):
    for token in re.split(r"(\s+)", markdown):
        yield token
        if token.strip():
            time.sleep(WORD_SECONDS)


def sources(paper: dict, result: dict | None) -> None:
    if not result or result["abstained"]:
        return
    cited = sorted({n for _, labels in result["sentences"] for n in labels})
    if not cited:
        return
    passages = {p["label"]: p for p in result["passages"]}
    items = "".join(
        f'<li><span class="gs-src-n">{n}</span><div><b>{esc(passages[n]["section"] or "Untitled section")}</b>'
        f'<p>{esc(passages[n]["text"])}</p></div></li>'
        for n in cited
    )
    st.html(f'<details class="gs-sources"><summary>Sources · {len(cited)} passage{"s" if len(cited) > 1 else ""} '
            f'from this paper</summary><ol>{items}</ol>'
            f'<a class="gs-src-link" href="{esc(paper["html_url"])}" target="_blank" rel="noopener">'
            "Check them in the full paper ↗</a></details>")


# ---------- pieces ----------

def top_bar(saved: int) -> None:
    st.html(
        '<nav class="rg-bar" aria-label="Research Guru">'
        '<span class="rg-brand"><span class="rg-mark" aria-hidden="true"></span><b>Research Guru</b></span>'
        f'<span class="rg-scope"><span class="gs-demo">Preview</span> arXiv cs.CL · <b>{saved}</b> saved</span>'
        "</nav>"
    )


def paper_links(paper: dict) -> str:
    return (f'<a class="gs-link" href="{esc(paper["abs_url"])}" target="_blank" rel="noopener">Read on arXiv ↗</a>'
            f'<a class="gs-link" href="{esc(paper["pdf_url"])}" target="_blank" rel="noopener">PDF ↗</a>')


def result_html(paper: dict) -> str:
    r = paper["result"]
    if r is None:
        return '<div class="rg-result none">No single figure reported in the abstract.</div>'
    return (f'<div class="rg-result"><div class="fig">{esc(r["raw_value"])}</div>'
            f'<div class="what"><b>{esc(r["method"])}</b>{esc(r["metric"])} · {esc(r["dataset"])}</div></div>')


def card_html(paper: dict, css: str = "") -> str:
    topics = "".join(f'<span class="rg-topic">{esc(t)}</span>' for t in paper["topics"])
    chips = lambda names: ("".join(f'<span class="rg-chip">{esc(n)}</span>' for n in names[:3])
                           or '<span class="rg-dash">none named</span>')
    authors = ", ".join(paper["authors"][:4]) + (f" +{len(paper['authors']) - 4}" if len(paper["authors"]) > 4 else "")
    return (
        f'<article class="gs-card {css}">'
        f'<div class="rg-card-top"><div class="rg-topics">{topics}</div>'
        f'<a class="gs-card-link" href="{esc(paper["abs_url"])}" target="_blank" rel="noopener">'
        f'arXiv {esc(paper["id"])} ↗</a></div>'
        f'<h3>{esc(paper["title"])}</h3>'
        f'<p class="rg-authors">{esc(authors)}</p>'
        f'<p class="rg-quote"><small>In the authors’ words</small>{esc(paper["claim"])}</p>'
        + result_html(paper)
        + f'<dl class="rg-chips"><dt>Datasets</dt><dd>{chips(paper["datasets"])}</dd>'
        f'<dt>Methods</dt><dd>{chips(paper["methods"])}</dd></dl>'
        "</article>"
    )


def ghost_html(last: dict | None) -> str:
    if not last:
        return ""
    stamp = "Saved" if last["direction"] == "right" else "Skipped"
    return card_html(last["paper"], f'ghost out-{last["direction"]}').replace(
        "</article>", f'<span class="gs-stamp {last["direction"]}">{stamp}</span></article>')


def deck_html(papers: list[dict], index: int, last: dict | None) -> str:
    behind = min(2, len(papers) - index - 1)
    layers = "".join(f'<div class="gs-behind b{i}"></div>' for i in range(behind, 0, -1))
    top = card_html(papers[index], "enter" if last else "")
    return f'<div class="gs-deck">{layers}{top}{ghost_html(last)}</div>'


def progress(index: int, total: int) -> str:
    dots = "".join(f'<span class="{"done" if i < index else "now" if i == index else ""}"></span>' for i in range(total))
    return f'<div class="gs-progress"><span>Paper {min(index + 1, total)} of {total}</span><div class="gs-dots">{dots}</div></div>'


# ---------- screens ----------

def welcome_screen(papers: list[dict]) -> None:
    with st.container(key="band"):
        top_bar(0)
        text_col, art_col = st.columns([3, 2], vertical_alignment="center")
        with text_col:
            st.html(
                '<div class="gs-welcome"><p class="gs-kicker">Good morning</p>'
                "<h1>Welcome to Research Guru</h1>"
                f"<p>Here are the {len(papers)} newest papers in cs.CL. Swipe right to keep a paper, left to skip it. "
                "Then chat with the ones you kept: each paper answers from its own pages, with citations.</p>"
                '<ol class="gs-steps"><li><b>Swipe</b> through the newest papers</li>'
                "<li><b>Match</b> with the ones worth your time</li>"
                "<li><b>Chat</b> with them, every answer cited</li></ol></div>"
            )
            st.button("Start swiping", type="primary", shortcut="Enter", on_click=go, args=("discover",), key="start")
        with art_col:
            fan = "".join(
                f'<div class="gs-fan-card f{i}"><span class="rg-topic">{esc(p["topics"][0] if p["topics"] else "new")}</span>'
                f'<b>{esc(short_title(p))}</b>'
                + (f'<span class="gs-fan-fig">{esc(p["result"]["raw_value"])}</span>' if p["result"] else "")
                + "</div>"
                for i, p in reversed(list(enumerate(papers[:3])))
            )
            st.html(f'<div class="gs-fan" aria-hidden="true">{fan}</div>')


def nav(saved: int) -> None:
    with st.container(horizontal=True, key="nav"):
        screen = st.session_state.screen
        st.button("Discover", type="primary" if screen == "discover" else "secondary", on_click=go, args=("discover",))
        st.button(f"Chats · {saved}", type="primary" if screen == "chats" else "secondary", on_click=go, args=("chats",))


def discover_screen(papers: list[dict]) -> None:
    s = state()
    with st.container(key="band"):
        top_bar(len(s.saved))
        nav(len(s.saved))
    with st.container(key="stage"):
        if s.index >= len(papers):
            by_id = {p["id"]: p for p in papers}
            done_panel([by_id[i] for i in s.saved], len(papers), s.last)
            s.last = None
            return
        paper = papers[s.index]
        st.html(progress(s.index, len(papers)))
        st.html(deck_html(papers, s.index, s.last))
        s.last = None
        with st.container(horizontal=True, horizontal_alignment="center", key="swipe_actions"):
            st.button("✕  Skip", key="skip", shortcut="Left", on_click=swipe, args=(paper, "left"),
                      help="Skip this paper")
            st.button("♥  Save", key="save", type="primary", shortcut="Right", on_click=swipe, args=(paper, "right"),
                      help="Save this paper and chat with it later")
        st.html('<p class="gs-hint">Use ← and → on your keyboard to swipe</p>')


def done_panel(saved: list[dict], total: int, last: dict | None) -> None:
    kept = "".join(f'<li><span class="rg-topic">{esc(p["topics"][0] if p["topics"] else "paper")}</span>'
                   f"{esc(short_title(p))}</li>" for p in saved)
    st.html(f'<div class="gs-deck">{ghost_html(last)}<div class="gs-done"><h2>You’re all caught up</h2>'
            f"<p>You went through all {total} new papers and matched with <b>{len(saved)}</b>. "
            "They’re waiting in your chats.</p>"
            + (f'<ul class="gs-peek">{kept}</ul>' if saved else "") + "</div></div>")
    with st.container(horizontal=True, horizontal_alignment="center", key="done_actions"):
        if saved:
            st.button("Open chats", type="primary", shortcut="Enter", on_click=go, args=("chats",))
        st.button("Start over", on_click=restart)


def chats_screen(papers: list[dict]) -> None:
    s = state()
    by_id = {p["id"]: p for p in papers}
    with st.container(key="band"):
        top_bar(len(s.saved))
        nav(len(s.saved))
    if not s.saved:
        st.html('<div class="rg-empty">No chats yet. Swipe right on a paper to start one.</div>')
        st.button("Discover papers", type="primary", on_click=go, args=("discover",))
        return
    if s.open_chat not in s.saved:
        open_chat(s.saved[0])

    list_col, chat_col = st.columns([1, 2], gap="medium")
    with list_col:
        with st.container(key="chat_list"):
            st.html('<h2 class="gs-list-title">Your papers</h2>')
            for paper_id in s.saved:
                paper = by_id[paper_id]
                unread = "" if paper_id in s.chats else "  :blue[● new]"
                st.button(short_title(paper) + unread, key=f"chat_{paper_id}",
                          type="primary" if paper_id == s.open_chat else "secondary",
                          on_click=open_chat, args=(paper_id,), width="stretch")
    with chat_col:
        chat_panel(by_id[s.open_chat])


def chat_panel(paper: dict) -> None:
    messages = st.session_state.chats[paper["id"]]
    with st.container(key="chat"):
        topic = f'<span class="rg-topic">{esc(paper["topics"][0])}</span>' if paper["topics"] else ""
        st.html(f'<div class="gs-chat-head">{topic}<h2>{esc(paper["title"])}</h2>'
                f'<p>{esc(", ".join(paper["authors"]))} · {esc(paper["date"])}</p>'
                f'<div class="gs-links">{paper_links(paper)}</div></div>')
        for message in messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(md(message["text"]))
                continue
            with st.chat_message("assistant", avatar=":material/description:"):
                if message.pop("new", False):
                    if message["kind"] == "answer":
                        stored = stored_answer(paper, message["question"])
                        if stored is None:
                            with st.spinner("Reading my pages…", show_time=True):
                                message["answer"] = live_answer(paper, message["question"])
                        else:
                            message["answer"] = stored
                            with st.spinner("Reading my pages…", show_time=False):
                                time.sleep(TYPING_SECONDS)
                    else:
                        with st.spinner("typing…", show_time=False):
                            time.sleep(TYPING_SECONDS)
                    st.write_stream(stream(reply_markdown(paper, message)))
                else:
                    st.markdown(reply_markdown(paper, message))
                if message["kind"] == "answer":
                    sources(paper, message.get("answer"))

        asked = {m["answer"]["q"] for m in messages if m.get("answer")}
        open_questions = [a["q"] for a in load_answers().get(paper["id"], []) if a["q"] not in asked]
        if open_questions:
            with st.container(horizontal=True, key="suggestions"):
                for i, q in enumerate(open_questions):
                    st.button(q, key=f"q_{paper['id']}_{i}", on_click=ask, args=(paper, q))
    typed = st.chat_input(f"Ask “{short_title(paper)}” a question")
    if typed:
        ask(paper, typed)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Research Guru", layout="wide")
    st.html("<style>" + "\n".join(p.read_text(encoding="utf-8") for p in STYLES) + "</style>")
    papers = load_papers()
    s = state()
    if toast := s.pop("toast", None):
        st.toast(toast, icon=":material/favorite:")
    {"welcome": welcome_screen, "discover": discover_screen, "chats": chats_screen}[s.screen](papers)
    st.html('<p class="gs-footer">Papers: the newest arXiv cs.CL papers in the store. Cards are written by hand from '
            "each abstract for now. Answers come from the pipeline: the paper’s 5 most relevant passages (dense "
            "retrieval) and Qwen3-8B, citing each claim; suggested questions were answered ahead of time.</p>")


main()
