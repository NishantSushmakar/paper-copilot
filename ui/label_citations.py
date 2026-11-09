"""Citation labeller: one sentence per page, its cited chunks beside it, labels saved on every click.

    uv run --with streamlit streamlit run ui/label_citations.py

Reads evals/a3_citation_labels/items.json and writes evals/a3_citation_labels/labels.csv
(made by `uv run python -m evals.a3_citation_labels`). The labelling rules are the ones in
reports/a03.md §9; the app only displays the items and stores what you pick.
"""

import csv
import html
import json
import re
from pathlib import Path

import streamlit as st

LABEL_DIR = Path(__file__).resolve().parent.parent / "evals" / "a3_citation_labels"
ITEMS = LABEL_DIR / "items.json"
LABELS = LABEL_DIR / "labels.csv"
CHOICES = ["y", "n"]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
:root { --sky-50:#f0f9ff; --sky-100:#e0f2fe; --sky-200:#bae6fd; --sky-500:#0ea5e9; --sky-600:#0284c7;
        --sky-800:#075985; --ink:#0c2a43; --muted:#4b6478; --amber-100:#fef3c7; --amber-700:#b45309; }
html, body, [class*="st-"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
/* Streamlit's icons are ligatures in an icon font: with the text font they show as words like "arrow_drop_down" */
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"] { font-family: 'Material Symbols Rounded' !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * { color: #fff !important; }
.stApp { background: var(--sky-50); }
.lc-band { background: linear-gradient(135deg, var(--sky-100), #fff); border: 1px solid var(--sky-200);
           border-radius: 14px; padding: 18px 22px; margin-bottom: 14px; }
.lc-kicker { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--sky-600); font-weight: 600; }
.lc-question { font-family: 'Source Serif 4', serif; font-size: 22px; line-height: 1.35; margin: 6px 0 0; }
.lc-box { background: #fff; border: 1px solid var(--sky-200); border-radius: 12px; padding: 14px 18px; margin: 8px 0; }
.lc-label { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
.lc-answer { white-space: pre-wrap; line-height: 1.55; font-size: 14px; color: var(--muted); }
.lc-sentence { font-family: 'Source Serif 4', serif; font-size: 20px; line-height: 1.45;
               border-left: 4px solid var(--sky-500); padding-left: 14px; }
.lc-cite { background: var(--sky-100); color: var(--sky-800); border-radius: 6px; padding: 0 4px; font-weight: 600; }
.lc-cite-this { background: var(--amber-100); color: var(--amber-700); }
.lc-chunk { white-space: pre-wrap; line-height: 1.6; font-size: 14.5px; max-height: 420px; overflow-y: auto; }
.lc-chunk-head { font-weight: 600; color: var(--sky-800); margin-bottom: 6px; }
.lc-rules li { margin-bottom: 4px; }
"""

RULES = """
<ul class="lc-rules">
<li>Judge against the <b>original question and answer</b>.</li>
<li>A chunk supports the claim only if it <b>states</b> it. A paraphrase counts; an inference doesn't
(“We train on WMT14 En–De” does not state “they use a German dataset”).</li>
<li><b>Partial support is n</b> (a chunk with SQuAD only, for “SQuAD and TriviaQA”). The removal rule rescues it.</li>
<li><b>Together</b>: do all the cited chunks, read as one, state the whole sentence?</li>
<li><b>Without this one</b> (3+ citations only): do the <i>other</i> cited chunks still state it if this chunk is removed?</li>
<li>A sentence that's broken or makes no sense alone: label as you can and say so in the notes.</li>
</ul>
"""


@st.cache_data
def load_items() -> list[dict]:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


def load_rows() -> list[dict]:
    with open(LABELS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict]) -> None:
    tmp = LABELS.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(LABELS)  # never a half-written labels file


def marked(text: str, this_sentence: set[int]) -> str:
    """Escape the answer and colour its [n] markers; this sentence's chunks in amber."""
    def cite(match):
        labels = {int(n) for n in match.group(1).split(",")}
        cls = "lc-cite lc-cite-this" if labels & this_sentence else "lc-cite"
        return f'<span class="{cls}">{match.group(0)}</span>'
    return CITATION.sub(cite, html.escape(text))


def is_done(row: dict) -> bool:
    return row["supports"] in CHOICES and row["supports_without_this"] in CHOICES + ["-"]


def main() -> None:
    st.set_page_config(page_title="Citation labels", layout="wide")
    st.html(f"<style>{STYLE}</style>")
    if not ITEMS.exists() or not LABELS.exists():
        st.error("No items yet. Run `uv run python -m evals.a3_citation_labels` first.")
        return

    items = load_items()
    rows = load_rows()
    by_item = {row["item"]: row for row in rows}
    sentence_done = [all(is_done(by_item[i]) for i in [it["sentence"]] + [c["item"] for c in it["chunks"]])
                     for it in items]

    if "page" not in st.session_state:  # start at the first sentence not finished
        st.session_state.page = next((i for i, done in enumerate(sentence_done) if not done), 0)

    with st.sidebar:
        st.markdown(f"**{sum(sentence_done)} of {len(items)} sentences done**")
        st.progress(sum(sentence_done) / len(items))
        pairs = [r for r in rows if r["level"] == "pair"]
        st.caption(f"{sum(is_done(r) for r in pairs)} of {len(pairs)} pairs labelled")
        st.selectbox("Jump to", range(len(items)), key="page",
                     format_func=lambda i: f"{'✓' if sentence_done[i] else '·'}  {items[i]['sentence']}  "
                                           f"{items[i]['question_type']}, {len(items[i]['chunks'])} cited")
        st.html(f"<div class='lc-label'>Rules</div>{RULES}")

    item = items[st.session_state.page]
    sentence_row = by_item[item["sentence"]]
    cited = {c["label"] for c in item["chunks"]}

    st.html(f"""<div class="lc-band"><div class="lc-kicker">{item['sentence']} · {item['question_type']} ·
        {len(item['chunks'])} cited</div><div class="lc-question">{html.escape(item['question'])}</div></div>""")

    left, right = st.columns([5, 6], gap="large")
    changed = False
    with left:
        st.html(f"""<div class="lc-box"><div class="lc-label">Sentence to judge</div>
            <div class="lc-sentence">{html.escape(item['text'])}</div></div>""")
        with st.expander("Full answer, as the model wrote it", expanded=True):
            st.html(f"<div class='lc-answer'>{marked(item['answer'], cited)}</div>")

        st.markdown("**All cited chunks together state the claim?**")
        together = st.radio("together", CHOICES, horizontal=True, label_visibility="collapsed",
                            index=CHOICES.index(sentence_row["supports"]) if sentence_row["supports"] in CHOICES else None,
                            key=f"{item['sentence']}-together", format_func=lambda c: {"y": "Yes", "n": "No"}[c])
        notes = st.text_area("Notes", value=sentence_row["notes"], key=f"{item['sentence']}-notes", height=80)
        for field, value in (("supports", together or ""), ("notes", notes)):
            if sentence_row[field] != value:
                sentence_row[field], changed = value, True

    with right:
        for chunk in item["chunks"]:
            row = by_item[chunk["item"]]
            with st.container(border=True):
                st.html(f"""<div class="lc-chunk-head">Chunk [{chunk['label']}]</div>
                    <div class="lc-chunk">{html.escape(chunk['text'])}</div>""")
                a, b = st.columns(2)
                with a:
                    alone = st.radio("States the claim on its own?", CHOICES, horizontal=True,
                                     index=CHOICES.index(row["supports"]) if row["supports"] in CHOICES else None,
                                     key=f"{chunk['item']}-alone", format_func=lambda c: {"y": "Yes", "n": "No"}[c])
                    if row["supports"] != (alone or ""):
                        row["supports"], changed = alone or "", True
                if row["supports_without_this"] != "-":
                    with b:
                        rest = st.radio("Others still state it without this one?", CHOICES, horizontal=True,
                                        index=(CHOICES.index(row["supports_without_this"])
                                               if row["supports_without_this"] in CHOICES else None),
                                        key=f"{chunk['item']}-rest", format_func=lambda c: {"y": "Yes", "n": "No"}[c])
                        if row["supports_without_this"] != (rest or ""):
                            row["supports_without_this"], changed = rest or "", True

    if changed:
        save_rows(rows)
        st.toast("Saved")

    prev, _, nxt = st.columns([1, 4, 1])
    prev.button("← Previous", disabled=st.session_state.page == 0, use_container_width=True,
                on_click=lambda: st.session_state.update(page=st.session_state.page - 1))
    nxt.button("Next →", disabled=st.session_state.page == len(items) - 1, use_container_width=True, type="primary",
               on_click=lambda: st.session_state.update(page=st.session_state.page + 1))


main()
