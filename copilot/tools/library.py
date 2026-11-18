"""The research-library tools: `find_papers` and `read_paper` (decisions in reports/a04.md §4).

The split is finding vs reading. An earlier pair, `get_paper_card` (a named paper) and
`search_library` (a topic), overlapped: both took a short text and returned papers, and the
models couldn't tell which to use (SMALL used the search on 17 of 18 named papers).

`find_papers` answers "which papers?". Code decides whether the query names a paper, by
matching its words against titles, typo-tolerant. Dense similarity was measured first and
can't do that job: on synthetic names it put the right paper first only 66-84% of the time,
and famous papers that aren't in the library scored 0.49-0.60 against their nearest title, as
high as real matches (median 0.50-0.53), so no threshold separates "found" from "not found".
The title match put the right paper among the matches for 94-100% of synthetic names, and
absent papers reached at most 0.57. Title matches come first with their full abstract; the
papers closest by topic (dense similarity of title + abstract, the EMBED slot) fill the list.
`name_match` says what the title match found: one, several (2-3), many (more than 3: the model
should ask the user which one) or none.

`read_paper` answers "what does it say?": the A3 retrieval inside one paper (structure-aware
chunks, dense, k = 5, A3's query instruction), passages with their section headings.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path
from typing import Annotated

import numpy as np
from langchain_core.tools import tool
from pydantic import Field

from copilot.cards.backfill import stored_papers
from copilot.cards.chunking import count_tokens
from copilot.ingest.paper import PaperModel
from copilot.retrieval.chunking import Chunk, chunk_id, structure_chunks
from copilot.retrieval.dense import dense_search, embed_chunks, embed_query
from copilot.tools.registry import BadArgument, Registry

ROOT = Path(__file__).resolve().parent.parent.parent
QUEUE_DB = ROOT / "data" / "ingest" / "queue.db"
INDEX_CONFIG = "title-abstract"

NAME_INSTRUCTION = ("Given a short, possibly partial or misspelled reference to a research paper, "
                    "retrieve the paper it refers to")
TOPIC_INSTRUCTION = "Given a topic or question about research, retrieve research papers that are relevant to it"

# From a synthetic calibration on all 296 titles, not from the eval cases (reports/a04.md §4):
# real matches scored 1.0, absent papers at most 0.57. Re-check on the dev cases.
MIN_SCORE = 0.75  # at least 3 of 4 of the name's words found in the title
GAP = 0.1         # papers within this of the top title score count as tied
FUZZY = 0.85      # SequenceMatcher ratio for a misspelled word (5+ letters) to count as found
# Words that say nothing about which paper is meant.
STOPWORDS = frozenset("""a about an and are as at by do does for from how in is it its my of on one
or our paper papers study that the this to via we what which with""".split())
SEVERAL = 3          # up to this many title matches is "several"; more is "many"
SNIPPET_CHARS = 300  # of the abstract, for papers found by topic
READ_K = 5           # passages per read: the A3 setting (dense, k = 5)

ARXIV_ID = r"^\d{4}\.\d{5}(v\d+)?$"  # YYMM.NNNNN, optional version; all 296 library IDs fit


@dataclass(frozen=True)
class Library:
    papers: tuple[PaperModel, ...]  # newest stored version of each paper
    by_id: dict[str, PaperModel]
    vectors: np.ndarray             # one normalized title + abstract vector per paper, same order

    def rank(self, query: str, instruction: str) -> list[tuple[PaperModel, float]]:
        """Every paper with its cosine similarity to `query`, best first (ties: newest ID first)."""
        scores = self.vectors @ embed_query(query, instruction)
        order = sorted(range(len(self.papers)), key=lambda i: (-scores[i], i))
        return [(self.papers[i], float(scores[i])) for i in order]


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def word_found(word: str, title_words: list[str]) -> bool:
    """Exact; most of a title word ("benchmark" -> "benchmarking", but not "lora" -> "lorascan",
    found on the dev case missing_05); a title word that is most of it ("benchmarks" ->
    "benchmark", but not "word2vec" -> "word"); or a typo."""
    for t in title_words:
        if word == t or (len(word) >= 4 and t.startswith(word) and len(word) / len(t) >= 0.55):
            return True
        if len(t) >= 5 and word.startswith(t) and len(t) / len(word) >= 0.75:
            return True
        if len(word) >= 5 and SequenceMatcher(None, word, t).ratio() >= FUZZY:
            return True
    return False


def title_words(title: str) -> list[str]:
    """A title's words, plus each hyphenated word written joined ("Low-Resource" -> "lowresource")."""
    joined = [re.sub(r"[^a-z0-9]", "", w) for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", title.lower())]
    return words(title) + joined


def title_score(name_words: list[str], title: str) -> float:
    """The share of the name's words found in the title."""
    found_in = title_words(title)
    return sum(word_found(w, found_in) for w in name_words) / len(name_words)


def document(paper: PaperModel) -> Chunk:
    text = f"{paper.title}\n\n{paper.abstract}"
    return Chunk(chunk_id(paper, INDEX_CONFIG, 0, text), 0, None, count_tokens(text), text)


@cache
def library() -> Library:
    newest: dict[str, PaperModel] = {}
    for paper in stored_papers(QUEUE_DB):  # newest version first
        newest.setdefault(paper.id, paper)
    papers = tuple(newest.values())
    return Library(papers, dict(newest), embed_chunks([document(paper) for paper in papers]))


@cache
def chunks(paper_id: str) -> tuple[Chunk, ...]:
    return tuple(structure_chunks(library().by_id[paper_id]))


def title_matches(query: str) -> list[PaperModel]:
    """The papers whose title score is within GAP of the top, best first; none if the top is
    below MIN_SCORE. Tied papers are ordered by dense similarity to the query."""
    name_words = words(query)
    if not name_words:
        return []
    lib = library()
    scores = {paper.id: title_score(name_words, paper.title) for paper in lib.papers}
    top = max(scores.values())
    if top < MIN_SCORE:
        return []
    close = {pid for pid, score in scores.items() if top - score < GAP}
    if len(close) == 1:
        return [lib.by_id[close.pop()]]
    return [paper for paper, _ in lib.rank(query, NAME_INSTRUCTION) if paper.id in close]


def entry(paper: PaperModel, match: str) -> dict:
    found = {"paper_id": paper.id, "title": paper.title, "match": match}
    if match == "title":
        return {**found, "authors": paper.authors or [], "abstract": paper.abstract}
    return {**found, "abstract": paper.abstract[:SNIPPET_CHARS]}


def name_note(query: str, name_match: str, n: int) -> str | None:
    if name_match == "several":
        return f"{n} papers have titles matching {query!r}. If the user means one paper, use the one that fits the question."
    if name_match == "many":
        return f"{n} papers have titles matching {query!r}. If the user means one paper, ask them which one."
    if name_match == "none":
        return (f"No paper's title matches {query!r}; these are the closest papers by topic. "
                "If the user named a specific paper, it isn't in the library.")
    return None


@tool
def find_papers(
    query: Annotated[str, Field(description=(
        "A paper's name in the user's words (\"the braille paper\"), a topic, or an arXiv ID."))],
    k: Annotated[int, Field(ge=1, le=10, description="How many papers to return, 1 to 10.")] = 5,
) -> dict:
    """Find papers in the user's research library, for a paper the user names (even loosely or with
    typos) or for a topic. Papers whose title matches the query come first, with their full abstract;
    papers closest to the query's topic follow. `name_match` says whether the title matched one
    paper, several, many, or none. Every paper comes with its paper_id, which read_paper takes. Covers
    only papers already in the library, not new papers on arXiv."""
    lib = library()
    query_id = query.strip().split("v")[0]
    if re.fullmatch(ARXIV_ID, query.strip()) and query_id in lib.by_id:
        named = [lib.by_id[query_id]]
    else:
        named = title_matches(query)
    name_match = {0: "none", 1: "one"}.get(len(named), "several" if len(named) <= SEVERAL else "many")

    papers = [entry(p, "title") for p in named[:k]]
    taken = {p.id for p in named}
    for paper, _ in lib.rank(query, TOPIC_INSTRUCTION):
        if len(papers) >= k:
            break
        if paper.id not in taken:
            papers.append(entry(paper, "topic"))

    result = {"name_match": name_match, "papers": papers}
    if note := name_note(query, name_match, len(named)):
        result["note"] = note
    return result


def passage_text(chunk: Chunk) -> str:
    """A chunk's text without the section heading it starts with."""
    text = chunk.text
    if chunk.heading and text.startswith(chunk.heading):
        text = text[len(chunk.heading):]
    return text.strip()


@tool
def read_paper(
    paper_id: Annotated[str, Field(pattern=ARXIV_ID, description=(
        "The paper's arXiv ID from a find_papers result, such as 2410.10347."))],
    question: Annotated[str, Field(description="What you want to know from the paper.")],
) -> dict:
    """Read inside one paper of the library: returns the 5 passages of its full text most relevant
    to the question, each with its section heading. Use it for details an abstract doesn't give
    (methods, datasets, models, numbers). Get the paper_id from find_papers first."""
    paper = library().by_id.get(paper_id.split("v")[0])
    if paper is None:
        raise BadArgument(f"No paper with ID {paper_id} is in the library.",
                          "Find the paper with find_papers and use the paper_id it returns.")
    by_id = {c.chunk_id: c for c in chunks(paper.id)}
    hits = dense_search(question, list(by_id.values()), k=READ_K)
    return {"paper_id": paper.id, "title": paper.title,
            "passages": [{"n": h.rank, "section": by_id[h.chunk_id].heading or "",
                          "text": passage_text(by_id[h.chunk_id])} for h in hits]}


# My call: generous next to the measured cost once everything is embedded (2026-11-16: a dense
# ranking 34 ms, a title match 45 ms). Building the title + abstract index took 51 s the first
# time (cached after that), so build_registry() does it before any call's time limit, and
# scripts/embed_library.py embeds every paper's chunks ahead of any read.
TIMEOUTS = {"find_papers": 10.0, "read_paper": 10.0}


def build_registry() -> Registry:
    library()  # load the embedding model and build the index outside any call's time limit
    registry = Registry()
    registry.register(find_papers, side_effect=False, timeout=TIMEOUTS["find_papers"])
    registry.register(read_paper, side_effect=False, timeout=TIMEOUTS["read_paper"])
    return registry
