"""The library tools (copilot/tools/library.py) on a fake library, run through the registry.
No embedding model: the fake ranks papers in a fixed order, and read_paper's search is faked."""

import json
from datetime import datetime, timezone

import pytest

from copilot.ingest.paper import PaperModel
from copilot.retrieval.chunking import Chunk
from copilot.retrieval.hit import Hit
from copilot.tools import library as lib_module
from copilot.tools.library import find_papers, read_paper, title_score, word_found, words
from copilot.tools.registry import Registry

TITLES = {
    "2610.36195": "Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs",
    "2509.25002": "Circuit Distillation",
    "2610.26433": "The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures",
    "2610.37624": "Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions in Agent Teams",
    "2610.31303": "The Value of Memory in Language Agents",
    "2610.36918": "TrawMem: Reframing Agent Memory as Task-Oriented Workspaces",
    "2610.37443": "ETAMem: Agent Memory for Long-Term Conversational Agents",
    "2610.35544": "StateLoop: Persistent State for Looped Transformers",
    "2610.35556": "LoopFold: Parallelizing Recurrent Depth in Looped Transformers",
}


def paper(pid, title):
    return PaperModel(source="arxiv", id=pid, version=2, title=title, authors=["A. Author"],
                      abstract=f"Abstract of {title}. " + "x" * 400, sections=[],
                      fetched_at=datetime(2026, 11, 1, tzinfo=timezone.utc))


class FakeLibrary:
    def __init__(self):
        self.papers = tuple(paper(pid, t) for pid, t in TITLES.items())
        self.by_id = {p.id: p for p in self.papers}

    def rank(self, query, instruction):
        return [(p, 1.0 - i / 100) for i, p in enumerate(self.papers)]


CHUNKS = (Chunk("c0", 0, "Abstract", 10, "Abstract\nThe abstract."),
          Chunk("c1", 1, "4 Experiments", 10, "4 Experiments\nWe test GPT-5 and Gemini."))


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(lib_module, "library", lambda: FakeLibrary())
    monkeypatch.setattr(lib_module, "chunks", lambda paper_id: CHUNKS)
    monkeypatch.setattr(lib_module, "dense_search",
                        lambda question, chunks, k: [Hit("c1", 0.9, 1), Hit("c0", 0.5, 2)][:k])


@pytest.fixture
def registry():
    r = Registry()
    r.register(find_papers, side_effect=False, timeout=5)
    r.register(read_paper, side_effect=False, timeout=5)
    return r


def call(registry, name, **args):
    result = registry.execute({"name": name, "args": args, "id": "c1"})
    return result, json.loads(result.content)


# ---------- word matching ----------

@pytest.mark.parametrize("word, title_word", [("braille", "braille"), ("benchmark", "benchmarking"),
                                              ("distilation", "distillation"), ("benchmarks", "benchmark")])
def test_words_that_count_as_found(word, title_word):
    assert word_found(word, [title_word])


@pytest.mark.parametrize("word, title_word", [("word2vec", "word"), ("dropout", "drop"), ("llama", "lama2x"),
                                              ("lora", "lorascan")])
def test_words_that_do_not(word, title_word):
    assert not word_found(word, [title_word])


def test_a_hyphenated_title_word_typed_joined_still_matches():
    assert title_score(words("lowresource text generation"), "Text Generation in Low-Resource Languages") == 1.0


# ---------- find_papers ----------

def test_a_loose_name_matches_one_title_first_then_topic_papers_fill_the_list(registry):
    result, out = call(registry, "find_papers", query="the braille paper")
    assert result.ok and out["name_match"] == "one" and "note" not in out
    assert out["papers"][0] == {"paper_id": "2610.36195", "title": TITLES["2610.36195"], "match": "title",
                                "authors": ["A. Author"], "abstract": out["papers"][0]["abstract"]}
    assert len(out["papers"]) == 5 and all(p["match"] == "topic" for p in out["papers"][1:])


def test_a_title_match_gets_the_full_abstract_and_topic_papers_a_snippet(registry):
    _, out = call(registry, "find_papers", query="braille")
    assert len(out["papers"][0]["abstract"]) > lib_module.SNIPPET_CHARS
    assert all(len(p["abstract"]) <= lib_module.SNIPPET_CHARS for p in out["papers"][1:])


def test_a_typo_still_matches(registry):
    _, out = call(registry, "find_papers", query="circuit distilation")
    assert out["name_match"] == "one" and out["papers"][0]["paper_id"] == "2509.25002"


def test_two_tied_titles_are_several(registry):
    _, out = call(registry, "find_papers", query="the looped transformer paper")
    assert out["name_match"] == "several" and "fits the question" in out["note"]
    assert {p["paper_id"] for p in out["papers"] if p["match"] == "title"} == {"2610.35544", "2610.35556"}


def test_more_than_three_tied_titles_are_many_and_say_ask_the_user(registry):
    _, out = call(registry, "find_papers", query="agent memory")
    assert out["name_match"] == "many" and "ask them" in out["note"]
    assert all(p["match"] == "title" for p in out["papers"])  # 5 title matches fill k = 5


def test_no_title_match_returns_topic_papers_and_says_so(registry):
    _, out = call(registry, "find_papers", query="SecAlign")
    assert out["name_match"] == "none" and "isn't in the library" in out["note"]
    assert len(out["papers"]) == 5 and all(p["match"] == "topic" for p in out["papers"])


def test_an_arxiv_id_as_the_query_matches_that_paper(registry):
    _, out = call(registry, "find_papers", query="2509.25002v1")
    assert out["name_match"] == "one" and out["papers"][0]["paper_id"] == "2509.25002"


def test_k_sets_the_list_length_and_is_capped(registry):
    assert len(call(registry, "find_papers", query="braille", k=2)[1]["papers"]) == 2
    assert call(registry, "find_papers", query="braille", k=11)[1]["error"] == "bad_argument"


# ---------- read_paper ----------

def test_read_returns_passages_with_sections_and_no_heading_in_the_text(registry):
    result, out = call(registry, "read_paper", paper_id="2610.36195", question="which vlms")
    assert result.ok and out["paper_id"] == "2610.36195" and out["title"] == TITLES["2610.36195"]
    assert out["passages"][0] == {"n": 1, "section": "4 Experiments", "text": "We test GPT-5 and Gemini."}


def test_read_accepts_a_version_suffix(registry):
    assert call(registry, "read_paper", paper_id="2610.36195v2", question="q")[1]["paper_id"] == "2610.36195"


def test_read_of_an_unknown_id_points_to_find_papers(registry):
    result, out = call(registry, "read_paper", paper_id="2601.99999", question="q")
    assert out["error"] == "bad_argument" and "find_papers" in out["what_to_do"]
    assert result.metadata["layer"] == "tool"


def test_read_with_a_title_instead_of_an_id_fails_the_schema(registry):
    result, out = call(registry, "read_paper", paper_id="the braille paper", question="q")
    assert out["error"] == "bad_argument" and result.metadata["layer"] == "registry"


def test_read_needs_a_question(registry):
    assert call(registry, "read_paper", paper_id="2610.36195")[1]["error"] == "bad_argument"
