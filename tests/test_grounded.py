"""Grounded answers (copilot/answer/grounded.py): the prompt, citation parsing, label -> chunk ID."""

from datetime import timedelta

import pytest

from copilot.answer import grounded
from copilot.llm.llm_client import LLMResponse
from copilot.retrieval.chunking import Chunk
from copilot.retrieval.hit import Hit

IDS = ["c-a", "c-b", "c-c", "c-d", "c-e"]  # labels [1] to [5]


def test_one_citation_per_claim():
    parsed = grounded.parse("News Articles [2], Twitter [4]", IDS)
    assert parsed.answer == "News Articles, Twitter"
    assert [s.chunk_ids for s in parsed.sentences] == [["c-b", "c-d"]]
    assert parsed.uncited == 0 and not parsed.abstained


@pytest.mark.parametrize("raw, answer, ids", [
    ("yes [1]", "yes", ["c-a"]),
    ("Answer: yes [1]", "yes", ["c-a"]),
    ("BLEU [2][4]", "BLEU", ["c-b", "c-d"]),
    ("BLEU [2, 4]", "BLEU", ["c-b", "c-d"]),
    ("BLEU [2][2]", "BLEU", ["c-b"]),
    ("They use BLEU [3].", "They use BLEU.", ["c-c"]),
])
def test_marker_forms(raw, answer, ids):
    parsed = grounded.parse(raw, IDS)
    assert parsed.answer == answer
    assert parsed.sentences[0].chunk_ids == ids


def test_a_marker_after_the_full_stop_belongs_to_that_sentence():
    parsed = grounded.parse("They use BLEU. [3]", IDS)
    assert len(parsed.sentences) == 1 and parsed.sentences[0].chunk_ids == ["c-c"]


def test_a_sentence_without_a_citation_is_flagged():
    parsed = grounded.parse("They use BLEU [3]. It is computed on newstest2014.", IDS)
    assert [s.chunk_ids for s in parsed.sentences] == [["c-c"], []]
    assert parsed.uncited == 1


def test_a_label_outside_1_to_k_is_invalid_and_does_not_count_as_a_citation():
    parsed = grounded.parse("BLEU [7]", IDS)
    assert parsed.invalid_labels == [7] and parsed.sentences[0].labels == [7]
    assert parsed.sentences[0].chunk_ids == [] and parsed.uncited == 1


@pytest.mark.parametrize("raw", ["Unanswerable", "unanswerable.", '"Unanswerable"', "Answer: Unanswerable"])
def test_unanswerable_is_an_abstention_not_an_uncited_answer(raw):
    parsed = grounded.parse(raw, IDS)
    assert parsed.abstained and parsed.uncited == 0


def test_the_user_prompt_labels_chunks_from_1_in_order():
    chunks = [Chunk("x", 0, "2 Data", 3, "2 Data\nWMT14."), Chunk("y", 1, "4 Results", 3, "4 Results\n28.4 BLEU.")]
    assert grounded.user_prompt("Which corpus?", chunks) == (
        "[Evidence]\n[1] 2 Data\nWMT14.\n[2] 4 Results\n28.4 BLEU.\n[End of evidence]\nQuestion: Which corpus?")


def test_the_system_prompt_is_the_v1_file():
    prompt = grounded.system_prompt()
    assert prompt.startswith("You are a researcher in NLP field.")
    assert "Answer: News Articles [2], Twitter [4]" in prompt and 'Just say "Unanswerable"' in prompt


def test_answer_question_puts_the_retrieved_chunks_in_rank_order_and_maps_labels_back(monkeypatch):
    chunks = [Chunk(f"id{i}", i, None, 1, f"text {i}") for i in range(4)]
    seen = {}

    def fake_retrieve(question, chunks_, k):
        return [Hit("id3", 0.9, 1), Hit("id1", 0.5, 2)][:k]

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return LLMResponse(text="It is text 3 [1].", thinking=None, prompt_tokens=1, completion_tokens=1,
                           finish_reason="stop", model="m", latency=timedelta(seconds=1), cache_hit=False)

    monkeypatch.setattr(grounded, "chat_with_llm", fake_chat)
    parsed, _ = grounded.answer_question("q?", chunks, fake_retrieve, 2, slot="MEDIUM", temperature=0.6,
                                         max_tokens=4000, thinking=True)
    assert parsed.retrieved == ["id3", "id1"]
    assert "[1] text 3\n[2] text 1" in seen["messages"][1]["content"]
    assert parsed.sentences[0].chunk_ids == ["id3"] and parsed.answer == "It is text 3."


@pytest.mark.parametrize("raw, texts", [
    ("They combine the scores via Eq. 19, and optional constraint detection [2].",
     ["They combine the scores via Eq. 19, and optional constraint detection."]),
    ("They use deep models, e.g. BERT [1].", ["They use deep models, e.g. BERT."]),
    ("The patterns include: 1. Rhetorical questions [1] 2. Hyperbole [2]",
     ["The patterns include: 1. Rhetorical questions 2. Hyperbole"]),
    ("Patterns:\n1. Rhetorical questions [1]\n2. Hyperbole [2]", ["Patterns:", "1. Rhetorical questions", "2. Hyperbole"]),
    ("It was released in 2019. They use it [1].", ["It was released in 2019.", "They use it."]),
])
def test_sentence_splits(raw, texts):
    assert [s.text for s in grounded.parse(raw, IDS).sentences] == texts


@pytest.mark.parametrize("raw", ["The corpus covers typography. [2], [3], [4]", "They probe POS tagging, etc. [3]."])
def test_markers_after_the_full_stop_with_punctuation_belong_to_that_sentence(raw):
    parsed = grounded.parse(raw, IDS)
    assert len(parsed.sentences) == 1 and parsed.sentences[0].chunk_ids and parsed.uncited == 0


def test_a_bare_number_answer_is_a_sentence():
    parsed = grounded.parse("327 [1]", IDS)
    assert [s.text for s in parsed.sentences] == ["327"] and parsed.uncited == 0


def test_shuffled_keeps_the_same_hits_in_a_seeded_order():
    from evals.systems.qasper import shuffled
    hits = [Hit(f"c{i}", 1.0 - i / 10, i + 1) for i in range(5)]
    first = shuffled(lambda q, c, k: list(hits), "qid-1")("q", [], 5)
    again = shuffled(lambda q, c, k: list(hits), "qid-1")("q", [], 5)
    other = shuffled(lambda q, c, k: list(hits), "qid-2")("q", [], 5)
    assert sorted(h.chunk_id for h in first) == [h.chunk_id for h in hits]
    assert first == again and first != hits and other != first
