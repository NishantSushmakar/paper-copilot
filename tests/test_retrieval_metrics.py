"""A3 retrieval metrics (evals/metrics/retrieval.py), on cases decided in reports/a03.md §3."""

import pytest

from evals.metrics.retrieval import AnnotatorEvidence, best_annotator, covered, words

GOLD = "We train on the WMT14 English German corpus with four and a half million sentence pairs"  # 16 words
OTHER = "Our encoder is a six layer Transformer with relative position embeddings in every layer"


def evidence(gold_texts, chunks: dict[str, str]) -> AnnotatorEvidence:
    return AnnotatorEvidence(gold_texts, {cid: words(text) for cid, text in chunks.items()})


def tokens(chunks):
    return {cid: 100 for cid in chunks}


def test_a_whole_paragraph_inside_a_chunk_is_fully_found():
    gold = words(GOLD)
    assert covered(gold, words(f"2 Data\n{GOLD}. {OTHER}.")) == set(range(len(gold)))


def test_half_a_paragraph_is_half_found():
    gold = words(GOLD)
    assert len(covered(gold, gold[:8] + words(OTHER))) == 8


def test_stray_short_overlaps_are_not_found():
    # "on the" and "with" also occur in the chunk, but not in a run of 5 gold words.
    assert covered(words(GOLD), words("We rely on the model with care, as in the WMT14 setup")) == set()


def test_a_gold_string_shorter_than_the_run_is_found_whole():
    assert covered(words("4.2 Results"), words("4.2 Results\nThe model reaches 28.4 BLEU.")) == {0, 1, 2}


def test_captions_are_never_found():
    caption = "FLOAT SELECTED: Table 2: BLEU on newstest2014 for every model we train"
    ev = evidence([caption], {"c1": caption})
    scores = ev.score(["c1"], tokens(["c1"]))
    assert scores["coverage@1"] == 0 and scores["recall@1@60"] == 0


def test_hit_thresholds_60_80_90():
    gold = words(GOLD)  # 16 words
    ev = evidence([GOLD], {"c1": " ".join(gold[:13]) + " " + OTHER})  # 13/16 = 81%
    scores = ev.score(["c1"], tokens(["c1"]))
    assert (scores["recall@1@60"], scores["recall@1@80"], scores["recall@1@90"]) == (1, 1, 0)


def test_two_halves_cover_the_paragraph_but_neither_is_a_hit():
    gold = words(GOLD)
    chunks = {"a": " ".join(gold[:8]), "b": " ".join(gold[8:]), "x": OTHER}
    scores = evidence([GOLD], chunks).score(["a", "b", "x"], tokens(chunks))
    assert scores["coverage@1"] == 0.5 and scores["coverage@3"] == 1
    assert scores["recall@3@60"] == 0  # no single chunk holds 60% of it


def test_recall_counts_gold_strings_and_mrr_takes_the_first_hit():
    second = "Errors concentrate on rare named entities and on very long sentences"
    chunks = {"x": OTHER, "a": GOLD, "y": "unrelated words only here", "b": second}
    ev = evidence([GOLD, second], chunks)
    scores = ev.score(["x", "a", "y", "b"], tokens(chunks))
    assert scores["recall@1@80"] == 0 and scores["recall@3@80"] == 0.5 and scores["recall@5@80"] == 1
    assert scores["mrr@80"] == 0.5


def test_mrr_is_zero_without_a_hit_in_the_top_10():
    chunks = {f"x{i}": OTHER for i in range(10)} | {"a": GOLD}
    ranking = [f"x{i}" for i in range(10)] + ["a"]
    assert evidence([GOLD], chunks).score(ranking, tokens(chunks))["mrr@80"] == 0


def test_precision_and_tokens():
    chunks = {"a": GOLD, "x": OTHER}
    scores = evidence([GOLD], chunks).score(["a", "x"], {"a": 30, "x": 20})
    assert scores["precision@1"] == 1
    assert scores["precision@3"] == pytest.approx(16 / (16 + len(words(OTHER))))
    assert scores["tokens@1"] == 30 and scores["tokens@3"] == 50


def test_empty_evidence_strings_are_ignored():
    ev = evidence(["", GOLD, "  "], {"a": GOLD})
    assert len(ev) == 1 and ev.score(["a"], {"a": 1})["recall@1@90"] == 1


def test_best_annotator_takes_each_metrics_maximum():
    assert best_annotator([{"r": 0.5, "m": 1.0}, {"r": 1.0, "m": 0.25}]) == {"r": 1.0, "m": 1.0}
