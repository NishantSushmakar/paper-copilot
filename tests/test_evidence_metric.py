"""QASPER evidence F1 for cited answers (evals/metrics/evidence.py), as decided in reports/a03.md §10."""

from evals.metrics.evidence import evidence_f1, paragraph_f1, predicted_paragraphs

P1 = "We train the model on the WMT14 English German corpus with four million sentence pairs."
P2 = "The results show a gain of two BLEU points over the strongest baseline on newstest2014."
P3 = "Future work will look at other language pairs and at document level translation."


def test_a_paragraph_inside_a_cited_chunk_is_predicted():
    chunk = "4 Experiments\n" + P1 + "\n" + P2
    assert predicted_paragraphs([P1, P2, P3], [chunk]) == [P1, P2]


def test_a_paragraph_mostly_outside_every_chunk_is_not_predicted():
    clipped = " ".join(P1.split()[:8])  # 8 of 15 words: under 80%
    assert predicted_paragraphs([P1], [clipped]) == []


def test_no_cited_chunks_predicts_nothing():
    assert predicted_paragraphs([P1, P2], []) == []


def test_paragraph_f1_matches_the_official_rules():
    assert paragraph_f1([], []) == 1.0
    assert paragraph_f1([P1], []) == 0.0 and paragraph_f1([], [P1]) == 0.0
    assert paragraph_f1([P1, P2], [P1]) == 2 * 0.5 * 1.0 / 1.5


def test_the_best_annotator_counts():
    assert evidence_f1([P1], [[P2], [P1]]) == 1.0


def test_an_abstention_on_a_question_without_evidence_scores_1():
    assert evidence_f1([], [[], [P1]]) == 1.0
