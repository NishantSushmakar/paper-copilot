"""Tests for the A2 SciREX adapter and metric (evals/adapters/scirex.py, evals/metrics/scirex.py).

Each case is a decision recorded in reports/a02.md §6. The adapter is tested on small
hand-made SciREX records, so no data download is needed.
"""

import pytest

from copilot.cards.schema import PaperCardModel
from evals.adapters.scirex import GoldCluster, SciREXDoc, parse_doc
from evals.metrics.scirex import (
    Counts,
    bootstrap_scores,
    match_field,
    normalize,
    score_paper,
    scores,
    total,
)

# ---------- adapter ----------

RAW = {
    "doc_id": "d1",
    #         0        1    2   3       4     5    6       7         8      9
    "words": ["V-Net", ":", "a", "net", ".", "We", "use", "PROMISE", "2012", "."],
    "sections": [[0, 5], [5, 10]],
    "ner": [[0, 1, "Method"], [7, 9, "Material"], [2, 4, "Method"]],
    "coref": {
        "V-Net": [[0, 1], [2, 4]],
        "PROMISE_2012": [[7, 9]],
        "Dice_Score": [],  # mention-less, but in a relation: kept
        "Mystery": [],  # no relation role, no NER tag: dropped
    },
    "n_ary_relations": [
        {"Material": "PROMISE_2012", "Method": "V-Net", "Metric": "Dice_Score", "Task": "Seg", "score": "0.869"}
    ],
}


def test_text_has_sections_and_round_trips_to_the_words():
    doc = parse_doc(RAW)
    assert doc.text == "V-Net : a net .\n\nWe use PROMISE 2012 ."
    assert doc.text.split() == RAW["words"]


def test_clusters_are_typed_and_mapped_to_card_fields():
    gold = {g.cluster_id: g for g in parse_doc(RAW).gold}
    assert gold["PROMISE_2012"].field == "datasets"  # Material is the card's datasets
    assert gold["V-Net"].field == "methods"
    assert gold["V-Net"].mentions == ["V-Net", "a net"]


def test_mention_less_clusters_are_kept_and_untyped_ones_dropped():
    doc = parse_doc(RAW)
    assert "Dice_Score" in {g.cluster_id for g in doc.gold}
    assert next(g for g in doc.gold if g.cluster_id == "Dice_Score").mentions == []
    assert doc.untyped_clusters == ["Mystery"]


def test_relation_role_wins_over_ner():
    raw = {**RAW, "ner": [[7, 9, "Method"]]}  # a wrong NER tag on the dataset's mention
    gold = {g.cluster_id: g for g in parse_doc(raw).gold}
    assert gold["PROMISE_2012"].field == "datasets"


def test_ner_is_the_fallback_without_a_relation():
    raw = {**RAW, "n_ary_relations": []}
    gold = {g.cluster_id: g for g in parse_doc(raw).gold}
    assert gold["V-Net"].field == "methods"
    assert "Dice_Score" in parse_doc(raw).untyped_clusters  # no mentions, no role: nothing to type it by


# ---------- normalization and matching ----------

@pytest.mark.parametrize(
    ("a", "b"),
    [("V - Net", "V-Net"), ("PROMISE 2012", "PROMISE2012"), ("SQuAD", "squad"), ("F 1", "F1")],
)
def test_normalization_ignores_case_spaces_and_punctuation(a, b):
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize(("a", "b"), [("BLEU-1", "BLEU-4"), ("SQuAD v1.1", "SQuAD v2.0"), ("ResNet-50", "ResNet-18")])
def test_digits_are_kept(a, b):
    assert normalize(a) != normalize(b)


def cluster(cid, *mentions, field="datasets"):
    return GoldCluster(cluster_id=cid, field=field, mentions=list(mentions))


def test_exact_match_after_normalization_not_containment():
    clusters = [cluster("P", "PROMISE 2012 dataset", "” PROMISE2012 ” challenge dataset")]
    assert match_field(["PROMISE 2012"], clusters).tp == 0  # the words are inside a mention, but not equal
    assert match_field(["PROMISE 2012 dataset"], clusters).tp == 1


def test_a_second_prediction_for_one_cluster_is_a_false_positive():
    clusters = [cluster("SQ", "SQuAD", "Stanford Question Answering Dataset")]
    counts = match_field(["SQuAD", "Stanford Question Answering Dataset"], clusters)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 0)


def test_maximum_matching_does_not_depend_on_prediction_order():
    # "segmentation" fits both clusters; "real-time segmentation" only the second
    clusters = [
        cluster("Seg", "segmentation", field="tasks"),
        cluster("RT", "segmentation", "real-time segmentation", field="tasks"),
    ]
    for predictions in (["segmentation", "real-time segmentation"], ["real-time segmentation", "segmentation"]):
        assert match_field(predictions, clusters).tp == 2


def test_mention_less_clusters_are_always_missed_and_lower_the_ceiling():
    clusters = [cluster("A", "CIFAR-10"), cluster("TableOnly")]
    counts = match_field(["CIFAR-10", "TableOnly"], clusters)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    assert (counts.gold, counts.gold_with_mentions) == (2, 1)
    assert scores(counts).recall_ceiling == 0.5


def test_a_prediction_that_normalizes_to_nothing_matches_nothing():
    assert match_field(["—"], [cluster("A", "—")]).tp == 0


# ---------- papers, micro averaging, CIs ----------

DOC = SciREXDoc(
    doc_id="d1",
    text="",
    gold=[cluster("SQ", "SQuAD"), cluster("BERT", "BERT", field="methods"), cluster("F1", "F1", field="metrics")],
)


def test_score_paper_uses_the_scoring_name_only():
    card = PaperCardModel(datasets=[{"names": ["Stanford QA set", "SQuAD"]}])  # SQuAD is not names[0]
    assert score_paper(card, DOC)["datasets"].tp == 0


def test_a_missing_card_is_an_empty_card():
    counts = score_paper(None, DOC)
    assert sum(c.fn for c in counts.values()) == 3
    assert sum(c.tp + c.fp for c in counts.values()) == 0


def test_micro_sums_counts_over_papers():
    per_paper = [Counts(tp=1, fp=1, fn=0, gold=1, gold_with_mentions=1), Counts(tp=0, fp=0, fn=3, gold=3, gold_with_mentions=2)]
    s = scores(total(per_paper))
    assert (s.precision, s.recall) == (0.5, 0.25)  # macro recall would be (1 + 0) / 2 = 0.5
    assert s.f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert s.recall_ceiling == 0.75


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    per_paper = [Counts(tp=t, fp=1, fn=2 - t, gold=2, gold_with_mentions=2) for t in (0, 1, 2, 1, 2, 0, 1)]
    a = bootstrap_scores(per_paper, n_resamples=2000)
    b = bootstrap_scores(per_paper, n_resamples=2000)
    assert a == b
    for interval in a.values():
        assert interval.lower <= interval.value <= interval.upper
