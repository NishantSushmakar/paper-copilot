"""Tests for the A2 merge (copilot/cards/merge.py).

Each case is a decision recorded in reports/a02.md §4 ("Merge"): entities linked by a
shared name (or, for the dev ablation, by name pairs at least two chunks agree on),
the scoring name first, results matched through merged entities and parsed value,
code_url by majority, and only valid chunks merged, with coverage recorded.
"""

import logging

import pytest

from copilot.cards.extract import ChunkOutcome
from copilot.cards.merge import merge_paper
from copilot.cards.schema import PaperCardModel


def outcome(chunk_index, card=None, status="valid", fixes=(), tokens=(100, 50), latency=10.0, attempts=1):
    return ChunkOutcome(
        paper_id="p1",
        chunk_index=chunk_index,
        strategy="S1",
        slot="MEDIUM",
        prompt_version="v3",
        status=status,
        card=PaperCardModel.model_validate(card) if card is not None else None,
        failed_fields=["results"] if status == "content" else [],
        silent_fixes=list(fixes),
        raw_text="{}",
        attempts=attempts,
        finish_reason="stop",
        prompt_tokens=tokens[0],
        completion_tokens=tokens[1],
        latency_s=latency,
        cache_hit=False,
    )


def card(**fields):
    """A chunk card; entity fields take lists of name lists."""
    out = {key: [{"names": names} for names in fields.get(key, [])] for key in ("tasks", "datasets", "methods", "metrics")}
    out["results"] = fields.get("results", [])
    out["code_url"] = fields.get("code_url")
    return out


def result(dataset, metric, method, raw_value):
    return {"dataset": dataset, "metric": metric, "method": method, "raw_value": raw_value}


def names(merged, field):
    return [entity.names for entity in getattr(merged.card, field)]


# ---------- entities ----------

def test_shared_names_link_entities_across_chunks():
    merged = merge_paper([
        outcome(0, card(datasets=[["Stanford Question Answering Dataset", "SQuAD"]])),
        outcome(1, card(datasets=[["SQuAD"]])),
        outcome(2, card(datasets=[["SQuAD", "SQuAD v1.1"]])),
    ])
    assert names(merged, "datasets") == [["SQuAD", "Stanford Question Answering Dataset", "SQuAD v1.1"]]


def test_scoring_name_is_the_most_frequent_then_the_longest():
    merged = merge_paper([
        outcome(0, card(methods=[["FRRN", "Full-Resolution Residual Networks"]])),
        outcome(1, card(methods=[["FRRN"]])),
    ])
    assert names(merged, "methods")[0][0] == "FRRN"  # in 2 chunks, against 1

    tie = merge_paper([outcome(0, card(methods=[["FRRN", "Full-Resolution Residual Networks"]]))])
    assert names(tie, "methods")[0][0] == "Full-Resolution Residual Networks"  # both in 1 chunk: longest


def test_case_and_spacing_are_ignored_and_the_papers_spelling_kept():
    merged = merge_paper([
        outcome(0, card(datasets=[["SQuAD"]])),
        outcome(1, card(datasets=[["squad"]])),
        outcome(2, card(datasets=[["SQuAD"]])),
        outcome(3, card(datasets=[["PROMISE  2012"]])),
        outcome(4, card(datasets=[["PROMISE 2012"]])),
    ])
    assert names(merged, "datasets") == [["SQuAD"], ["PROMISE  2012"]]


def test_the_same_entity_twice_in_one_card_is_merged():
    merged = merge_paper([outcome(0, card(datasets=[["SQuAD"], ["squad"]]))])
    assert names(merged, "datasets") == [["SQuAD"]]


def test_entity_types_are_merged_separately():
    merged = merge_paper([outcome(0, card(tasks=[["BLEU"]], metrics=[["BLEU"]]))])
    assert names(merged, "tasks") == [["BLEU"]]
    assert names(merged, "metrics") == [["BLEU"]]


SLOPPY = [
    outcome(0, card(methods=[["FRRN A", "FRRN B", "ResNet"]])),  # glues a baseline in
    outcome(1, card(methods=[["FRRN A", "FRRN B"]])),
    outcome(2, card(methods=[["ResNet"]])),
]


def test_share_name_lets_one_sloppy_card_glue_entities():
    merged = merge_paper(SLOPPY, merge_rule="share_name")
    assert len(names(merged, "methods")) == 1


def test_agree2_needs_two_chunks_to_join_different_names():
    merged = merge_paper(SLOPPY, merge_rule="agree2")
    methods = names(merged, "methods")
    assert sorted(map(sorted, methods)) == [["FRRN A", "FRRN B"], ["ResNet"]]  # A+B agreed by chunks 0 and 1


def test_agree2_trusts_the_card_of_a_single_chunk_paper():
    merged = merge_paper([outcome(0, card(datasets=[["Stanford Question Answering Dataset", "SQuAD"]]))], merge_rule="agree2")
    assert len(names(merged, "datasets")) == 1


# ---------- results ----------

def test_results_match_through_merged_entities_and_parsed_value():
    merged = merge_paper([
        outcome(0, card(datasets=[["Cityscapes"]], metrics=[["mean IoU", "IoU"]], methods=[["FRRN A"]],
                        results=[result("Cityscapes", "mean IoU", "FRRN A", "65.7%")])),
        outcome(1, card(datasets=[["cityscapes"]], metrics=[["IoU"]], methods=[["FRRN A"]],
                        results=[result("cityscapes", "IoU", "FRRN A", "65.7 %")])),  # overlap repeat
        outcome(2, card(datasets=[["Cityscapes"]], metrics=[["mean IoU"]], methods=[["FRRN A"]],
                        results=[result("Cityscapes", "mean IoU", "FRRN A", "71.8%")])),  # another value
    ])
    rows = [(r.dataset, r.metric, r.method, r.raw_value) for r in merged.card.results]
    assert rows == [
        ("Cityscapes", "mean IoU", "FRRN A", "65.7%"),  # rewritten to scoring names, first raw text kept
        ("Cityscapes", "mean IoU", "FRRN A", "71.8%"),
    ]


def test_no_value_markers_collapse():
    merged = merge_paper([
        outcome(i, card(datasets=[["X"]], metrics=[["F1"]], methods=[["M"]], results=[result("X", "F1", "M", marker)]))
        for i, marker in enumerate(["—", "N/A"])
    ])
    assert len(merged.card.results) == 1


def test_results_agree_with_the_entity_lists():
    merged = merge_paper([
        outcome(0, card(methods=[["Residual Network Baseline", "ResNet"]], datasets=[["X"]], metrics=[["F1"]],
                        results=[result("X", "F1", "ResNet", "62.8")])),
        outcome(1, card(methods=[["Residual Network Baseline"]], datasets=[["X"]], metrics=[["F1"]])),
    ])
    assert merged.card.results[0].method == names(merged, "methods")[0][0] == "Residual Network Baseline"


# ---------- code_url ----------

def test_code_url_most_frequent_non_null():
    merged = merge_paper([
        outcome(0, card(code_url="https://github.com/a/real")),
        outcome(1, card(code_url=None)),
        outcome(2, card(code_url="https://github.com/org/repo")),
        outcome(3, card(code_url="https://github.com/a/real")),
    ])
    assert str(merged.card.code_url) == "https://github.com/a/real"


def test_code_url_tie_goes_to_the_earliest_chunk_and_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="copilot.cards.merge"):
        merged = merge_paper([
            outcome(0, card(code_url="https://github.com/first/one")),
            outcome(1, card(code_url="https://github.com/second/one")),
        ])
    assert str(merged.card.code_url) == "https://github.com/first/one"
    assert "disagreement" in caplog.text


def test_code_url_null_when_no_chunk_gives_one():
    assert merge_paper([outcome(0, card()), outcome(1, card())]).card.code_url is None


# ---------- coverage, failures, totals ----------

def test_only_valid_chunks_are_merged_and_coverage_recorded():
    merged = merge_paper([
        outcome(0, card(datasets=[["SQuAD"]])),
        outcome(1, status="cut_off"),
        outcome(2, status="content"),
        outcome(3, card(datasets=[["CoNLL-2003"]])),
    ])
    assert (merged.chunks_total, merged.chunks_valid) == (4, 2)
    assert [(f.chunk_index, f.status) for f in merged.chunk_failures] == [(1, "cut_off"), (2, "content")]
    assert names(merged, "datasets") == [["SQuAD"], ["CoNLL-2003"]]


def test_no_valid_chunk_gives_an_empty_card():
    merged = merge_paper([outcome(0, status="format"), outcome(1, status="cut_off")])
    assert merged.chunks_valid == 0
    assert merged.card == PaperCardModel()


def test_totals_and_silent_fixes_over_all_chunks():
    merged = merge_paper([
        outcome(0, card(), fixes=["Silent fix: a"], tokens=(4000, 1000), latency=50.0),
        outcome(1, status="format", fixes=["Silent fix: b"], tokens=(4100, 900), latency=60.0, attempts=3),
    ])
    assert (merged.prompt_tokens, merged.completion_tokens, merged.latency_s, merged.attempts) == (8100, 1900, 110.0, 4)
    assert merged.silent_fixes == ["chunk 0: Silent fix: a", "chunk 1: Silent fix: b"]


def test_outcomes_from_two_papers_are_refused():
    other = outcome(1, card()).model_copy(update={"paper_id": "p2"})
    with pytest.raises(ValueError):
        merge_paper([outcome(0, card()), other])
