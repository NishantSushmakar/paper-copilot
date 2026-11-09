"""Citation recall and precision (evals/metrics/citation.py): the ALCE rules decided in reports/a03.md §9."""

import pytest

from evals.metrics import citation
from evals.metrics.citation import Verdict, agreement, answer_scores, cohen_kappa, judge_sentence


def fake_judge(supports: set[frozenset[str]]):
    """Entails when the premise's chunks (split on newlines) include one of the supporting sets."""
    def judge(premise: str, hypothesis: str) -> Verdict:
        chunks = frozenset(premise.split("\n"))
        entailed = any(s <= chunks for s in supports)
        return Verdict(entailed=entailed, p_entailment=0.9 if entailed else 0.1, truncated=False)
    return judge


def test_one_chunk_that_states_the_claim_is_supported_and_relevant():
    v = judge_sentence(fake_judge({frozenset({"a"})}), "claim", ["a"])
    assert v.supported and v.relevant == [True] and v.without == [None]


def test_partial_chunks_are_rescued_by_the_removal_rule():
    # "SQuAD and TriviaQA [2][5]": neither alone, both together
    v = judge_sentence(fake_judge({frozenset({"squad", "trivia"})}), "claim", ["squad", "trivia"])
    assert v.supported and [a.entailed for a in v.alone] == [False, False]
    assert v.relevant == [True, True]


def test_an_unnecessary_citation_is_irrelevant():
    v = judge_sentence(fake_judge({frozenset({"a"})}), "claim", ["a", "noise"])
    assert v.supported and v.relevant == [True, False]


def test_an_unsupported_sentence_earns_no_precision():
    v = judge_sentence(fake_judge(set()), "claim", ["a", "b"])
    assert not v.supported and v.relevant == [False, False]


def test_a_sentence_without_citations_is_unsupported_and_has_no_citations():
    v = judge_sentence(fake_judge({frozenset({"a"})}), "claim", [])
    assert not v.supported and v.together is None and v.relevant == []


def test_answer_scores_count_uncited_sentences_against_recall():
    judge = fake_judge({frozenset({"a"})})
    verdicts = [judge_sentence(judge, "x", ["a", "noise"]), judge_sentence(judge, "y", [])]
    scores = answer_scores(verdicts)
    assert scores["recall"] == 0.5 and scores["precision"] == 0.5 and scores["citations"] == 2


def test_an_answer_with_no_citations_has_precision_0():
    assert answer_scores([judge_sentence(fake_judge(set()), "x", [])])["precision"] == 0.0


def test_hypothesis_a_is_the_question_then_the_sentence():
    assert citation.hypothesis_a(" What metric? ", "BLEU ") == "What metric? BLEU"


def test_hypothesis_b_returns_the_first_line_without_a_claim_prefix(monkeypatch):
    class Reply:
        text = "Claim: They use BLEU for evaluation.\n"
    seen = {}
    monkeypatch.setattr(citation, "chat_with_llm", lambda **kw: seen.update(kw) or Reply())
    assert citation.hypothesis_b("What metric?", "BLEU") == "They use BLEU for evaluation."
    assert seen["slot"] == "MEDIUM" and seen["thinking"] is False
    assert seen["messages"][1]["content"] == "Question: What metric?\nAnswer sentence: BLEU"


@pytest.mark.parametrize("gold, pred, kappa", [
    ([True, True, False, False], [True, True, False, False], 1.0),
    ([True, False, True, False], [True, True, False, False], 0.0),
    ([True, True, True, True], [True, True, True, True], None),  # one class only: chance agreement is 1
])
def test_cohen_kappa(gold, pred, kappa):
    assert cohen_kappa(gold, pred) == kappa


def test_agreement_counts():
    a = agreement([True, True, False, False], [True, False, True, False])
    assert (a["accuracy"], a["tp"], a["fn"], a["fp"], a["tn"]) == (0.5, 1, 1, 1, 1)


def test_the_real_judge_uses_the_top_label_and_flags_truncation():
    judge = citation.NLIJudge()
    v = judge("We trained the model on ImageNet for 90 epochs.", "The model was trained on ImageNet.")
    assert v.entailed and v.p_entailment > 0.5 and not v.truncated
    long = judge("word " * 600, "The model was trained on ImageNet.")
    assert long.truncated
