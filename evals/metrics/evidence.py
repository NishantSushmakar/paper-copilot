"""QASPER evidence F1 (Dasigi et al. 2021) for cited answers (A3, reports/a03.md §10).

The system's evidence is the chunks its answer cites (an abstention cites none). QASPER scores
evidence as paragraphs, so a paragraph of the paper counts as predicted when at least 80% of its
words sit inside one cited chunk (the retrieval eval's 5-word-run matching and threshold).
Predicted and gold paragraphs are then compared as exact strings, as in the official script:
precision = shared / predicted, recall = shared / gold, both empty = 1, and the best annotator
counts. Caption evidence ("FLOAT SELECTED: ...") is never a paragraph, so it can't be matched.
"""

from collections.abc import Sequence

from evals.metrics.retrieval import covered, words

THRESHOLD = 80  # percent of a paragraph's words inside one cited chunk


def predicted_paragraphs(paragraphs: Sequence[str], cited_chunks: Sequence[str],
                         threshold: int = THRESHOLD) -> list[str]:
    chunk_words = [words(chunk) for chunk in cited_chunks]
    predicted = []
    for paragraph in paragraphs:
        paragraph_words = words(paragraph)
        if paragraph_words and any(len(covered(paragraph_words, cw)) * 100 >= threshold * len(paragraph_words)
                                   for cw in chunk_words):
            predicted.append(paragraph)
    return predicted


def paragraph_f1(prediction: Sequence[str], gold: Sequence[str]) -> float:
    if not prediction and not gold:
        return 1.0
    shared = len(set(prediction) & set(gold))
    if shared == 0:
        return 0.0
    precision, recall = shared / len(set(prediction)), shared / len(set(gold))
    return 2 * precision * recall / (precision + recall)


def evidence_f1(prediction: Sequence[str], annotators: Sequence[Sequence[str]]) -> float:
    """Best annotator, as QASPER's evaluator does."""
    return max(paragraph_f1(prediction, gold) for gold in annotators)
