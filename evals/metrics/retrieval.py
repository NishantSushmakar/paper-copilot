"""Retrieval metrics for A3 (reports/a03.md §3): how much of the gold evidence a ranking found.

Words are lowercased runs of letters and digits. A gold word counts as found in a chunk only
inside a run of at least MIN_RUN consecutive gold words that the chunk also has, in order (or
the whole gold string, if it is shorter). So a whole paragraph in a chunk is fully found, the
half of it that a window cut off is half found, and a stray "the results of" elsewhere in the
paper isn't found at all.

For one annotator's evidence and one ranking, at each k:
* coverage@k   share of all the annotator's gold words found in the top k chunks together
* precision@k  those found gold words over all words in the top k chunks
* tokens@k     tokens in the top k chunks (what the reader would be given)
* recall@k@t   share of the annotator's gold strings that are a hit in the top k, where a hit
               is one chunk holding at least t% of that string's words (t = 60, 80, 90)
* mrr@t        1 / rank of the first chunk in the top 10 that is a hit for any gold string

Figure and table captions ("FLOAT SELECTED: ...") are never found: they aren't in the text
the chunkers see, so they count as misses (a recall ceiling of about 90% on dev).
A question scores as its best annotator, metric by metric, as A0's answer F1 does.
"""

import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher

WORD = re.compile(r"[^\W_]+")
CAPTION_PREFIX = "FLOAT SELECTED"
MIN_RUN = 5  # words
THRESHOLDS = (60, 80, 90)  # percent of a gold string's words
KS = (1, 3, 5, 10)
MRR_DEPTH = 10


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def covered(gold: Sequence[str], chunk: Sequence[str], min_run: int = MIN_RUN) -> set[int]:
    """Positions in `gold` that lie in a run of at least min_run words the chunk also has, in order."""
    if not gold:
        return set()
    need = min(min_run, len(gold))
    matcher = SequenceMatcher(None, gold, chunk, autojunk=False)
    return {i for block in matcher.get_matching_blocks() if block.size >= need
            for i in range(block.a, block.a + block.size)}


class AnnotatorEvidence:
    """One annotator's gold evidence, matched once against every chunk of the paper."""

    def __init__(self, evidence: Sequence[str], chunk_words: Mapping[str, Sequence[str]]):
        golds = [(text, words(text)) for text in evidence]
        golds = [(text, gold) for text, gold in golds if gold]  # nothing to find in "" or "-"
        self.sizes = [len(gold) for _, gold in golds]
        self.chunk_words = chunk_words
        self.found = {
            chunk_id: [set() if text.startswith(CAPTION_PREFIX) else covered(gold, chunk)
                       for text, gold in golds]
            for chunk_id, chunk in chunk_words.items()
        }

    def __len__(self) -> int:
        return len(self.sizes)

    def is_hit(self, chunk_id: str, gold: int, threshold: int) -> bool:
        return len(self.found[chunk_id][gold]) * 100 >= threshold * self.sizes[gold]

    def score(self, ranking: Sequence[str], tokens: Mapping[str, int]) -> dict[str, float]:
        """Every metric for one ranking (chunk IDs, best first)."""
        golds = range(len(self))
        metrics = {}
        for k in KS:
            top = ranking[:k]
            found = sum(len(set().union(*(self.found[c][g] for c in top))) for g in golds)
            retrieved = sum(len(self.chunk_words[c]) for c in top)
            metrics[f"coverage@{k}"] = found / sum(self.sizes)
            metrics[f"precision@{k}"] = found / retrieved if retrieved else 0.0
            metrics[f"tokens@{k}"] = float(sum(tokens[c] for c in top))
            for t in THRESHOLDS:
                hits = sum(any(self.is_hit(c, g, t) for c in top) for g in golds)
                metrics[f"recall@{k}@{t}"] = hits / len(self)
        for t in THRESHOLDS:
            first = next((rank for rank, c in enumerate(ranking[:MRR_DEPTH], 1)
                          if any(self.is_hit(c, g, t) for g in golds)), None)
            metrics[f"mrr@{t}"] = 1 / first if first else 0.0
        return metrics


def best_annotator(per_annotator: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The question's score: each metric's maximum over its annotators."""
    return {metric: max(scores[metric] for scores in per_annotator) for metric in per_annotator[0]}
