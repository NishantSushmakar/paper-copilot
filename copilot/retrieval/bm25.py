"""BM25 over one paper's chunks (rank_bm25's BM25Okapi, default k1 = 1.5, b = 0.75).

Words are lowercased runs of letters and digits ("BERT-base's F1" -> bert, base, s, f1), with
no stemming and no stopword list. The index is built per call: a paper has about 30 chunks.

With so few chunks, a word in more than half of them (often the paper's own topic words) gets
a negative IDF, which BM25Okapi replaces with a small floor (epsilon x the mean IDF).
"""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from copilot.retrieval.chunking import Chunk
from copilot.retrieval.hit import Hit, ranked

WORD = re.compile(r"[^\W_]+")  # letters and digits, in any script


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def bm25_search(question: str, chunks: Sequence[Chunk], k: int = 10) -> list[Hit]:
    if not chunks:
        return []
    scorer = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
    return ranked(chunks, scorer.get_scores(tokenize(question)), k)
