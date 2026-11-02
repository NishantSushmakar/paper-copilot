import random
from typing import Optional

from pydantic import BaseModel

from copilot.cards.schema import PaperCardModel
from evals.adapters.scirex import GoldCluster, SciREXDoc

FIELDS = ("tasks", "datasets", "methods", "metrics")


def normalize(name: str) -> str:
    """Casefold and keep only letters and digits: "V - Net" and "V-Net" both give "vnet"."""
    return "".join(ch for ch in name.casefold() if ch.isalnum())


class Counts(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold: int = 0
    gold_with_mentions: int = 0  # clusters that can be found at all: the recall ceiling

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{key: getattr(self, key) + getattr(other, key) for key in Counts.model_fields})


class Scores(BaseModel):
    precision: float
    recall: float
    f1: float
    recall_ceiling: Optional[float]


def max_matching(edges: list[set[int]], n_right: int) -> int:
    """Size of a maximum matching; edges[i] holds the clusters prediction i can match."""
    match_of_right = [-1] * n_right

    def try_assign(left: int, seen: set[int]) -> bool:
        for right in edges[left]:
            if right in seen:
                continue
            seen.add(right)
            if match_of_right[right] == -1 or try_assign(match_of_right[right], seen):
                match_of_right[right] = left
                return True
        return False

    return sum(try_assign(left, set()) for left in range(len(edges)))


def match_field(predictions: list[str], clusters: list[GoldCluster]) -> Counts:
    """One field of one paper: predictions are scoring names, clusters the gold of that field."""
    cluster_keys = [{normalize(m) for m in cluster.mentions} - {""} for cluster in clusters]
    edges = []
    for prediction in predictions:
        key = normalize(prediction)
        edges.append({j for j, keys in enumerate(cluster_keys) if key and key in keys})
    tp = max_matching(edges, len(clusters))
    return Counts(
        tp=tp,
        fp=len(predictions) - tp,
        fn=len(clusters) - tp,
        gold=len(clusters),
        gold_with_mentions=sum(bool(keys) for keys in cluster_keys),
    )


def score_paper(card: Optional[PaperCardModel], doc: SciREXDoc) -> dict[str, Counts]:
    """Counts per field for one paper. A missing card scores as an empty card."""
    card = card or PaperCardModel()
    return {
        field: match_field(
            [entity.names[0] for entity in getattr(card, field)],
            [cluster for cluster in doc.gold if cluster.field == field],
        )
        for field in FIELDS
    }


def scores(counts: Counts) -> Scores:
    """Micro precision, recall and F1 from summed counts. 0/0 counts as 0."""
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / counts.gold if counts.gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ceiling = counts.gold_with_mentions / counts.gold if counts.gold else None
    return Scores(precision=precision, recall=recall, f1=f1, recall_ceiling=ceiling)


def total(per_paper: list[Counts]) -> Counts:
    return sum(per_paper, Counts())


class Interval(BaseModel):
    value: float
    lower: float
    upper: float


def bootstrap_scores(
    per_paper: list[Counts],
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Interval]:
    """Percentile CIs for micro P, R and F1: resample papers with replacement, re-sum the counts.

    Same conventions as evals/metrics/bootstrap.py: a dedicated seeded generator,
    resamples the size of the input, 2.5th and 97.5th percentiles at 95%.
    """
    if not per_paper:
        raise ValueError("no papers to bootstrap")
    rng = random.Random(seed)
    n = len(per_paper)
    samples = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_resamples):
        resampled = scores(total(rng.choices(per_paper, k=n)))
        for metric in samples:
            samples[metric].append(getattr(resampled, metric))

    tail = (1 - confidence) / 2
    point = scores(total(per_paper))
    intervals = {}
    for metric, values in samples.items():
        values.sort()
        intervals[metric] = Interval(
            value=getattr(point, metric),
            lower=values[round(tail * n_resamples)],
            upper=values[round((1 - tail) * n_resamples) - 1],
        )
    return intervals
