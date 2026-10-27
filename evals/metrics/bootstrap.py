

import random
from typing import Optional

from pydantic import BaseModel


class ConfidenceInterval(BaseModel):
    n: int
    mean: Optional[float]
    lower: Optional[float]
    upper: Optional[float]


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    n = len(scores)
    if n == 0:
        # Nothing measured: no mean and no interval, rather than a fake 0.0.
        return ConfidenceInterval(n=0, mean=None, lower=None, upper=None)

    # A dedicated generator, so nothing else in the program can shift the interval.
    rng = random.Random(seed)
    resample_means = sorted(
        sum(rng.choices(scores, k=n)) / n for _ in range(n_resamples)
    )

    # 95% -> cut 2.5% from each end: index 250 and 9,749 of 10,000 sorted means.
    tail = (1 - confidence) / 2
    lower = resample_means[round(tail * n_resamples)]
    upper = resample_means[round((1 - tail) * n_resamples) - 1]

    return ConfidenceInterval(n=n, mean=sum(scores) / n, lower=lower, upper=upper)
