"""Freeze QASPER dev and test subsets.

Run once:  uv run python -m evals.make_subsets
Re-running must reproduce byte-identical files; `git diff` is the check.

Sampling is stratified proportionally by question type: each type keeps its
natural share of the split. Counts are rounded with the largest-remainder
method so they always sum to the subset size. For test, only IDs and types
are read, never questions or answers.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from evals.adapters.qasper import load_split

SEED = 42
SIZE = 200
SPLITS = ["dev", "test"]
OUT_DIR = Path(__file__).resolve().parent / "subsets"


def proportional_counts(group_sizes: dict[str, int], size: int) -> dict[str, int]:
    """Split `size` across groups in proportion to their sizes, summing exactly to `size`."""
    total = sum(group_sizes.values())
    exact = {g: size * n / total for g, n in group_sizes.items()}
    counts = {g: math.floor(x) for g, x in exact.items()}
    leftover = size - sum(counts.values())
    # Give the remaining slots to the largest fractional parts (ties broken by name).
    for g in sorted(exact, key=lambda g: (-(exact[g] - counts[g]), g))[:leftover]:
        counts[g] += 1
    return counts


def make_subset(split: str) -> dict:
    ids_by_type: dict[str, list[str]] = defaultdict(list)
    for q in load_split(split):
        ids_by_type[q.question_type].append(q.question_id)

    group_sizes = {t: len(ids) for t, ids in sorted(ids_by_type.items())}
    counts = proportional_counts(group_sizes, SIZE)

    rng = random.Random(SEED)
    chosen = []
    for t in sorted(ids_by_type):
        # Sort first so the sample depends only on the seed and the set of IDs.
        chosen += rng.sample(sorted(ids_by_type[t]), counts[t])

    return {
        "dataset": "qasper-v0.3",
        "split": split,
        "seed": SEED,
        "size": SIZE,
        "stratified_by": "question_type (proportional)",
        "split_type_counts": group_sizes,
        "subset_type_counts": counts,
        "question_ids": sorted(chosen),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for split in SPLITS:
        subset = make_subset(split)
        path = OUT_DIR / f"qasper_{split}.json"
        path.write_text(json.dumps(subset, indent=2) + "\n")
        print(f"{split}: {len(subset['question_ids'])} IDs -> {path.relative_to(OUT_DIR.parent.parent)}")
        print(f"   split:  {subset['split_type_counts']}")
        print(f"   subset: {subset['subset_type_counts']}")


if __name__ == "__main__":
    main()
