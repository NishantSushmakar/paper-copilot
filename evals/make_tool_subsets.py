"""Freeze dev and test subsets of the tool-calling cases.

Run once:  uv run python -m evals.make_tool_subsets
Re-running must reproduce byte-identical files; `git diff` is the check.

Same method as make_subsets.py: stratified proportionally by case kind, counts
rounded with the largest-remainder method, seed 42. Dev and test are disjoint
and together cover every case.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from evals.make_subsets import proportional_counts

SEED = 42
DEV_SIZE = 25
ROOT = Path(__file__).resolve().parent
CASES = ROOT / "data" / "tool_calls_cases.jsonl"
OUT_DIR = ROOT / "subsets"


def main() -> None:
    ids_by_kind: dict[str, list[str]] = defaultdict(list)
    with CASES.open() as f:
        for line in f:
            case = json.loads(line)
            ids_by_kind[case["kind"]].append(case["id"])

    kind_sizes = {k: len(ids) for k, ids in sorted(ids_by_kind.items())}
    dev_counts = proportional_counts(kind_sizes, DEV_SIZE)

    rng = random.Random(SEED)
    dev, test = [], []
    for kind in sorted(ids_by_kind):
        ids = sorted(ids_by_kind[kind])
        chosen = set(rng.sample(ids, dev_counts[kind]))
        dev += [i for i in ids if i in chosen]
        test += [i for i in ids if i not in chosen]

    for split, ids in (("dev", dev), ("test", test)):
        counts = {k: sum(i in set(ids) for i in v) for k, v in sorted(ids_by_kind.items())}
        subset = {
            "dataset": "tool_calls_cases",
            "split": split,
            "seed": SEED,
            "size": len(ids),
            "stratified_by": "kind (proportional)",
            "kind_counts": counts,
            "case_ids": sorted(ids),
        }
        path = OUT_DIR / f"tool_calls_{split}.json"
        path.write_text(json.dumps(subset, indent=2) + "\n")
        print(f"{split}: {len(ids)} cases -> {path.relative_to(ROOT.parent)}  {counts}")


if __name__ == "__main__":
    main()
