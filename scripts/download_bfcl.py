"""Download the Berkeley Function Calling Leaderboard (BFCL v3) categories used in A4 into data/bfcl/.

Usage: uv run python scripts/download_bfcl.py

Fetches the question files and their possible answers from the gorilla-llm Hugging Face
dataset, pinned to one commit, and prints a SHA-256 per file for provenance. The five
categories are simple, multiple, parallel, parallel_multiple and irrelevance; irrelevance
has no answer file, because the correct output is no call at all. Prints only file names,
sizes and line counts. Download only: loading, splits and frozen subsets belong in
evals/adapters/.
"""

import hashlib
from pathlib import Path

import httpx

COMMIT = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"  # gorilla-llm/Berkeley-Function-Calling-Leaderboard, 2026-04-29
BASE = f"https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/{COMMIT}"
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bfcl"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.rename(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = [f"BFCL_v3_{c}.json" for c in CATEGORIES]
    files += [f"possible_answer/BFCL_v3_{c}.json" for c in CATEGORIES if c != "irrelevance"]
    for name in files:
        dest = OUT_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"skip download (exists): {name}")
        else:
            download(f"{BASE}/{name}", dest)
        with dest.open() as f:
            lines = sum(1 for line in f if line.strip())  # one JSON object per line
        print(f"  {name}  ({dest.stat().st_size / 1e6:.2f} MB, {lines} lines)  sha256 {sha256(dest)[:16]}")


if __name__ == "__main__":
    main()
