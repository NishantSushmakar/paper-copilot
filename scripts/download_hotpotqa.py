"""Download the HotpotQA distractor dev set (used in A5) into data/hotpotqa/.

Usage: uv run python scripts/download_hotpotqa.py

Fetches the distractor validation split (7,405 questions, each with its 10 paragraphs: 2 gold
and 8 distractors) from the hotpotqa/hotpot_qa Hugging Face dataset, pinned to one commit, and
prints a SHA-256 for provenance. The official test set has no public answers, so this split is
the pool for frozen dev and test subsets. Prints only the file name, size and row count.
Download only: loading, splits and frozen subsets belong in evals/adapters/.
"""

import hashlib
from pathlib import Path

import httpx
import pyarrow.parquet as pq

COMMIT = "1908d6afbbead072334abe2965f91bd2709910ab"  # hotpotqa/hotpot_qa
NAME = "distractor/validation-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{COMMIT}/{NAME}"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "hotpotqa"


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
    dest = OUT_DIR / "distractor_dev.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip download (exists): {dest.name}")
    else:
        print(f"downloading {URL} ...")
        download(URL, dest)
    rows = pq.ParquetFile(dest).metadata.num_rows
    print(f"  {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB, {rows} questions)  sha256 {sha256(dest)}")
    print(f"  columns: {pq.ParquetFile(dest).schema_arrow.names}")


if __name__ == "__main__":
    main()
