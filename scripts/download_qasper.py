"""Download the official QASPER v0.3 release into data/qasper/.

Usage: uv run python scripts/download_qasper.py

Fetches the train/dev and test archives from AllenAI's S3 bucket, extracts them,
and prints a SHA-256 per archive for provenance. The test archive also contains the
official evaluator script, which you can use to cross-check your own F1.
Download only: parsing, splits and frozen subsets belong in evals/adapters/.
"""

import hashlib
import tarfile
from pathlib import Path

import httpx

BASE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
ARCHIVES = [
    "qasper-train-dev-v0.3.tgz",
    "qasper-test-and-evaluator-v0.3.tgz",
]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "qasper"


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        archive = OUT_DIR / name
        if archive.exists():
            print(f"skip download (exists): {archive.name}")
        else:
            print(f"downloading {name} ...")
            download(f"{BASE_URL}/{name}", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(OUT_DIR, filter="data")
        print(f"  sha256 {sha256(archive)}")

    print(f"\nfiles in {OUT_DIR}:")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file() and not p.name.endswith(".tgz"):
            print(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
