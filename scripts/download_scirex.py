"""Download the official SciREX release into data/scirex/.

Usage: uv run python scripts/download_scirex.py

Fetches release_data.tar.gz from the allenai/SciREX GitHub repo, pinned to the commit
that last changed it, extracts {train, dev, test}.jsonl, and prints a SHA-256 for
provenance. Prints only file names, sizes and line counts: no document contents.
Download only: parsing, splits and frozen subsets belong in evals/adapters/.
"""

import hashlib
import tarfile
from pathlib import Path

import httpx

COMMIT = "7ac761a6b1e3d9dfacf3147eaa621ed09665f7b4"  # 2020-04-24, last change to the archive
URL = f"https://raw.githubusercontent.com/allenai/SciREX/{COMMIT}/scirex_dataset/release_data.tar.gz"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "scirex"


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
    archive = OUT_DIR / "release_data.tar.gz"
    if archive.exists():
        print(f"skip download (exists): {archive.name}")
    else:
        print(f"downloading {URL} ...")
        download(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(OUT_DIR, filter="data")
    print(f"  sha256 {sha256(archive)}")

    print(f"\nfiles in {OUT_DIR}:")
    for p in sorted(OUT_DIR.rglob("*.jsonl")):
        with p.open() as f:
            n_docs = sum(1 for _ in f)
        print(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size / 1e6:.1f} MB, {n_docs} docs)")


if __name__ == "__main__":
    main()
