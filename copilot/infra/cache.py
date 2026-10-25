"""Disk storage for cached LLM responses.

Storage only: what makes two calls "the same" (the key) is decided by the caller.
Entries live in evals/cache/ (gitignored) and survive restarts.
"""

from pathlib import Path

from diskcache import Cache

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = ROOT_DIR / "evals" / "cache" / "llm"


class ResponseCache:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self._cache = Cache(str(directory))

    def get(self, key: str) -> dict | None:
        """Stored response for `key`, or None on a miss."""
        return self._cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self._cache.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()
