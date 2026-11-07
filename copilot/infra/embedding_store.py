"""Disk storage for chunk embeddings.

Storage only: the caller decides the model key, and the chunk ID already pins the text
(it ends in a hash of it). Entries live in evals/cache/embeddings/ (gitignored).
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from diskcache import Cache

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = ROOT_DIR / "evals" / "cache" / "embeddings"


class EmbeddingStore:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self._cache = Cache(str(directory))

    def get_many(self, model_key: str, chunk_ids: Iterable[str]) -> dict[str, np.ndarray]:
        """The stored vectors among `chunk_ids`; missing IDs are left out."""
        found = {}
        for chunk_id in chunk_ids:
            vector = self._cache.get(f"{model_key}|{chunk_id}")
            if vector is not None:
                found[chunk_id] = vector
        return found

    def set_many(self, model_key: str, vectors: Mapping[str, np.ndarray]) -> None:
        with self._cache.transact():
            for chunk_id, vector in vectors.items():
                self._cache.set(f"{model_key}|{chunk_id}", np.asarray(vector, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._cache.close()
