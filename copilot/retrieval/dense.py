"""Dense retrieval over one paper's chunks with the EMBED slot (Qwen3-Embedding-0.6B).

Chunks are embedded without an instruction and stored by model + chunk ID, so each chunk is
embedded once. The question gets the instruction in Qwen's format, "Instruct: ...\\nQuery:...".
Vectors are normalized, so the inner product of an exact FAISS index (IndexFlatIP) is cosine
similarity, and the ranking has no approximate-search error in it.
"""

from collections.abc import Sequence
from functools import cache
from pathlib import Path

# torch must be imported before faiss: each ships its own OpenMP runtime, and with faiss's
# loaded first, loading the model in float32 segfaulted on 5 of 5 tries (reports/a03.md §5).
import torch  # isort: skip
import faiss
import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from copilot.infra.embedding_store import EmbeddingStore
from copilot.llm.slots import config
from copilot.retrieval.chunking import Chunk
from copilot.retrieval.hit import Hit, ranked

EMBED = config.EMBED
BATCH_SIZE = 32
# The weights are stored in bfloat16 and transformers would load them that way, which leaves
# "normalized" vectors 0.2-0.3% off unit length. Computing in float32 avoids that.
DTYPE = torch.float32

# The task sentence (mine, 2026-11-07, written before any dense result). Qwen's default is
# "Given a web search query, retrieve relevant passages that answer the query".
QUERY_INSTRUCTION = "Given a question about research paper, retrieve relevant passages that help answer the question"


@cache
def model() -> SentenceTransformer:
    return SentenceTransformer(EMBED.model, device=EMBED.device, local_files_only=True,
                               model_kwargs={"dtype": DTYPE})


@cache
def model_key() -> str:
    """Model name, local snapshot revision and dtype: other weights or precision never reuse old vectors."""
    revision = Path(snapshot_download(EMBED.model, local_files_only=True)).name
    return f"{EMBED.model}@{revision[:12]}/{str(DTYPE).removeprefix('torch.')}"


@cache
def default_store() -> EmbeddingStore:
    return EmbeddingStore()


def embed_chunks(chunks: Sequence[Chunk], store: EmbeddingStore | None = None) -> np.ndarray:
    """One normalized vector per chunk, in order; only chunks not stored yet are embedded."""
    store = store or default_store()
    vectors = store.get_many(model_key(), [chunk.chunk_id for chunk in chunks])
    missing = list({chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_id not in vectors}.values())
    if missing:
        new = model().encode([chunk.text for chunk in missing], batch_size=BATCH_SIZE,
                             normalize_embeddings=True, convert_to_numpy=True)
        new_vectors = {chunk.chunk_id: vector for chunk, vector in zip(missing, new)}
        store.set_many(model_key(), new_vectors)
        vectors.update(new_vectors)
    return np.stack([vectors[chunk.chunk_id] for chunk in chunks]).astype(np.float32)


def embed_query(question: str, instruction: str | None = None) -> np.ndarray:
    instruction = QUERY_INSTRUCTION if instruction is None else instruction
    if not instruction.strip():
        raise ValueError("QUERY_INSTRUCTION is empty: write the task instruction in copilot/retrieval/dense.py")
    return model().encode([question], prompt=f"Instruct: {instruction}\nQuery:",
                          normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)


def dense_search(question: str, chunks: Sequence[Chunk], k: int = 10, instruction: str | None = None,
                 store: EmbeddingStore | None = None) -> list[Hit]:
    if not chunks:
        return []
    vectors = embed_chunks(chunks, store)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, rows = index.search(embed_query(question, instruction)[None, :], len(chunks))
    by_row = dict(zip(rows[0].tolist(), scores[0].tolist()))
    return ranked(chunks, [by_row[row] for row in range(len(chunks))], k)
