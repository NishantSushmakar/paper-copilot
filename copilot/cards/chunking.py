
import logging
import re
from dataclasses import dataclass

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)


@dataclass
class Chunk:
    index: int
    start_char: int
    end_char: int
    start_word: int
    end_word: int
    token_count: int
    text: str


def count_tokens(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


def validate_coverage(text: str, chunks: list[Chunk]):
    """Ensure every non-whitespace character and word is covered."""

    words = list(re.finditer(r"\S+", text))

    if not words:
        return

    # Merge chunk intervals to account for overlap.
    intervals = sorted(
        (chunk.start_char, chunk.end_char)
        for chunk in chunks
    )

    merged = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    # Check every word is fully covered.
    missing_words = []

    for i, word in enumerate(words):
        covered = any(
            start <= word.start() and end >= word.end()
            for start, end in merged
        )

        if not covered:
            missing_words.append(i)

    if missing_words:
        raise ValueError(
            f"Coverage failure: {len(missing_words)} words missing. "
            f"First missing word indices: {missing_words[:10]}"
        )

    # Check non-whitespace characters are covered.
    for start, end in merged:
        pass

    covered_chars = set()

    for start, end in merged:
        covered_chars.update(range(start, end))

    missing_chars = [
        i for i, char in enumerate(text)
        if not char.isspace() and i not in covered_chars
    ]

    if missing_chars:
        raise ValueError(
            f"Coverage failure: {len(missing_chars)} "
            f"non-whitespace characters missing. "
            f"First positions: {missing_chars[:10]}"
        )


def chunk_text(
    text: str,
    chunk_size: int = 2600,
    chunk_overlap: int = 100
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size"
        )

    margin = min(8, max(1, chunk_size // 100))
    safe_size = chunk_size - margin

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=safe_size,
        chunk_overlap=min(chunk_overlap, safe_size - 1),
        add_start_index=False,
        separators=["\n\n", "\n", " ", ""]
    )

    raw_chunks = splitter.split_text(text)

    words = list(re.finditer(r"\S+", text))

    chunk_data = []
    previous_start = -1

    for raw_chunk in raw_chunks:

        start = text.find(raw_chunk, previous_start + 1)

        if start == -1:
            raise ValueError(
                f"Could not locate chunk {len(chunk_data)}"
            )

        end = start + len(raw_chunk)

        token_count = count_tokens(raw_chunk)

        if token_count > chunk_size:
            raise ValueError(
                f"Chunk {len(chunk_data)} exceeds token limit: "
                f"{token_count} > {chunk_size}"
            )

        start_word = sum(
            word.end() <= start for word in words
        )

        end_word = sum(
            word.start() < end for word in words
        )

        if (
            start > 0
            and start < len(text)
            and not text[start - 1].isspace()
            and not text[start].isspace()
        ):
            logger.warning(
                "Chunk %d starts inside a word at character %d",
                len(chunk_data),
                start
            )

        if (
            end > 0
            and end < len(text)
            and not text[end - 1].isspace()
            and not text[end].isspace()
        ):
            logger.warning(
                "Chunk %d ends inside a word at character %d",
                len(chunk_data),
                end
            )

        chunk_data.append(
            Chunk(
                index=len(chunk_data),
                start_char=start,
                end_char=end,
                start_word=start_word,
                end_word=end_word,
                token_count=token_count,
                text=raw_chunk
            )
        )

        previous_start = start

    validate_coverage(text, chunk_data)

    return chunk_data
