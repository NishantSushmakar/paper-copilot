"""The two A3 chunkers (reports/a03.md §3). Both read a PaperModel and see the same words.

The text of a paper is its abstract (headed "Abstract") followed by its body sections, each
heading on its own line above its paragraphs. Back matter is dropped with A2's rule
(no dev evidence is in it) and the title is left out.

* fixed_chunks: blind windows of 256 tokens with 64 tokens of overlap over that whole text
  (LangChain's split_text_on_tokens). Cuts fall wherever the token count says, inside a word
  or across sections.
* structure_chunks: whole paragraphs packed within one section up to 256 tokens
  (LangChain's RecursiveCharacterTextSplitter, run once per section, no overlap: paragraph
  breaks first, then sentence ends, then spaces). A chunk never cuts a paragraph that fits and
  never crosses a section, and starts with its section heading.

Tokens are counted with the EMBED tokenizer (the one A2's chunker loads). A chunk ID is
source:id v version:config:index:hash, so it is the same every time the same paper version is
chunked the same way, and a change of text under the same index gives an ID that no longer exists.
"""

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter, Tokenizer, split_text_on_tokens

from copilot.cards.chunking import count_tokens, tokenizer
from copilot.cards.paper_text import is_back_matter
from copilot.ingest.paper import PaperModel

CHUNK_SIZE = 256  # tokens
CHUNK_OVERLAP = 64  # fixed windows only
HASH_CHARS = 8
# The splitter adds up the tokens of the pieces it joins, and that sum can come out a little
# under the token count of the joined text, so it packs to a little less than what's left.
MARGIN = 4

FIXED = f"fixed-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
STRUCTURE = f"structure-{CHUNK_SIZE}-para-sent"

# A sentence ends at . ! or ? followed by whitespace and a capital, digit, quote or bracket,
# but not after "et al.", "e.g.", "Fig." and the like, or a single initial ("J. Smith").
SENTENCE_BREAK = (
    r"(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bFigs\.)"
    r"(?<!\bEq\.)(?<!\bEqs\.)(?<!\bTab\.)(?<!\bSec\.)(?<!\bSect\.)(?<!\bNo\.)(?<!\bapprox\.)"
    r"(?<!\bresp\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
)
SEPARATORS = ["\n\n", SENTENCE_BREAK, " ", ""]  # paragraph, sentence, word, character

windows = Tokenizer(
    chunk_overlap=CHUNK_OVERLAP,
    tokens_per_chunk=CHUNK_SIZE,
    decode=tokenizer.decode,
    encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    index: int
    heading: str | None  # the section of a structure-aware chunk; None for fixed windows
    token_count: int
    text: str


def sections(paper: PaperModel) -> list[tuple[str | None, list[str]]]:
    """(heading, paragraphs) for the abstract and every body section that isn't back matter."""
    kept = [("Abstract", [paper.abstract])] if paper.abstract.strip() else []
    kept += [(section.heading, section.paragraphs) for section in paper.sections
             if section.paragraphs and not is_back_matter(section.heading)]
    return kept


def block(heading: str | None, paragraphs: list[str]) -> str:
    body = "\n\n".join(paragraphs)
    return f"{heading}\n{body}" if heading else body


def full_text(paper: PaperModel) -> str:
    """The text the fixed windows cut: every kept section's block, a blank line between them."""
    return "\n\n".join(block(heading, paragraphs) for heading, paragraphs in sections(paper))


def chunk_id(paper: PaperModel, config: str, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:HASH_CHARS]
    return f"{paper.source}:{paper.id}v{paper.version}:{config}:{index}:{digest}"


def fixed_chunks(paper: PaperModel) -> list[Chunk]:
    texts = split_text_on_tokens(text=full_text(paper), tokenizer=windows)
    return [Chunk(chunk_id(paper, FIXED, index, text), index, None, count_tokens(text), text)
            for index, text in enumerate(texts)]


def section_splitter(heading: str | None) -> RecursiveCharacterTextSplitter:
    """Packs a section's paragraphs into what's left of 256 tokens after its heading line."""
    heading_tokens = count_tokens(f"{heading}\n") if heading else 0
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE - heading_tokens - MARGIN,
        chunk_overlap=0,
        separators=SEPARATORS,
        is_separator_regex=True,
    )


def structure_chunks(paper: PaperModel) -> list[Chunk]:
    texts = [(heading, block(heading, [part]))
             for heading, paragraphs in sections(paper)
             for part in section_splitter(heading).split_text("\n\n".join(paragraphs))]
    return [Chunk(chunk_id(paper, STRUCTURE, index, text), index, heading, count_tokens(text), text)
            for index, (heading, text) in enumerate(texts)]


CHUNKERS = {FIXED: fixed_chunks, STRUCTURE: structure_chunks}
