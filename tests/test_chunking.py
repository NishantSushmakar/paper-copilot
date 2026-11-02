"""Tests for the A2 chunker (copilot/cards/chunking.py).

Each case is a decision recorded in reports/a02.md §4. The chunker counts tokens with
the Qwen3 tokenizer itself, so these tests count with the same tokenizer and skip when
it isn't cached locally.

Expected interface: chunk_text(text, chunk_tokens=2600, overlap_tokens=100) returns a
list of chunks with .index, .start_word, .end_word and .text, where start_word and
end_word are offsets into text.split() and end_word is exclusive (Python slice style).

Cutting rule: at a section break (blank line) when possible, then inside a section,
between words, and inside a word only as a last resort. Overlap (at most
overlap_tokens) is only required where a cut lands inside a section.
"""

import logging

import pytest

transformers = pytest.importorskip("transformers")
try:
    TOKENIZER = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
except OSError:
    pytest.skip("Qwen3 tokenizer not cached locally", allow_module_level=True)

from copilot.cards.chunking import chunk_text  # noqa: E402

SENTENCE = "The model {} reaches 85.3 F1 on SQuAD , while the baseline gets 81.0 ."  # ~27 tokens, 15 words
SIZE, OVERLAP = 120, 30  # small sizes keep the texts short; one test uses the real defaults


def n_tokens(text):
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def text_of(n_sentences, first=0):
    # Every sentence is numbered, so no passage repeats. LangChain's add_start_index finds
    # each chunk by searching the text, and repeated passages would make it find the wrong copy.
    return " ".join(SENTENCE.format(i) for i in range(first, first + n_sentences))


def check_invariants(text, chunks, size, overlap, must_overlap):
    words = text.split()
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].start_word == 0
    assert chunks[-1].end_word == len(words)
    for c in chunks:
        assert c.text.split() == words[c.start_word:c.end_word]  # cuts fall between words
        assert n_tokens(c.text) <= size
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.start_word < nxt.start_word <= prev.end_word  # moves forward, no gap
        # overlap, where there is one, is at most `overlap` tokens
        assert n_tokens(" ".join(words[nxt.start_word:prev.end_word])) <= overlap
        if must_overlap:  # a cut inside a section repeats some words
            assert nxt.start_word < prev.end_word


def test_empty_text_gives_no_chunks():
    assert chunk_text("", SIZE, OVERLAP) == []
    assert chunk_text("   \n  ", SIZE, OVERLAP) == []


def test_short_text_is_one_chunk():
    text = text_of(3)  # ~80 tokens
    chunks = chunk_text(text, SIZE, OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].text.split() == text.split()
    assert (chunks[0].start_word, chunks[0].end_word) == (0, len(text.split()))


def test_just_over_budget_gives_an_overlapping_second_chunk():
    text = text_of(5)  # ~135 tokens: a little more than one chunk
    assert n_tokens(text) > SIZE
    chunks = chunk_text(text, SIZE, OVERLAP)
    assert len(chunks) == 2
    check_invariants(text, chunks, SIZE, OVERLAP, must_overlap=True)


def test_long_text_without_sections():
    text = text_of(40)  # ~1,000 tokens, one long section: every cut is inside it
    chunks = chunk_text(text, SIZE, OVERLAP)
    assert len(chunks) > 5
    check_invariants(text, chunks, SIZE, OVERLAP, must_overlap=True)


def test_cuts_fall_at_section_breaks():
    # sections of 2 sentences (~54 tokens) separated by blank lines: a chunk holds 2 whole sections
    sections = [text_of(2, first=2 * k) for k in range(8)]
    text = "\n\n".join(sections)
    chunks = chunk_text(text, SIZE, OVERLAP)
    check_invariants(text, chunks, SIZE, OVERLAP, must_overlap=False)
    words_per_section = len(sections[0].split())
    for c in chunks:
        assert c.start_word % words_per_section == 0  # starts at a section start
        assert c.end_word % words_per_section == 0  # ends at a section end


def test_big_section_is_cut_inside():
    # a small section, then one far bigger than a chunk
    text = text_of(1) + "\n\n" + text_of(20, first=1)
    chunks = chunk_text(text, SIZE, OVERLAP)
    assert len(chunks) > 3
    check_invariants(text, chunks, SIZE, OVERLAP, must_overlap=False)


def test_default_sizes():
    text = "\n\n".join(text_of(20, first=20 * k) for k in range(20))  # ~10,000 tokens in 20 sections, about an A1 median paper
    chunks = chunk_text(text)
    assert len(chunks) > 3
    check_invariants(text, chunks, 2600, 100, must_overlap=False)


def test_word_longer_than_a_chunk_is_cut_inside_and_logged(caplog):
    # one "word" of ~6,400 tokens; it never repeats itself ("x0x1x2..."), because a periodic
    # string like "ab1ab1..." makes the position of a piece ambiguous from its text alone
    huge = "".join(f"x{i}" for i in range(1500))
    with caplog.at_level(logging.INFO, logger="copilot.cards.chunking"):
        chunks = chunk_text(huge, SIZE, OVERLAP)
    assert len(chunks) >= 2
    assert all(n_tokens(c.text) <= SIZE for c in chunks)
    assert huge.startswith(chunks[0].text)
    assert huge.endswith(chunks[-1].text)
    assert any(r.name == "copilot.cards.chunking" for r in caplog.records)
