"""The A3 chunkers (reports/a03.md §3): stable IDs, fixed windows, structure-aware packing."""

import re
from datetime import datetime, timezone

import pytest

from copilot.cards.chunking import count_tokens
from copilot.ingest.paper import PaperModel, SectionModel
from copilot.retrieval.chunking import (
    CHUNK_OVERLAP, CHUNK_SIZE, CHUNKERS, FIXED, SENTENCE_BREAK, STRUCTURE, fixed_chunks, full_text,
    structure_chunks,
)
from copilot.cards.chunking import tokenizer


def para(section: int, n: int, sentences_: int = 4) -> str:
    """A paragraph of distinct words, so any piece of it is found in one place only."""
    return " ".join(f"Sentence s{section}p{n}x{k} reports result{section}{n}{k} on dataset{k}."
                    for k in range(sentences_))


def make_paper(sections: list[SectionModel] | None = None, version: int = 1) -> PaperModel:
    if sections is None:
        sections = [SectionModel(heading=f"{s} Section {s}", paragraphs=[para(s, n) for n in range(5)])
                    for s in range(1, 5)]
        sections.append(SectionModel(heading="Acknowledgments", paragraphs=["We thank everyone."]))
    return PaperModel(source="arxiv", id="2611.00001", version=version, title="A Title Nobody Retrieves",
                      abstract="The abstract says what the paper does.", sections=sections,
                      fetched_at=datetime(2026, 11, 7, tzinfo=timezone.utc))


@pytest.mark.parametrize("config", [FIXED, STRUCTURE])
def test_same_paper_twice_gives_the_same_ids(config):
    first = [c.chunk_id for c in CHUNKERS[config](make_paper())]
    again = [c.chunk_id for c in CHUNKERS[config](make_paper())]
    assert first == again
    assert len(set(first)) == len(first)


def test_two_configs_give_disjoint_ids():
    paper = make_paper()
    assert not {c.chunk_id for c in fixed_chunks(paper)} & {c.chunk_id for c in structure_chunks(paper)}


def test_a_new_version_gives_new_ids():
    old = {c.chunk_id for c in structure_chunks(make_paper(version=1))}
    new = {c.chunk_id for c in structure_chunks(make_paper(version=2))}
    assert not old & new


def test_changing_one_paragraph_changes_only_the_ids_of_its_chunks():
    """Structure-aware only. The edit keeps the paragraph's length, so the chunk count (and so
    every later index) stays the same; an edit that changes the count shifts later IDs too."""
    paper = make_paper()
    edited = paper.model_copy(deep=True)
    old_paragraph = edited.sections[1].paragraphs[2]
    edited.sections[1].paragraphs[2] = old_paragraph.replace("dataset0", "datasetZ")

    before, after = structure_chunks(paper), structure_chunks(edited)
    assert len(before) == len(after)
    changed = [b.index for b, a in zip(before, after) if b.chunk_id != a.chunk_id]
    containing = [b.index for b in before if "dataset0" in b.text and "s2p2x0" in b.text]
    assert changed == containing and len(changed) == 1


def test_fixed_windows_are_256_tokens_and_stitch_back_to_the_text():
    paper = make_paper()
    text, chunks = full_text(paper), fixed_chunks(paper)
    assert len(chunks) > 2
    assert all(c.token_count == CHUNK_SIZE for c in chunks[:-1])
    assert 0 < chunks[-1].token_count <= CHUNK_SIZE

    # Each window is a piece of the text; the first starts it, the last ends it, and
    # consecutive windows overlap, so no text is lost between them.
    starts = [text.index(c.text) for c in chunks]
    ends = [s + len(c.text) for s, c in zip(starts, chunks)]
    assert starts[0] == 0 and ends[-1] == len(text)
    assert all(next_start < end for next_start, end in zip(starts[1:], ends[:-1]))


def test_consecutive_fixed_windows_share_64_tokens():
    chunks = fixed_chunks(make_paper())
    ids = tokenizer(full_text(make_paper()), add_special_tokens=False)["input_ids"]
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for n in range(len(chunks) - 1):
        assert ids[n * step + step:n * step + CHUNK_SIZE] == ids[(n + 1) * step:(n + 1) * step + CHUNK_OVERLAP]


def test_fixed_text_has_headings_inline_and_drops_back_matter_and_title():
    text = full_text(make_paper())
    assert text.startswith("Abstract\nThe abstract says what the paper does.\n\n1 Section 1\n")
    assert "Acknowledgments" not in text and "We thank everyone." not in text
    assert "A Title Nobody Retrieves" not in text


def test_structure_chunks_start_with_their_heading_and_never_exceed_256_tokens():
    for chunk in structure_chunks(make_paper()):
        assert chunk.text.startswith(f"{chunk.heading}\n")
        assert chunk.token_count == count_tokens(chunk.text) <= CHUNK_SIZE  # heading included (my call)


def test_every_paragraph_that_fits_is_whole_in_exactly_one_structure_chunk():
    paper = make_paper()
    chunks = structure_chunks(paper)
    for section in paper.sections[:-1]:
        for paragraph in section.paragraphs:
            holders = [c for c in chunks if paragraph in c.text]
            assert len(holders) == 1 and holders[0].heading == section.heading


def test_structure_chunks_never_cross_sections():
    paper = make_paper()
    for chunk in structure_chunks(paper)[1:]:  # [0] is the abstract
        section = next(s for s in paper.sections if s.heading == chunk.heading)
        assert all(p in section.paragraphs for p in chunk.text.split("\n", 1)[1].split("\n\n"))


def test_back_matter_is_dropped_and_the_abstract_kept():
    chunks = structure_chunks(make_paper())
    assert chunks[0].heading == "Abstract"
    assert all(c.heading != "Acknowledgments" for c in chunks)


def test_a_long_paragraph_is_split_at_sentences():
    long_paragraph = para(1, 0, sentences_=40)
    paper = make_paper([SectionModel(heading="1 Intro", paragraphs=[long_paragraph])])
    parts = [c for c in structure_chunks(paper) if c.heading == "1 Intro"]
    assert len(parts) > 1
    assert all(c.token_count <= CHUNK_SIZE for c in parts)
    for c in parts:  # every cut is at a sentence end
        assert c.text.removeprefix("1 Intro\n").endswith(".")
    assert " ".join(c.text.removeprefix("1 Intro\n") for c in parts) == long_paragraph


def test_a_sentence_too_long_for_a_chunk_is_split_by_tokens():
    one_sentence = " ".join(f"word{k}" for k in range(600)) + "."
    paper = make_paper([SectionModel(heading="1 Table", paragraphs=[one_sentence])])
    parts = [c for c in structure_chunks(paper) if c.heading == "1 Table"]
    assert len(parts) > 1 and all(c.token_count <= CHUNK_SIZE for c in parts)


@pytest.mark.parametrize("text, expected", [
    ("We follow Smith et al. (2019). It works.", ["We follow Smith et al. (2019).", "It works."]),
    ("See Fig. 3 for details. Results improve.", ["See Fig. 3 for details.", "Results improve."]),
    ("Methods, e.g. BERT, help. Others don't.", ["Methods, e.g. BERT, help.", "Others don't."]),
    ("Done by J. Smith. Then more.", ["Done by J. Smith.", "Then more."]),
    ("Accuracy is 0.91. F1 is 0.88.", ["Accuracy is 0.91.", "F1 is 0.88."]),
])
def test_sentence_break(text, expected):
    assert re.split(SENTENCE_BREAK, text) == expected
