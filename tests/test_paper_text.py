"""Tests for the text the card extractor reads (copilot/cards/paper_text.py).

Each case is a decision recorded in reports/a02.md §8: title, abstract, then each section's
heading above its paragraphs; back matter (acknowledgements, ethics, AI-use and
reproducibility statements, references, author contributions, funding and broader impact)
left out; appendices and every other section kept. The headings are real ones from the
A1 store.
"""

from datetime import datetime, timezone

import pytest

from copilot.cards.paper_text import is_back_matter, paper_text
from copilot.ingest.paper import PaperModel, SectionModel


def paper(*sections):
    return PaperModel(
        source="arxiv",
        id="2610.00001",
        version=1,
        title="  A Title  ",
        abstract=" The abstract. ",
        sections=[SectionModel(heading=h, paragraphs=list(ps)) for h, ps in sections],
        fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("heading", [
    "Acknowledgments",
    "Acknowledgement",
    "13Acknowledgement",
    "9 Acknowledgments & Disclosure of Funding",
    "Contributions and Acknowledgments ::: Contributors",
    "Author Contributions",
    "8 Ethics Statement",
    "6 Ethical Considerations",
    "C Ethical Use of Data",
    "Compliance with Ethical Standards",
    "Broader Impacts and Safeguards",
    "AI Use Statement",
    "AI Assistance Statement",
    "AI Usage Disclosure",
    "Appendix ::: Use of LLMs",
    "A.10 Information About Use of AI Assistants 29",
    "Conclusion ::: Generative AI Usage Disclosure.",
    "Conclusion ::: Reproducibility statement",
    "Conclusion ::: Disclosure of Interests.",
    "References",
    "8 Bibliography",
])
def test_back_matter_headings_are_dropped(heading):
    assert is_back_matter(heading)


@pytest.mark.parametrize("heading", [
    None,
    "",
    "1 Introduction",
    "1.1 Our Contributions",
    "Introduction ::: Contributions",
    "2Background",
    "Limitations",
    "A Appendix",
    "A Reproducibility Details",
    "Appendix ::: Implementation details and reproducibility",
    "6Results ::: 6.1 Reproducing the baselines",
    "Data, Code, and Ethics",
    "Ethical Dilemmas Benchmark",
    "Analysis ::: Impact of covariates",
    "4.3. Entity Extraction Against the Human Reference",
])
def test_content_headings_are_kept(heading):
    assert not is_back_matter(heading)


def test_title_then_abstract_then_sections():
    assert paper_text(paper(("1 Introduction", ["First.", "Second."]))) =="A Title\n\nAbstract\nThe abstract.\n\n1 Introduction\nFirst.\n\nSecond."


def test_back_matter_section_is_left_out_but_its_parent_is_kept():
    text = paper_text(paper(
        ("Conclusion", ["We conclude."]),
        ("Conclusion ::: AI use statement", ["We used an assistant."]),
        ("Acknowledgments", ["Thanks."]),
    ))
    assert "We conclude." in text
    assert "assistant" not in text
    assert "Thanks" not in text


def test_section_without_heading_keeps_its_paragraphs():
    assert paper_text(paper((None, ["Body."]))).endswith("The abstract.\n\nBody.")


def test_empty_paragraphs_and_heading_only_sections_are_skipped():
    text = paper_text(paper(
        ("3 Method", []),
        ("3 Method ::: 3.1 Model", ["  ", "Real text."]),
    ))
    assert text.endswith("The abstract.\n\n3 Method ::: 3.1 Model\nReal text.")
