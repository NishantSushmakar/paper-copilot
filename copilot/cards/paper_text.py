"""The text the card extractor reads for a stored paper (reports/a02.md §8).

Title, abstract, then each section: its heading on the line above its first paragraph,
and a blank line between paragraphs, so the chunker splits between paragraphs first.
Back matter (acknowledgements, ethics, AI-use and reproducibility statements, references
and the like) is left out: it rarely names a salient entity, and it costs chunks.
Appendices stay in.
"""

import re
from typing import Optional

from copilot.ingest.paper import PaperModel

HEADING_SEPARATOR = ":::"  # the parsers join nested headings: "Conclusion ::: AI use statement"

# A leading section number: "8 ", "13" (glued to the word), "A ", "A.10 ", "4.2. ". A capital
# counts only with a space or dot after it, so "Acknowledgments" and "AI Use" keep theirs.
NUMBERING = re.compile(r"^(?:[A-Z](?:\.\d+)*\.?\s+|\d+(?:\.\d+)*\.?\s*)")

# Matched at the start of each heading part, ignoring case. The variants are the ones in
# the A1 store; any other heading counts as content.
BACK_MATTER = re.compile(
    r"acknowledge?ments?\b|contributions and acknowledge?ments?\b|author contributions\b|(?:core )?contributors$"
    r"|ethics(?: statement)?$|ethical (?:considerations?|statement|concerns|use of data)\b"
    r"|compliance with ethical standards|ethics and (?:broader|societal) impacts?\b"
    r"|broader impacts?\b|impact statement\b|societal impacts?\b"
    r"|(?:generative )?ai (?:use|usage|assistan|disclosure|involvement|contribution)"
    r"|(?:information about )?use of (?:ai|llms?|large language models|generative ai)\b|llm usage\b"
    r"|reproducibility (?:statement|checklist)\b"
    r"|disclosure of (?:funding|interests)\b|funding$|competing interests\b|conflicts? of interest\b"
    r"|references$|bibliography$",
    re.IGNORECASE,
)


def is_back_matter(heading: Optional[str]) -> bool:
    """True if any part of a (nested) heading names a back-matter section."""
    if not heading:
        return False
    parts = (NUMBERING.sub("", part.strip()).strip() for part in heading.split(HEADING_SEPARATOR))
    return any(BACK_MATTER.match(part) for part in parts)


def paper_text(paper: PaperModel) -> str:
    """Title, abstract and the content sections, as one text for chunk_text."""
    blocks = [paper.title.strip(), f"Abstract\n{paper.abstract.strip()}"]
    for section in paper.sections:
        paragraphs = [p.strip() for p in section.paragraphs if p.strip()]
        if not paragraphs or is_back_matter(section.heading):
            continue  # a heading-only section adds nothing: its subsections repeat it
        if section.heading and section.heading.strip():
            paragraphs[0] = f"{section.heading.strip()}\n{paragraphs[0]}"
        blocks.extend(paragraphs)
    return "\n\n".join(blocks)
