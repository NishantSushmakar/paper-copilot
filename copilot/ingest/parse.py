
import io
import re
import statistics

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from copilot.ingest.paper import SectionModel

NEST = " ::: "


def clean_text(text: str) -> str:
    """Collapse runs of whitespace and line breaks into single spaces."""
    return " ".join(text.split())


# ---------------------------------------------------------------- HTML

_SECTION_CLASSES = {"ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _is_section(tag: Tag) -> bool:
    return tag.name == "section" and bool(_SECTION_CLASSES & set(tag.get("class", [])))


def _own_heading(section: Tag) -> str | None:
    heading = section.find(lambda t: t.name in _HEADING_TAGS and "ltx_title" in t.get("class", []), recursive=False)
    if heading is None:
        return None
    for number in heading.select(".ltx_tag"):  # "3.2" or "Appendix A": QASPER headings carry no numbers
        number.decompose()
    return clean_text(heading.get_text()) or None


def html_sections(html: str) -> list[SectionModel]:
    soup = BeautifulSoup(html, "html.parser")
    # Footnotes, figures and tables would break paragraphs apart or add caption noise.
    for noise in soup.select(".ltx_note, figure"):
        noise.decompose()
    # MathML spells formulas out symbol by symbol; the LaTeX source in alttext reads better.
    for math in soup.find_all("math"):
        math.replace_with(f" {math.get('alttext', '')} ")

    sections = []
    for section in soup.find_all(_is_section):
        path = [_own_heading(s) for s in reversed(list(section.find_parents(_is_section)))]
        own = _own_heading(section)
        heading = NEST.join(h for h in [*path, own] if h) or None
        # Only paragraphs whose nearest section is this one; subsections get their own entry.
        paragraphs = [
            text
            for p in section.select("p.ltx_p")
            if p.find_parent(_is_section) is section and (text := clean_text(p.get_text()))
        ]
        sections.append(SectionModel(heading=heading, paragraphs=paragraphs))
    return sections


# ---------------------------------------------------------------- PDF

_NUMBERED_HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}|[A-H](?:\.\d{1,2}){0,2})\.?\s+([A-Z][^.!?]{1,80})$")
_NAMED_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method", "methods", "methodology",
    "experiments", "results", "discussion", "conclusion", "conclusions", "limitations",
    "ethics statement", "ethical considerations", "acknowledgments", "acknowledgements",
    "references", "bibliography", "appendix",
}
_SKIP = {"abstract", "references", "bibliography"}


def _heading_of(line: str) -> str | None:
    if line.lower().rstrip(":") in _NAMED_HEADINGS:
        return line.rstrip(":")
    match = _NUMBERED_HEADING.match(line)
    if match and len(match.group(1).split()) <= 10 and sum(c.isdigit() for c in match.group(1)) <= 3:
        return line
    return None


def _paragraphs(lines: list[str], typical_width: float) -> list[str]:
    """Rejoin wrapped lines. A paragraph ends at a short line that finishes a sentence."""
    paragraphs, current = [], ""
    for line in lines:
        if current.endswith("-") and line[:1].islower():
            current = current[:-1] + line  # word hyphenated across a line break
        else:
            current = f"{current} {line}".strip()
        if line[-1:] in ".?!" and len(line) < 0.8 * typical_width:
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


def pdf_sections(pdf_bytes: bytes) -> list[SectionModel]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    lines = [
        line.strip()
        for line in text.splitlines()
        # Page numbers and arXiv's side stamp are not content.
        if line.strip() and not line.strip().isdigit() and not line.strip().startswith("arXiv:")
    ]
    if not lines:
        return []
    typical_width = statistics.median(len(line) for line in lines)

    sections, heading, body = [], None, []

    def close():
        # Text before the first heading is the title block; the abstract and references are skipped.
        if heading is not None and heading.lower() not in _SKIP:
            sections.append(SectionModel(heading=heading, paragraphs=_paragraphs(body, typical_width)))

    for line in lines:
        new_heading = _heading_of(line)
        if new_heading:
            close()
            heading, body = new_heading, []
        else:
            body.append(line)
    close()
    return sections
