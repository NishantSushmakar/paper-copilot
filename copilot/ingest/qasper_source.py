import json
from pathlib import Path

from copilot.infra.ingest_config import load_ingest_config
from copilot.ingest.paper import PaperModel, SectionModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
QASPER_DIR = ROOT_DIR / "data" / "qasper"
SPLITS = ("train", "dev", "test")


def load_qasper_papers(splits: tuple[str, ...] = SPLITS) -> list[PaperModel]:
    fetched_at = load_ingest_config().qasper.fetched_at
    papers = []

    for split in splits:
        with open(QASPER_DIR / f"qasper-{split}-v0.3.json") as f:
            data = json.load(f)

        for paper_id, raw in data.items():
            sections = [
                SectionModel(
                    # QASPER marks a missing heading as either None or "": keep one form.
                    heading=sec["section_name"] or None,
                    # Some papers contain blank paragraph strings; they would become empty chunks.
                    paragraphs=[para for para in sec["paragraphs"] if para.strip()],
                )
                for sec in raw["full_text"]
            ]
            papers.append(PaperModel(
                source="qasper",
                id=paper_id,
                version=0,
                title=raw["title"],
                authors=None,
                abstract=raw["abstract"],
                sections=sections,
                fetched_at=fetched_at,
            ))

    return papers
