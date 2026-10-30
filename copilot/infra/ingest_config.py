"""Load and validate configs/ingest.yaml."""

from pathlib import Path

import yaml
from pydantic import AwareDatetime, BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INGEST_CONFIG = ROOT_DIR / "configs" / "ingest.yaml"


class QasperIngestConfig(BaseModel):
    fetched_at: AwareDatetime


class ArxivIngestConfig(BaseModel):
    categories: list[str] = Field(min_length=1)
    # arXiv's terms ask for at least 3 s between requests; a smaller value is rejected.
    request_interval_seconds: float = Field(ge=3)
    page_size: int = Field(gt=0, le=2000)


class QueueConfig(BaseModel):
    db_path: Path
    max_attempts: int = Field(gt=0)
    backoff_base_seconds: float = Field(ge=0)
    backoff_jitter: float = Field(ge=0, lt=1)
    lease_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)

    @property
    def db_file(self) -> Path:
        """db_path resolved against the repo root, so it works from any working directory."""
        return self.db_path if self.db_path.is_absolute() else ROOT_DIR / self.db_path


class IngestConfig(BaseModel):
    qasper: QasperIngestConfig
    arxiv: ArxivIngestConfig
    queue: QueueConfig


def load_ingest_config(path: Path = INGEST_CONFIG) -> IngestConfig:
    """Read the ingestion config; fails loudly if a field is missing or a time has no time zone."""
    return IngestConfig.model_validate(yaml.safe_load(Path(path).read_text()))
