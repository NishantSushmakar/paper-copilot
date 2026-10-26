"""Read and write JSON Lines files (one JSON object per line).

Rows can be dicts or Pydantic models. Use `append_jsonl` inside an eval loop so
every finished row is on disk even if the run crashes halfway.
"""

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


def _to_dict(row: Any) -> dict:
    if isinstance(row, BaseModel):
        # mode="json" turns timedeltas, datetimes etc. into JSON-safe values.
        return row.model_dump(mode="json")
    return row


def append_jsonl(path: Path, row: Any) -> None:
    """Append one row and flush it to disk immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Write all rows, replacing the file if it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read every row back as a dict. Blank lines are skipped."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
