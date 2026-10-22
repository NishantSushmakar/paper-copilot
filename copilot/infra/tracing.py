"""LangSmith tracing setup.

Once enabled, every LangChain and LangGraph call is traced automatically. Grouping
calls under a run ID and attaching example IDs is done per call, by passing
`config={"run_name": ..., "tags": [...], "metadata": {...}}` to `.invoke()`.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_env(path: Path = ROOT_DIR / ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def enable_langsmith(project: str = "paper-copilot") -> None:
    """Turn on LangSmith tracing for this process, sending runs to `project`."""
    load_env()
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is not set; add it to .env")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", project)
