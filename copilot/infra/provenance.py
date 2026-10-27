"""Provenance for eval runs: everything needed to say exactly how a number was produced.

The assignment requires every results file to record the git commit, model names,
quantization, context length, temperature, seed, subset file and date. This module
collects those facts; the harness decides which settings to pass in.
"""

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from copilot.infra.server_info import ollama_digest, ollama_version, server_profile

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRACKED_PACKAGES = ["langchain-core", "langchain-ollama", "langchain-openai", "pydantic"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_info() -> dict:
    """Current commit, and whether the working tree has uncommitted changes.

    A dirty tree means the commit hash alone does not reproduce the run.
    evals/results/ is excluded: run outputs can't change what a run computes.
    """
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)evals/results")),
    }


def file_sha256(path: Path) -> str:
    """Fingerprint of a file, e.g. a frozen subset, so silent edits are detectable."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slot_info(slot_name: str, slot) -> dict:
    """Facts about one chat slot (a validated slot from copilot.llm.slots).

    The digest and server version are read from the server at runtime.
    """
    provider = getattr(slot, "provider", "ollama")
    host = str(slot.base_url).rstrip("/").removesuffix("/v1")
    info = {
        "slot": slot_name,
        "provider": provider,
        "model": slot.model,
        "quantization": slot.quantization,
        "context_length": slot.context_length,
        "base_url": str(slot.base_url),
    }
    if provider == "ollama":
        info["digest"] = ollama_digest(host, slot.model)
        info["server_version"] = ollama_version(host)
    else:
        info["digest"] = getattr(slot, "digest", None)
    return info


def _package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(*, slots: dict, settings: dict, subset_path: Path, seed: int) -> dict:
    """Everything a results file needs to be reproducible.

    slots:        {"MEDIUM": slot_config, ...} for every slot the run used
    settings:     generation settings the harness used (temperature, max_tokens, thinking, ...)
    subset_path:  the frozen subset file
    seed:         the seed used for anything random in the run (e.g. the bootstrap)
    """
    subset_path = Path(subset_path)
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "slots": [slot_info(name, slot) for name, slot in slots.items()],
        "server_profile": server_profile(),
        "settings": settings,
        "seed": seed,
        "subset": {
            "path": str(subset_path.resolve().relative_to(ROOT_DIR)),
            "sha256": file_sha256(subset_path),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
