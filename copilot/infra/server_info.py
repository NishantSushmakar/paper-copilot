"""Facts about the model server that belong in cache keys and run provenance."""

from functools import lru_cache
from pathlib import Path

import httpx
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SERVER_CONFIG = ROOT_DIR / "configs" / "server.yaml"


def server_profile() -> str:
    """Label for server settings that can't be queried (see configs/server.yaml)."""
    return yaml.safe_load(SERVER_CONFIG.read_text())["profile"]


@lru_cache(maxsize=None)
def ollama_version(host: str) -> str:
    """Server version, e.g. "0.40.5". `host` is the server root, without /v1."""
    response = httpx.get(f"{host}/api/version", timeout=5)
    response.raise_for_status()
    return response.json()["version"]


@lru_cache(maxsize=None)
def ollama_digest(host: str, model: str) -> str:
    """Full sha256 digest of the weights currently behind `model` on the server.

    Read at runtime so a re-pulled or retagged model can never reuse old cache
    entries, even if the digest in models.yaml was not updated.
    """
    response = httpx.get(f"{host}/api/tags", timeout=5)
    response.raise_for_status()
    for entry in response.json()["models"]:
        if entry["name"] == model:
            return entry["digest"]
    raise LookupError(f"Model {model!r} is not installed on {host}")
