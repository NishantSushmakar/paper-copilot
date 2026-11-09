"""The NLI slot (configs/models.yaml): tokenizer and model, loaded once, from the local cache only.

The label order differs between NLI models, so callers map through `label_id`, never a fixed index.
"""

from functools import cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from copilot.llm.slots import config

NLI = config.NLI


@cache
def nli_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(NLI.model, local_files_only=True)
    # The weights are stored in fp16; scoring in float32 keeps the probabilities stable on mps.
    model = AutoModelForSequenceClassification.from_pretrained(NLI.model, local_files_only=True, dtype=torch.float32)
    return tokenizer, model.to(NLI.device).eval()


def label_id(name: str) -> int:
    return nli_model()[1].config.label2id[name]


@cache
def model_key() -> str:
    """Model name + revision, for provenance."""
    revision = Path(snapshot_download(NLI.model, local_files_only=True)).name
    return f"{NLI.model}@{revision[:12]}/float32"
