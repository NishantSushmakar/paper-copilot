import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, AnyHttpUrl, model_validator,ValidationError

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatAPIConfig(StrictModel):
    model: str
    digest: str = Field(pattern=r"^[0-9a-fA-F]{12,}$")
    base_url: AnyHttpUrl
    quantization: str
    context_length: int = Field(gt=0)


class InProcessModelConfig(StrictModel):
    model: str
    device: Literal["mps", "cpu", "cuda"]


class JudgeConfig(StrictModel):
    slot: Literal["SMALL", "MEDIUM"]


class ModelSlots(StrictModel):
    SMALL: ChatAPIConfig
    MEDIUM: ChatAPIConfig
    EMBED: InProcessModelConfig
    RERANK: InProcessModelConfig
    NLI: InProcessModelConfig
    JUDGE: JudgeConfig


def get_chat_slot( config: ModelSlots,name: str) -> ChatAPIConfig:
    name = name.upper()

    if name == "JUDGE":
        return getattr(config, config.JUDGE.slot)

    if name in {"EMBED", "RERANK", "NLI"}:
        raise ValueError(
            f"{name} is not a chat model; "
            "it is an in-process model."
        )

    if name in {"SMALL", "MEDIUM"}:
        return getattr(config, name)

    raise ValueError(
        f"Unknown chat slot {name!r}. "
        "Valid slots: SMALL, MEDIUM, JUDGE."
    )


current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
yaml_path = root_dir / "configs" / "models.yaml"

with open(yaml_path,'r') as f: 
    data = yaml.safe_load(f)


config = ModelSlots.model_validate(data)

