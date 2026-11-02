from pydantic import BaseModel,HttpUrl,field_validator, Field, model_validator,TypeAdapter,ValidationError,ValidationInfo
from typing import List ,Optional
from collections.abc import Mapping
import re
import logging
from numbers import Real
logger = logging.getLogger(__name__) 
http_url_adapter = TypeAdapter(HttpUrl)


class EntityModel(BaseModel): 
    names : List[str]

    @field_validator("names",mode="after")
    @classmethod
    def clean_entity(cls,value):
        value = [item.strip() for item in value]
        value = [item for item in value if item]

        seen = set()
        result = []

        for item in value:
            normalized = item.casefold()

            if normalized not in seen:
                seen.add(normalized)
                result.append(item)

        return result



class ResultModel(BaseModel):
    dataset: str
    metric: str 
    method: str 
    raw_value:str 
    parsed_value : Optional[float]=None

    @field_validator("dataset", "metric", "method", mode="after")
    @classmethod
    def validate_names(cls, value: str, info: ValidationInfo):
        value = value.strip()

        if not value:
            raise ValueError(f"{info.field_name} cannot be empty")

        return value

    @model_validator(mode="before")
    @classmethod
    def parse_raw_value(cls, data):

        # 0. Validate input structure
        if not isinstance(data, Mapping):
            raise ValueError("Input must be a dictionary")

        data = dict(data)

        raw = data.get("raw_value")

        # 1. Convert numeric input to string
        if isinstance(raw, Real) and not isinstance(raw, bool):
            logger.info("Silent fix: converting raw_value %r to string", raw)
            raw = str(raw)
            data["raw_value"] = raw

        if not isinstance(raw, str):
            return data

        # 2. Strip whitespace for processing
        value = raw.strip()

        # 3. Handle missing-value markers
        if value.casefold() in {"—", "–", "-", "n/a"}:
            data["parsed_value"] = None
            return data

        # 4. Normalize Unicode minus
        value = value.replace("−", "-")

        

        # Identify all numeric components (main number and optional spread)
        # Commas must be correctly grouped in each component.
        components = re.split(r"±", value)

        for component in components:
            component = component.strip().removesuffix("%").strip()

            if "," in component:
                if not re.fullmatch(
                    r"[+-]?(?:\d{1,3}(?:,\d{3})+)(?:\.\d*)?",
                    component
                ):
                    raise ValueError(
                        f"Invalid comma grouping in raw_value: {raw!r}"
                    )

        # Remove validated thousands separators
        value = value.replace(",", "")

        # 6. Match allowed format:
        # [sign] number [%] [± number]

        number = r"(?:\d+(?:\.\d*)?|\.\d+)"

        pattern = (
            rf"^\s*"
            rf"(?P<main>[+-]?{number})"
            rf"\s*%?"
            rf"(?:\s*±\s*[+-]?{number}\s*%?)?"
            rf"\s*$"
        )

        match = re.fullmatch(pattern, value)

        if not match:
            raise ValueError(
                f"Invalid raw_value: {raw!r}. "
                "Expected a number, optionally followed by % or ± number."
            )

        # 7. Extract main number only
        parsed = float(match.group("main"))

        # Reject NaN and infinity
        if not __import__("math").isfinite(parsed):
            raise ValueError("parsed_value must be finite")

        data["parsed_value"] = parsed

        return data



class PaperCardModel(BaseModel):
    tasks : List[EntityModel] = Field(default_factory=list)
    datasets : List[EntityModel] = Field(default_factory=list)
    methods : List[EntityModel] = Field(default_factory=list)
    metrics : List[EntityModel] = Field(default_factory=list)
    results : List[ResultModel] = Field(default_factory=list)
    code_url: Optional[HttpUrl] = None

    @field_validator("tasks", "datasets", "methods", "metrics", mode="after")
    @classmethod
    def remove_empty_entities(cls, value):
        return [item for item in value if item.names]

    
    @field_validator("code_url", mode="before")
    @classmethod
    def validate_code_url(cls, value):
        # 1. None → None
        if value is None:
            return None

        # 2. Empty / spaces → None
        if isinstance(value, str) and not value.strip():
            return None

        # 3. Not a string → reject
        if not isinstance(value, str):
            raise ValueError(
                "code_url must be a URL string or null. "
                "If the paper provides no code link, return null."
            )

        original_value = value
        value = value.strip()

        # 4. Common missing-value markers → None
        if value.casefold() in {"none", "n/a","n.a.", "na", "null", "tbd"}:
            return None

        joined = re.sub(r"\s*([:/])\s*", r"\1", value)
        if joined != value:
            logger.info("Silent fix: removing spaces in code_url %r", original_value)
            value = joined

        # 5. Add https:// only if it looks like a hostname
        if "://" not in value:
            hostname = value.split("/")[0]

            if "." not in hostname:
                raise ValueError(
                    f"Invalid code_url: {original_value!r}. "
                    "Provide a valid URL, or null if the paper gives no link."
                )

            # Runs for hostname-like values without a scheme
            logger.info(
                "Silent fix: adding https:// to code_url %r",
                original_value
            )
            value = "https://" + value
        # 6. Validate URL
        try:
            return http_url_adapter.validate_python(value)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid code_url: {original_value!r}. "
                "Provide a valid URL, or null if the paper gives no link."
            ) from exc

    @model_validator(mode="after")
    def add_missing_entities(self):

        mappings = {
            "dataset": self.datasets,
            "metric": self.metrics,
            "method": self.methods,
        }

        for result in self.results:

            for field_name, entities in mappings.items():

                name = getattr(result, field_name)
                normalized = name.casefold()

                # Collect all existing names
                existing_names = {
                    entity_name.casefold()
                    for entity in entities
                    for entity_name in entity.names
                }

                # Add only if the name doesn't exist
                if normalized not in existing_names:

                    logger.info(
                        "Silent fix: adding missing %s entity %r from results",
                        field_name,
                        name
                    )

                    entities.append(
                        EntityModel(names=[name])
                    )

                    existing_names.add(normalized)

        return self

        
    
    