import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "scirex" / "release_data"

Field = Literal["tasks", "datasets", "methods", "metrics"]
# SciREX has no "Dataset" type: its Material is the card's datasets.
TYPE_TO_FIELD: dict[str, Field] = {"Task": "tasks", "Material": "datasets", "Method": "methods", "Metric": "metrics"}


class GoldCluster(BaseModel):
    cluster_id: str
    field: Field
    mentions: list[str]  # distinct mention strings as they appear in the text; may be empty


class SciREXDoc(BaseModel):
    doc_id: str
    text: str
    gold: list[GoldCluster]
    untyped_clusters: list[str] = []  # salient clusters with no type from either source: dropped


def doc_text(raw: dict) -> str:
    """Sections separated by a blank line, words joined by single spaces.

    Sentences are joined by spaces, not ". ": SciREX stores each full stop as a word,
    and text.split() must give back exactly SciREX's word list, so chunk offsets line
    up with the gold mention offsets.
    """
    words = raw["words"]
    return "\n\n".join(" ".join(words[start:end]) for start, end in raw["sections"])


def cluster_type(cluster_id: str, spans: list, span_tags: dict, relation_roles: dict) -> Optional[str]:
    """The cluster's role in the result relations first, then the majority NER tag of its mentions."""
    ner = Counter(span_tags[(start, end)] for start, end in spans if (start, end) in span_tags)
    roles = relation_roles.get(cluster_id, set())
    if len(roles) == 1:
        return next(iter(roles))
    if len(roles) > 1:  # one cluster in train: let its mentions break the tie
        for tag, _ in ner.most_common():
            if tag in roles:
                return tag
        return sorted(roles)[0]
    if ner:
        return ner.most_common(1)[0][0]
    return None


def parse_doc(raw: dict) -> SciREXDoc:
    relation_roles = defaultdict(set)
    for relation in raw["n_ary_relations"]:
        for role in TYPE_TO_FIELD:
            if relation.get(role):
                relation_roles[relation[role]].add(role)

    span_tags = {(start, end): tag for start, end, tag in raw["ner"]}
    words = raw["words"]
    gold, untyped = [], []
    for cluster_id, spans in raw["coref"].items():
        kind = cluster_type(cluster_id, spans, span_tags, relation_roles)
        if kind not in TYPE_TO_FIELD:
            untyped.append(cluster_id)
            continue
        mentions = list(dict.fromkeys(" ".join(words[start:end]) for start, end in spans))
        gold.append(GoldCluster(cluster_id=cluster_id, field=TYPE_TO_FIELD[kind], mentions=mentions))

    return SciREXDoc(doc_id=raw["doc_id"], text=doc_text(raw), gold=gold, untyped_clusters=untyped)


def split_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def load_split(split: Literal["train", "dev", "test"]) -> list[SciREXDoc]:
    with split_path(split).open(encoding="utf-8") as f:
        return [parse_doc(json.loads(line)) for line in f if line.strip()]
