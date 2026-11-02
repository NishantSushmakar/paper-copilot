import logging
from collections import Counter, defaultdict
from itertools import combinations
from typing import Literal, Optional

from pydantic import BaseModel

from copilot.cards.extract import ChunkOutcome
from copilot.cards.schema import PaperCardModel

logger = logging.getLogger(__name__)

MergeRule = Literal["share_name", "agree2"]
ENTITY_FIELDS = ("tasks", "datasets", "methods", "metrics")
RESULT_FIELDS = {"dataset": "datasets", "metric": "metrics", "method": "methods"}  # result field -> entity list


class ChunkFailure(BaseModel):
    chunk_index: int
    status: str
    failed_fields: list[str] = []


class MergedPaper(BaseModel):
    paper_id: str
    strategy: str
    slot: str
    prompt_version: str
    merge_rule: MergeRule
    chunks_total: int
    chunks_valid: int
    card: PaperCardModel
    chunk_failures: list[ChunkFailure] = []
    silent_fixes: list[str] = []  # "chunk N: ..." from every chunk, valid or not
    attempts: int  # model calls over all chunks
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def normalize(name: str) -> str:
    """The comparison key of a name: case and runs of whitespace are ignored."""
    return " ".join(name.split()).casefold()


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def merge_entities(
    cards: list[tuple[int, PaperCardModel]], field: str, rule: MergeRule
) -> tuple[list[list[str]], dict[str, str]]:
    """Merge one entity type over the chunk cards.

    Returns the merged entities as name lists (scoring name first) and a map from each
    normalized name to its entity's scoring name, for rewriting results.
    """
    groups = _UnionFind()
    chunks_with = defaultdict(set)  # normalized name -> chunks whose cards list it
    spellings = defaultdict(Counter)  # normalized name -> how often each spelling appears
    first_seen: dict[str, int] = {}  # normalized name -> order of first appearance
    pair_chunks = defaultdict(set)  # (name, name) -> chunks that put both in one entity

    for chunk_index, card in cards:
        for entity in getattr(card, field):
            keys = []
            for name in entity.names:
                key = normalize(name)
                groups.add(key)
                chunks_with[key].add(chunk_index)
                spellings[key][name] += 1
                first_seen.setdefault(key, len(first_seen))
                keys.append(key)
            for a, b in combinations(sorted(set(keys)), 2):
                pair_chunks[(a, b)].add(chunk_index)

    # A single-chunk paper has no second card to agree with, so it trusts its one card.
    min_chunks = 2 if rule == "agree2" and len(cards) > 1 else 1
    for (a, b), chunks in pair_chunks.items():
        if len(chunks) >= min_chunks:
            groups.union(a, b)

    members = defaultdict(list)
    for key in sorted(first_seen, key=first_seen.get):
        members[groups.find(key)].append(key)

    def spelling(key: str) -> str:
        # The paper's most common spelling; ties go to the one seen first.
        return spellings[key].most_common(1)[0][0]

    entities, scoring_name = [], {}
    for keys in sorted(members.values(), key=lambda keys: first_seen[keys[0]]):
        # Most chunk cards first, then the longest name, then the first seen.
        ranked = sorted(keys, key=lambda k: (-len(chunks_with[k]), -len(spelling(k)), first_seen[k]))
        names = [spelling(k) for k in ranked]
        entities.append(names)
        for key in keys:
            scoring_name[key] = names[0]
    return entities, scoring_name


def merge_results(cards: list[tuple[int, PaperCardModel]], scoring_names: dict[str, dict[str, str]]) -> list[dict]:
    """One row per distinct (dataset, metric, method, parsed value), in order of first appearance."""
    rows: dict[tuple, dict] = {}
    for _, card in cards:
        for result in card.results:
            names = {
                field: scoring_names[entity_field][normalize(getattr(result, field))]
                for field, entity_field in RESULT_FIELDS.items()
            }
            key = (names["dataset"], names["metric"], names["method"], result.parsed_value)
            rows.setdefault(key, {**names, "raw_value": result.raw_value})
    return list(rows.values())


def merge_code_url(cards: list[tuple[int, PaperCardModel]], paper_id: str) -> Optional[str]:
    """The most frequent non-null URL; ties go to the earliest chunk."""
    urls = [str(card.code_url) for _, card in cards if card.code_url is not None]
    if not urls:
        return None
    counts = Counter(urls)
    if len(counts) > 1:
        logger.info("code_url disagreement in %s: %s", paper_id, dict(counts))
    best = max(counts.values())
    return next(url for url in urls if counts[url] == best)


def merge_paper(outcomes: list[ChunkOutcome], *, merge_rule: MergeRule = "share_name") -> MergedPaper:
    """Merge one paper's chunk outcomes (one strategy, one slot) into a MergedPaper."""
    if not outcomes:
        raise ValueError("merge_paper needs at least one chunk outcome")
    first = outcomes[0]
    runs = {(o.paper_id, o.strategy, o.slot, o.prompt_version) for o in outcomes}
    if len(runs) != 1:
        raise ValueError(f"outcomes from more than one paper or run: {sorted(runs)}")

    outcomes = sorted(outcomes, key=lambda o: o.chunk_index)
    cards = [(o.chunk_index, o.card) for o in outcomes if o.status == "valid"]

    entity_lists, scoring_names = {}, {}
    for field in ENTITY_FIELDS:
        entity_lists[field], scoring_names[field] = merge_entities(cards, field, merge_rule)

    card = PaperCardModel(
        **{field: [{"names": names} for names in entity_lists[field]] for field in ENTITY_FIELDS},
        results=merge_results(cards, scoring_names),
        code_url=merge_code_url(cards, first.paper_id),
    )

    return MergedPaper(
        paper_id=first.paper_id,
        strategy=first.strategy,
        slot=first.slot,
        prompt_version=first.prompt_version,
        merge_rule=merge_rule,
        chunks_total=len(outcomes),
        chunks_valid=len(cards),
        card=card,
        chunk_failures=[
            ChunkFailure(chunk_index=o.chunk_index, status=o.status, failed_fields=o.failed_fields)
            for o in outcomes
            if o.status != "valid"
        ],
        silent_fixes=[f"chunk {o.chunk_index}: {fix}" for o in outcomes for fix in o.silent_fixes],
        attempts=sum(o.attempts for o in outcomes),
        prompt_tokens=sum(o.prompt_tokens for o in outcomes),
        completion_tokens=sum(o.completion_tokens for o in outcomes),
        latency_s=sum(o.latency_s for o in outcomes),
    )
