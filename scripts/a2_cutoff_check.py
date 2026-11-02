"""Re-run the chunks that were cut off in the 3-paper dev check, and show their thinking.

    uv run python -m scripts.a2_cutoff_check
    uv run python -m scripts.a2_cutoff_check --slot SMALL_16K --strategy S1 --case 027d6c52:1

Runs your own strategy (extract_s1 / extract_s3) on each chunk, so the call lands in the
cache with the same key the dev run will use. Then repeats that exact call, which is a
cache hit, to read the thinking text that ChunkOutcome doesn't keep. Prints the status,
the tokens used, and how much of the thinking repeats itself: a high repeat rate means a
loop, which a bigger box won't fix.
"""

import argparse
import re
from collections import Counter

from copilot.cards import extract
from copilot.cards.chunking import chunk_text
from copilot.llm.llm_client import chat_with_llm
from evals.adapters.scirex import load_split

# The cut-offs from run 20261102-163447_a2_scirex_dev: (slot it failed on, strategy, paper, chunk)
CUT_OFFS = [
    ("SMALL_16K", "S1", "027d6c52", 1),
    ("MEDIUM_16K", "S3", "007ff2ca", 0),
    ("MEDIUM_16K", "S3", "027d6c52", 1),
]
STRATEGIES = {"S1": (extract.extract_s1, None), "S3": (extract.extract_s3, extract.CARD_SCHEMA)}


def repeat_rate(thinking: str) -> tuple[int, int]:
    """(sentences, repeated sentences): sentences longer than 20 characters seen before."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", thinking) if len(s.strip()) > 20]
    counts = Counter(sentences)
    return len(sentences), sum(n - 1 for n in counts.values() if n > 1)


def check(doc, slot: str, strategy: str, chunk_index: int) -> None:
    chunk = chunk_text(doc.text)[chunk_index]
    run, response_format = STRATEGIES[strategy]
    outcome = run(chunk.text, slot=slot, paper_id=doc.doc_id, chunk_index=chunk_index)

    # Same arguments as extract._single_call, so this is a cache hit.
    messages = extract.build_messages(extract.load_prompt("v3"), chunk.text)
    response = chat_with_llm(slot=slot, messages=messages, response_format=response_format, **extract.GENERATION)
    thinking = response.thinking or ""
    sentences, repeated = repeat_rate(thinking)

    print(f"\n{slot} {strategy} {doc.doc_id[:8]} chunk {chunk_index}: {outcome.status}"
          f" | prompt {outcome.prompt_tokens}, completion {outcome.completion_tokens}"
          f" of max {extract.GENERATION['max_tokens']} | {outcome.latency_s:.0f} s"
          f" | cache hit on the repeat: {response.cache_hit}")
    print(f"  thinking: {len(thinking)} chars, {sentences} sentences, {repeated} repeated")
    if outcome.error:
        print(f"  error: {outcome.error.splitlines()[0]}")
    print(f"  thinking ends: …{thinking[-400:]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slot", choices=["SMALL_16K", "MEDIUM_16K"], help="only this slot")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="only this strategy")
    parser.add_argument("--case", help="paper id prefix and chunk, e.g. 027d6c52:1 (default: the 3 cut-offs)")
    args = parser.parse_args()

    cases = CUT_OFFS
    if args.case:
        prefix, index = args.case.split(":")
        cases = [(args.slot or "MEDIUM_16K", args.strategy or "S1", prefix, int(index))]
    cases = [c for c in cases if (not args.slot or c[0] == args.slot) and (not args.strategy or c[1] == args.strategy)]

    docs = load_split("dev")
    for slot, strategy, prefix, index in cases:
        doc = next(d for d in docs if d.doc_id.startswith(prefix))
        check(doc, slot, strategy, index)


if __name__ == "__main__":
    main()
