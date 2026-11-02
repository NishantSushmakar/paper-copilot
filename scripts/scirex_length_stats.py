"""Print length statistics for SciREX documents, to size chunks for A2.

Usage: uv run python scripts/scirex_length_stats.py [--split train|dev] [--budget 3000 5000]

Counts are in Qwen3 tokens of the SciREX words joined by spaces (the text the model
will see). Prints only numbers: no words, no titles, no labels. The test split is
refused on purpose; it is for the final run only.

The default tokenizer is the locally cached Qwen3-Embedding-0.6B one, which shares
its vocabulary with the Qwen3 chat models, so counts should match qwen3:4b and
qwen3:8b closely but not necessarily exactly. Check one chunk against Ollama's
prompt_eval_count before relying on a budget to the last token.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scirex" / "release_data"


def summary(name: str, values: list[int]) -> None:
    a = np.asarray(values)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    print(f"  {name:<28} n={len(a):<6} min={a.min():<6} median={p50:<8.0f} p90={p90:<8.0f} p99={p99:<8.0f} max={a.max()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--budget", type=int, nargs="*", default=[], help="paper-text token budgets to check")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def n_tokens(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

    doc_words, doc_tokens, n_sections, section_tokens, sentence_tokens = [], [], [], [], []
    per_doc_sections = []  # section token counts, per document, for the budget checks

    with (DATA_DIR / f"{args.split}.jsonl").open() as f:
        for line in f:
            doc = json.loads(line)
            words = doc["words"]
            sections = [" ".join(words[start:end]) for start, end in doc["sections"]]
            sentences = [" ".join(words[start:end]) for start, end in doc["sentences"]]

            sec = n_tokens(sections)
            doc_words.append(len(words))
            doc_tokens.append(sum(sec))
            n_sections.append(len(sections))
            section_tokens.extend(sec)
            sentence_tokens.extend(n_tokens(sentences))
            per_doc_sections.append(sec)

    print(f"SciREX {args.split}: {len(doc_tokens)} documents, tokenizer {args.tokenizer}\n")
    summary("words per document", doc_words)
    summary("tokens per document", doc_tokens)
    summary("sections per document", n_sections)
    summary("tokens per section", section_tokens)
    summary("tokens per sentence", sentence_tokens)
    ratio = sum(doc_tokens) / sum(doc_words)
    print(f"\n  tokens per word, overall: {ratio:.2f}")

    for budget in args.budget:
        chunks = [math.ceil(t / budget) for t in doc_tokens]
        long_sections = sum(t > budget for t in section_tokens)
        docs_with_long = sum(any(t > budget for t in secs) for secs in per_doc_sections)
        long_sentences = sum(t > budget for t in sentence_tokens)
        print(f"\nbudget {budget} tokens of paper text:")
        summary("chunks per document (no overlap)", chunks)
        print(f"  sections longer than budget: {long_sections} of {len(section_tokens)} ({docs_with_long} documents)")
        print(f"  sentences longer than budget: {long_sentences}")


if __name__ == "__main__":
    main()
