# Overrides

Log of "override:" requests where the AI assistant wrote core code.

| Date | Assignment | What was written | Reason |
|---|---|---|---|
| 2026-10-25 | A0 `(llm_access)` | Full `tests/test_cache.py`: key tests (same inputs, dict order, message order, each input changes the key) and integration tests (miss then hit, changed temperature misses, `0` vs `0.0`, survives reopen) with `tmp_path`, `monkeypatch` and a skip when Ollama is off | Speed: chose an override for the test code |
| 2026-10-26 | A0 `(oracle_baseline)` | `evals/make_subsets.py`: proportional stratified sampling by question type (seed 42, largest-remainder rounding) writing frozen dev/test subsets to `evals/subsets/` | User judged it routine and gave explicit permission to write the file directly |
| 2026-10-26 | A0 `(oracle_baseline)` | One-off sanity-check snippet (not saved in the repo): runs `oracle_copy` and `always_unanswerable` on the dev subset, scores with `compare_scores`, counts F1 = 1.0 per question type | User judged it obvious and asked for it directly |
| 2026-10-26 | A0 `(oracle_baseline)` | `run_all` and `summarise` for `evals/harness.py` (given in chat to paste): loop with per-row `append_jsonl`, refuse an existing results file; summary with n, mean F1 overall and per type, error count and types, truncation/cache-hit rates, latency and token totals (`None` when no LLM responses) | User judged it something the agent could write and asked directly; design (contents of the summary, None vs 0.0) was specified in words beforehand |
| 2026-10-27 | A0 `(eval_harness)` | `evals/metrics/bootstrap.py` (given in chat to paste): percentile bootstrap CI for a mean, own seeded RNG, resamples same size as input drawn with replacement, empty input returns None | User understands the method and made the design decisions (10,000 resamples, 2.5th/97.5th percentiles, seed 42); asked the agent to write the code |
| 2026-10-27 | A0 `(oracle_baseline)` | Assembled the oracle system prompt (given in chat to paste): the user's instruction wording plus one added sentence (answer from the evidence, copy its wording), typo fixes, and three train few-shot examples with their real annotator evidence in an evidence → question → answer layout | User asked for the whole prompt after choosing the layout and evidence policy |
