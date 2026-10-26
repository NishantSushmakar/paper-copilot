# Overrides

Log of "override:" requests where the AI assistant wrote core code.

| Date | Assignment | What was written | Reason |
|---|---|---|---|
| 2026-10-25 | A0 `(llm_access)` | Full `tests/test_cache.py`: key tests (same inputs, dict order, message order, each input changes the key) and integration tests (miss then hit, changed temperature misses, `0` vs `0.0`, survives reopen) with `tmp_path`, `monkeypatch` and a skip when Ollama is off | Speed: chose an override for the test code |
| 2026-10-26 | A0 `(oracle_baseline)` | `evals/make_subsets.py`: proportional stratified sampling by question type (seed 42, largest-remainder rounding) writing frozen dev/test subsets to `evals/subsets/` | User judged it routine and gave explicit permission to write the file directly |
