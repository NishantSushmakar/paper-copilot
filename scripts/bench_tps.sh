#!/usr/bin/env bash
# Measure prefill and decode tokens/sec for an Ollama model.
# Usage: bash scripts/bench_tps.sh qwen3:8b [runs]
# Every sentence carries a per-run random number so Ollama cannot reuse KV cache from an
# earlier run anywhere in the prompt (a differing prefix alone was not enough).
# Run 1 is warm-up; leave it out of reported numbers.
set -euo pipefail

MODEL=${1:-qwen3:8b}
RUNS=${2:-6}
REPS=20   # ~512 prompt tokens with the Qwen3 chat template; check "prompt N tok" in the output

for i in $(seq 1 "$RUNS"); do
  SEED=$RANDOM
  TEXT="$(for j in $(seq 1 $REPS); do printf 'Note %s-%s: transformers use self-attention to relate every token to every other token. ' "$SEED" "$j"; done)Summarise the above in detail."
  jq -n --arg m "$MODEL" --arg p "$TEXT" \
    '{model:$m, prompt:$p, stream:false, think:false, options:{num_predict:256, temperature:0}}' |
  curl -s http://localhost:11434/api/generate -d @- |
  jq -r --arg i "$i" '"run \($i): prompt \(.prompt_eval_count) tok @ \((.prompt_eval_count/.prompt_eval_duration*1e9)|floor) tok/s | output \(.eval_count) tok @ \((.eval_count/.eval_duration*1e9*10|floor)/10) tok/s | load \((.load_duration/1e9*100|floor)/100)s"'
done
