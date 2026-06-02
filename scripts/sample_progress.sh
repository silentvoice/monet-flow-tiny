#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/sample_progress.sh CONFIG_PATH RUN_DIR [extra sampler args...]" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
shift 2

OUTPUT_DIR="${OUTPUT_DIR:-samples/progress}"
UPLOAD_DIR="${UPLOAD_DIR:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-10}"
SAMPLE_STEPS="${SAMPLE_STEPS:-64}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"

ARGS=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-checkpoints "$MAX_CHECKPOINTS"
  --steps "$SAMPLE_STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --decode
)

if [[ -n "$UPLOAD_DIR" ]]; then
  ARGS+=(--upload-dir "$UPLOAD_DIR")
fi

if [[ -n "$PROMPTS_FILE" ]]; then
  ARGS+=(--prompts-file "$PROMPTS_FILE")
fi

scripts/sample_checkpoint_series.py "${ARGS[@]}" "$@"
