#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/tail_gce_training_logs.sh VM_NAME" >&2
  exit 2
fi

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
ZONE="${ZONE:-us-central1-a}"
TAIL_LINES="${TAIL_LINES:-200}"
SERIAL_START="${SERIAL_START:-}"

ARGS=(
  "$1"
  "--project=$PROJECT_ID"
  "--zone=$ZONE"
  "--port=1"
)

if [[ -n "$SERIAL_START" ]]; then
  ARGS+=("--start=$SERIAL_START")
fi

gcloud compute instances get-serial-port-output "${ARGS[@]}" | tail -n "$TAIL_LINES"
