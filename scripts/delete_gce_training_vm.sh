#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/delete_gce_training_vm.sh VM_NAME" >&2
  exit 2
fi

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
ZONE="${ZONE:-us-central1-a}"

gcloud compute instances delete "$1" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --quiet
