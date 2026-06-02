#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/sync_to_gcs.sh LOCAL_PATH gs://BUCKET/PREFIX" >&2
  exit 2
fi

gcloud storage rsync -r "$1" "$2"
