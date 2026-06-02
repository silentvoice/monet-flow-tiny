#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: scripts/submit_vertex_job.sh IMAGE_URI DISPLAY_NAME CONFIG_PATH STAGING_BUCKET [extra args...]" >&2
  exit 2
fi

IMAGE_URI="$1"
DISPLAY_NAME="$2"
CONFIG_PATH="$3"
STAGING_BUCKET="$4"
shift 4

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
REGION="${REGION:-us-central1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_A100}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

if [[ "$CONFIG_PATH" != gs://* ]]; then
  CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
  CONFIG_URI="${STAGING_BUCKET%/}/configs/${DISPLAY_NAME}-${CONFIG_BASENAME}"
  gcloud storage cp "$CONFIG_PATH" "$CONFIG_URI"
else
  CONFIG_URI="$CONFIG_PATH"
fi

JOB_ARGS=("--config=${CONFIG_URI}" "$@")
ARGS_CSV="$(IFS=,; echo "${JOB_ARGS[*]}")"

gcloud ai custom-jobs create \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --display-name "$DISPLAY_NAME" \
  --worker-pool-spec "machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},replica-count=1,container-image-uri=${IMAGE_URI}" \
  --args="$ARGS_CSV" \
  --labels "project=monet-flow-tiny" \
  --staging-bucket "$STAGING_BUCKET"
