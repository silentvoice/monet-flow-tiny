#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: scripts/submit_vertex_package_job.sh PACKAGE_URI DISPLAY_NAME CONFIG_PATH STAGING_BUCKET PYTHON_MODULE [extra args...]" >&2
  exit 2
fi

PACKAGE_URI="$1"
DISPLAY_NAME="$2"
CONFIG_PATH="$3"
STAGING_BUCKET="$4"
PYTHON_MODULE="$5"
shift 5

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
REGION="${REGION:-us-central1}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
EXECUTOR_IMAGE_URI="${EXECUTOR_IMAGE_URI:-us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest}"

if [[ "$CONFIG_PATH" != gs://* ]]; then
  CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
  CONFIG_URI="${STAGING_BUCKET%/}/configs/${DISPLAY_NAME}-${CONFIG_BASENAME}"
  gcloud storage cp "$CONFIG_PATH" "$CONFIG_URI"
else
  CONFIG_URI="$CONFIG_PATH"
fi

WORKER_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,executor-image-uri=${EXECUTOR_IMAGE_URI},python-module=${PYTHON_MODULE}"
if [[ -n "$ACCELERATOR_TYPE" ]]; then
  WORKER_SPEC="${WORKER_SPEC},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT}"
fi

JOB_ARGS=("--config=${CONFIG_URI}" "$@")
ARGS_CSV="$(IFS=,; echo "${JOB_ARGS[*]}")"

gcloud ai custom-jobs create \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --display-name "$DISPLAY_NAME" \
  --worker-pool-spec "$WORKER_SPEC" \
  --python-package-uris "$PACKAGE_URI" \
  --args="$ARGS_CSV" \
  --labels "project=monet-flow-tiny"
