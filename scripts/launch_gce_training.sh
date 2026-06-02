#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/launch_gce_training.sh CONFIG_PATH gs://BUCKET/PREFIX [extra train args...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
STAGING_PREFIX="${2%/}"
shift 2

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT to the VM service-account email.}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-monet-flow-tiny-train}"
RUN_LABEL="${RUN_LABEL:-monet-flow-tiny}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-14400s}"
ACCESS_SCOPES="${ACCESS_SCOPES:-storage-rw,logging-write}"
SELF_DELETE_ON_SUCCESS="${SELF_DELETE_ON_SUCCESS:-false}"

if [[ -z "${PACKAGE_URI:-}" ]]; then
  PACKAGE_URI="$(PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}" scripts/build_and_upload_package.sh "$STAGING_PREFIX")"
fi

SERVICE_ACCOUNT="$SERVICE_ACCOUNT" \
PROJECT_ID="$PROJECT_ID" \
RUN_LABEL="$RUN_LABEL" \
ZONE="$ZONE" \
MACHINE_TYPE="$MACHINE_TYPE" \
ACCELERATOR_TYPE="$ACCELERATOR_TYPE" \
PROVISIONING_MODEL="$PROVISIONING_MODEL" \
MAX_RUN_DURATION="$MAX_RUN_DURATION" \
ACCESS_SCOPES="$ACCESS_SCOPES" \
SELF_DELETE_ON_SUCCESS="$SELF_DELETE_ON_SUCCESS" \
scripts/create_gce_training_vm.sh \
  "$VM_NAME" \
  "$PACKAGE_URI" \
  "$CONFIG_PATH" \
  "$STAGING_PREFIX" \
  "$@"
