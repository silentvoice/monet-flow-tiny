#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-monet-flow-tiny}"
REPOSITORY="${REPOSITORY:-monet-flow}"

gcloud config set project "$PROJECT_ID"
if gcloud storage buckets describe "gs://${BUCKET_NAME}" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "bucket already exists: gs://${BUCKET_NAME}" >&2
else
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --uniform-bucket-level-access
fi

if gcloud artifacts repositories describe "$REPOSITORY" \
  --project "$PROJECT_ID" \
  --location "$REGION" >/dev/null 2>&1; then
  echo "artifact repository already exists: ${REPOSITORY}" >&2
else
  gcloud artifacts repositories create "$REPOSITORY" \
    --project "$PROJECT_ID" \
    --repository-format docker \
    --location "$REGION" \
    --description "MONET Self-Flow-lite containers"
fi

echo "bucket=gs://${BUCKET_NAME}"
echo "artifact_repository=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
