#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: scripts/create_gce_training_vm.sh VM_NAME PACKAGE_URI CONFIG_PATH STAGING_BUCKET [extra train args...]" >&2
  exit 2
fi

VM_NAME="$1"
PACKAGE_URI="$2"
CONFIG_PATH="$3"
STAGING_BUCKET="${4%/}"
shift 4

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id.}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-14400s}"
RUN_LABEL="${RUN_LABEL:-monet-flow}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"

case "${ACCELERATOR_TYPE,,}" in
  none|null|false)
    ACCELERATOR_TYPE=""
    ;;
esac

if [[ "$CONFIG_PATH" != gs://* ]]; then
  CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
  CONFIG_URI="${STAGING_BUCKET}/configs/${VM_NAME}-${CONFIG_BASENAME}"
  gcloud storage cp "$CONFIG_PATH" "$CONFIG_URI"
else
  CONFIG_URI="$CONFIG_PATH"
fi

STARTUP_SCRIPT="$(mktemp)"
TRAIN_ARGS=("$@")
if (( ${#TRAIN_ARGS[@]} > 0 )); then
  printf -v TRAIN_ARGS_QUOTED ' %q' "${TRAIN_ARGS[@]}"
else
  TRAIN_ARGS_QUOTED=""
fi

cat >"$STARTUP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euxo pipefail

LOG_FILE=/var/log/monet-flow-startup.log
exec > >(tee -a "\${LOG_FILE}") 2>&1

PACKAGE_URI="${PACKAGE_URI}"
CONFIG_URI="${CONFIG_URI}"
STAGING_BUCKET="${STAGING_BUCKET}"
RUN_LABEL="${RUN_LABEL}"
ZONE="${ZONE}"
TRAIN_ARGS=(${TRAIN_ARGS_QUOTED})

export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
export MONET_FLOW_GCS_CACHE=/mnt/disks/monet-cache/gcs
export MONET_FLOW_LOCAL_OUTPUT=/mnt/disks/monet-output

mkdir -p "\${MONET_FLOW_GCS_CACHE}" "\${MONET_FLOW_LOCAL_OUTPUT}" /opt/monet-flow

wait_for_metadata_token() {
  for attempt in {1..30}; do
    if curl -fsS -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
      >/dev/null; then
      return 0
    fi
    echo "metadata token unavailable, retry \${attempt}/30"
    sleep 5
  done
  echo "metadata token was not available after waiting" >&2
  return 1
}

cleanup() {
  set +e
  mkdir -p /opt/monet-flow/logs
  cp "\${LOG_FILE}" /opt/monet-flow/logs/startup.log
  gcloud storage cp "\${LOG_FILE}" "\${STAGING_BUCKET}/logs/\${RUN_LABEL}/startup-\$(hostname).log" || true
  gcloud storage rsync -r "\${MONET_FLOW_LOCAL_OUTPUT}" "\${STAGING_BUCKET}/runs-local-copy/\${RUN_LABEL}" || true
}
trap cleanup EXIT

delete_self_after_success() {
  local instance_name
  local instance_zone
  instance_name="\$(curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/name" || hostname)"
  instance_zone="\$(curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print \$NF}' || true)"
  instance_zone="\${instance_zone:-\${ZONE}}"
  gcloud compute instances delete "\${instance_name}" --zone="\${instance_zone}" --quiet \
    || shutdown -h now
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

wait_for_metadata_token
gcloud storage cp "\${PACKAGE_URI}" /opt/monet-flow/package.tar.gz
python3 -m pip install --upgrade pip
python3 -m pip install /opt/monet-flow/package.tar.gz

set +e
python3 -m monet_flow.train --config "\${CONFIG_URI}" "\${TRAIN_ARGS[@]}"
TRAIN_STATUS=\$?
set -e

if [[ "\${TRAIN_STATUS}" -eq 0 ]]; then
  cleanup
  trap - EXIT
  delete_self_after_success
fi

exit "\${TRAIN_STATUS}"
EOF

CREATE_ARGS=(
  "$VM_NAME"
  "--project=$PROJECT_ID"
  "--zone=$ZONE"
  "--machine-type=$MACHINE_TYPE"
  "--image-family=$IMAGE_FAMILY"
  "--image-project=$IMAGE_PROJECT"
  "--boot-disk-size=$BOOT_DISK_SIZE"
  "--metadata-from-file=startup-script=$STARTUP_SCRIPT"
  "--scopes=cloud-platform"
  "--maintenance-policy=TERMINATE"
  "--max-run-duration=$MAX_RUN_DURATION"
  "--instance-termination-action=DELETE"
  "--labels=project=monet-flow,run=$RUN_LABEL"
)

if [[ -n "$ACCELERATOR_TYPE" ]]; then
  CREATE_ARGS+=("--accelerator=type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}")
fi

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  CREATE_ARGS+=("--service-account=$SERVICE_ACCOUNT")
fi

if [[ -n "$PROVISIONING_MODEL" ]]; then
  CREATE_ARGS+=("--provisioning-model=$PROVISIONING_MODEL")
fi

gcloud compute instances create "${CREATE_ARGS[@]}"
rm -f "$STARTUP_SCRIPT"

echo "vm_name=${VM_NAME}"
echo "zone=${ZONE}"
echo "config_uri=${CONFIG_URI}"
echo "package_uri=${PACKAGE_URI}"
