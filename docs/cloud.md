# Optional Cloud Training

The project can run locally or against `gs://` paths.

Cloud support is intentionally generic:

- no default project id
- no default bucket
- no bundled account files
- no local auth folder

Use your own project settings through environment variables.

Install the cloud extra before using the package-build or GCS helpers:

```bash
python -m pip install -e ".[cloud]"
```

## 1. Create A Bucket And Artifact Repository

```bash
PROJECT_ID=YOUR_PROJECT_ID \
REGION=us-central1 \
BUCKET_NAME=YOUR_BUCKET_NAME \
REPOSITORY=monet-flow \
scripts/create_gcp_assets.sh
```

The script prints:

```text
bucket=gs://...
artifact_repository=...
```

## 2. Sync Data

```bash
scripts/sync_to_gcs.sh \
  data/processed/monet_train \
  gs://YOUR_BUCKET/datasets/monet_10k/train

scripts/sync_to_gcs.sh \
  data/processed/monet_val \
  gs://YOUR_BUCKET/datasets/monet_10k/val
```

## 3. Use A Cloud Config

Copy a placeholder config and replace `REPLACE_WITH_BUCKET`:

```bash
cp configs/baseline_gcs.yaml /tmp/baseline_gcs.yaml
python - <<'PY'
from pathlib import Path

path = Path("/tmp/baseline_gcs.yaml")
path.write_text(path.read_text().replace("REPLACE_WITH_BUCKET", "YOUR_BUCKET"))
PY
```

## 4. Launch A GCE Worker

The GCE helper requires an explicit service account. Grant that account access
to the training bucket and logs rather than relying on a default compute service
account. The default OAuth scopes are `storage-rw,logging-write`; override
`ACCESS_SCOPES` only when your launch path needs more.

```bash
PROJECT_ID=YOUR_PROJECT_ID \
SERVICE_ACCOUNT=YOUR_SERVICE_ACCOUNT_EMAIL \
ZONE=us-central1-a \
MACHINE_TYPE=g2-standard-8 \
ACCELERATOR_TYPE= \
scripts/launch_gce_training.sh \
  /tmp/baseline_gcs.yaml \
  gs://YOUR_BUCKET
```

Successful jobs stop the VM by default. If you want the VM to delete itself
after successful training, set `SELF_DELETE_ON_SUCCESS=true` and provide both
the required IAM permission and an access scope that permits Compute Engine API
calls.

## 5. Submit A Vertex Package Job

```bash
PACKAGE_URI="$(scripts/build_and_upload_package.sh gs://YOUR_BUCKET)"

PROJECT_ID=YOUR_PROJECT_ID \
REGION=us-central1 \
scripts/submit_vertex_package_job.sh \
  "$PACKAGE_URI" \
  monet-flow-tiny-baseline \
  /tmp/baseline_gcs.yaml \
  gs://YOUR_BUCKET \
  monet_flow.train
```

## 6. Sample A Cloud Checkpoint

```bash
python scripts/sample_checkpoint_series.py \
  --config /tmp/baseline_gcs.yaml \
  --run-dir gs://YOUR_BUCKET/runs/baseline \
  --output-dir samples/cloud_baseline_progress \
  --steps 64 \
  --guidance-scale 1.0 \
  --decode
```

To upload the gallery:

```bash
python scripts/sample_checkpoint_series.py \
  --config /tmp/baseline_gcs.yaml \
  --run-dir gs://YOUR_BUCKET/runs/baseline \
  --output-dir samples/cloud_baseline_progress \
  --upload-dir gs://YOUR_BUCKET/samples/cloud_baseline_progress \
  --steps 64 \
  --guidance-scale 1.0 \
  --decode
```
