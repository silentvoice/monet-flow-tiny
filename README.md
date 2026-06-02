# MONET Flow Tiny

A tiny educational text-to-image training project.

This repository trains a small latent flow model from scratch on MONET-style image latents, then compares it with a simplified Self-Flow-style ablation. It is designed for learning, debugging, and writing clear experiments, not for producing production image-generator quality.

The core idea:

```text
caption -> frozen text embedding
image   -> precomputed image latent

noise + text + timestep
  -> trainable latent flow model
  -> generated latent
  -> frozen decoder
  -> image
```

## What This Includes

- A minimal PyTorch latent flow trainer.
- A small transformer over image-latent tokens.
- Baseline rectified-flow / flow-matching objective.
- Self-Flow-lite:
  - per-token timesteps
  - token masking / heavy corruption
  - auxiliary clean-latent reconstruction loss
- MONET subset preparation from Hugging Face datasets.
- Toy-data smoke tests that run without external datasets.
- Sampling, VAE decoding, checkpoint galleries, and conditioning diagnostics.
- Optional GCS, GCE, and Vertex helper scripts using user-provided cloud settings.
- A visual long-form article in [docs/training-from-scratch-visual/article.md](docs/training-from-scratch-visual/article.md).

## What This Does Not Include

- No private project ids.
- No bucket names.
- No service-account files.
- No generated checkpoints.
- No local dataset shards.
- No published model weights.

All cloud paths in configs use placeholders such as `gs://REPLACE_WITH_BUCKET/...`.

## Project Layout

```text
configs/      Local and placeholder cloud training configs
docs/         Educational article and visual assets
scripts/      Data prep, sampling, diagnostics, and optional cloud helpers
src/          Python package source
tests/        Shape/objective tests
```

## Install

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For only the fastest local smoke test, the base install is enough:

```bash
python -m pip install -e .
```

For MONET data preparation and image decoding:

```bash
python -m pip install -e ".[data,decode]"
```

For optional cloud helpers:

```bash
python -m pip install -e ".[cloud]"
```

The cloud extra installs GCS support and the local package builder used by the
GCE and Vertex helper scripts.

## Quick Start: Synthetic Smoke Test

This creates fake latent shards and trains for a few steps. It does not download MONET or decode images.

```bash
python scripts/create_toy_subset.py --output-dir data/processed/toy --num-samples 128
python -m monet_flow.train --config configs/smoke.yaml
python -m monet_flow.train --config configs/smoke_self_flow_lite.yaml
```

Expected outputs:

```text
outputs/smoke/
outputs/smoke_self_flow_lite/
```

Each output folder contains:

```text
config.resolved.json
metrics.jsonl
checkpoints/
```

## Prepare A Small MONET Subset

MONET includes precomputed SANA DC-AE latents. This project trains on those latents so the generator can be small and understandable.

Before downloading MONET, review the dataset card and license terms linked in
[docs/sources.md](docs/sources.md). If Hugging Face requires authentication in
your environment, log in with the Hugging Face CLI or set `HF_TOKEN` before
running the preparation script.

Install data dependencies:

```bash
python -m pip install -e ".[data]"
```

Prepare a tiny probe:

```bash
python scripts/prepare_monet_subset.py \
  --output-dir data/processed/monet_probe \
  --max-samples 8 \
  --shard-size 8
```

Prepare a larger filtered subset:

```bash
python scripts/prepare_monet_subset.py \
  --output-dir data/processed/monet_10k \
  --max-samples 10000 \
  --max-scanned 150000 \
  --shard-size 1024 \
  --caption-column caption_gemini-2.5-flash-lite \
  --min-aesthetic 5.5 \
  --max-nsfw 0.2 \
  --max-watermark 0.2 \
  --min-aspect 0.75 \
  --max-aspect 1.33 \
  --min-least-dimension 512
```

Split into train/validation:

```bash
python scripts/split_subset.py \
  --input-dir data/processed/monet_10k \
  --train-dir data/processed/monet_train \
  --val-dir data/processed/monet_val \
  --val-shards 1
```

## Train Locally

Baseline:

```bash
python -m monet_flow.train --config configs/baseline.yaml
```

Self-Flow-lite:

```bash
python -m monet_flow.train --config configs/self_flow_lite.yaml
```

The local configs expect:

```text
data/processed/monet_train/
data/processed/monet_val/
```

## Sample A Checkpoint

Install decode dependencies:

```bash
python -m pip install -e ".[decode]"
```

Generate and decode a grid:

```bash
python -m monet_flow.sample \
  --config configs/baseline.yaml \
  --checkpoint outputs/monet_baseline_flow/checkpoints/latest.pt \
  --output-dir samples/manual_baseline \
  --steps 64 \
  --guidance-scale 1.0 \
  --decode \
  --prompts \
  "a red car on a road" \
  "a cat sitting on grass" \
  "a mountain landscape at sunset" \
  "a bowl of fruit on a table"
```

The decoded grid is written to:

```text
samples/manual_baseline/grid.png
```

## Track Progress Across Checkpoints

Use the same prompts and the same initial noise for every checkpoint:

```bash
scripts/sample_progress.sh \
  configs/baseline.yaml \
  outputs/monet_baseline_flow \
  --prompts \
  "a red car on a road" \
  "a cat sitting on grass" \
  "a mountain landscape at sunset" \
  "a bowl of fruit on a table"
```

This writes:

```text
samples/progress/
  step-0002500/grid.png
  step-0005000/grid.png
  index.json
  gallery.html
```

## Optional Cloud Usage

The trainer supports `gs://` data and output paths when the `cloud` extra is installed.

Example cloud config:

```text
configs/baseline_gcs.yaml
configs/self_flow_lite_gcs.yaml
configs/monet_probe_gcs_cpu.yaml
```

Replace `REPLACE_WITH_BUCKET` with your own bucket path.

Sync local shards:

```bash
scripts/sync_to_gcs.sh \
  data/processed/monet_train \
  gs://YOUR_BUCKET/datasets/monet_10k/train

scripts/sync_to_gcs.sh \
  data/processed/monet_val \
  gs://YOUR_BUCKET/datasets/monet_10k/val
```

Create basic cloud assets:

```bash
PROJECT_ID=YOUR_PROJECT_ID \
REGION=us-central1 \
BUCKET_NAME=YOUR_BUCKET_NAME \
scripts/create_gcp_assets.sh
```

Launch a GCE training worker:

```bash
PROJECT_ID=YOUR_PROJECT_ID \
SERVICE_ACCOUNT=YOUR_SERVICE_ACCOUNT_EMAIL \
ZONE=us-central1-a \
MACHINE_TYPE=g2-standard-8 \
ACCELERATOR_TYPE= \
scripts/launch_gce_training.sh \
  configs/baseline_gcs.yaml \
  gs://YOUR_BUCKET
```

The GCE helper requires an explicit service account. Give that account only the
storage/logging permissions it needs for your bucket and logs. Set
`ACCESS_SCOPES` yourself if you intentionally need broader VM OAuth scopes.

Submit a Vertex package job:

```bash
PACKAGE_URI="$(scripts/build_and_upload_package.sh gs://YOUR_BUCKET)"

PROJECT_ID=YOUR_PROJECT_ID \
REGION=us-central1 \
scripts/submit_vertex_package_job.sh \
  "$PACKAGE_URI" \
  monet-flow-tiny-baseline \
  configs/baseline_gcs.yaml \
  gs://YOUR_BUCKET \
  monet_flow.train
```

The cloud scripts intentionally do not ship any project-specific defaults.

See [docs/cloud.md](docs/cloud.md) for the same flow as a compact guide.

## Self-Flow-Lite In This Repo

The baseline uses one timestep per image.

Self-Flow-lite changes the corruption pattern:

```text
normal flow:
  every latent patch uses the same t

self-flow-lite:
  each latent patch can use a different t
  some patches can be heavily corrupted
  an auxiliary head reconstructs clean latent tokens
```

The goal is to make the model use context and text conditioning instead of only learning a uniform cleanup rule.

## Useful Commands

Run tests:

```bash
python -m pytest
```

Run lint:

```bash
python -m ruff check .
```

Evaluate a checkpoint:

```bash
python -m monet_flow.evaluate \
  --config configs/baseline.yaml \
  --checkpoint outputs/monet_baseline_flow/checkpoints/latest.pt \
  --output outputs/eval.json
```

Diagnose prompt sensitivity:

```bash
python scripts/diagnose_checkpoint_conditioning.py \
  --config configs/baseline.yaml \
  --checkpoint outputs/monet_baseline_flow/checkpoints/latest.pt \
  --output-dir samples/diagnostics/baseline_latest \
  --steps 64 \
  --decode
```

## Notes On Image Decoding

The SANA DC-AE decoder expects latents to be divided by its configured scaling factor before decode. The decoder helper in `src/monet_flow/vae.py` handles this:

```python
decoded = vae.decode(latents.float() / scaling_factor).sample
```

If decoded images look washed out or incorrectly scaled, check this path first.

## Sources And Attribution

This repository uses MONET metadata/latents and compares against ideas from
Self-Flow. See [docs/sources.md](docs/sources.md) for dataset, paper, and
third-party project links. This repository does not redistribute MONET itself.

## License

MIT. See [LICENSE](LICENSE).
