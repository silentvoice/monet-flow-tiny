#!/usr/bin/env bash
set -euo pipefail

python scripts/create_toy_subset.py --output-dir data/processed/toy --num-samples 128
python -m monet_flow.train --config configs/smoke.yaml
