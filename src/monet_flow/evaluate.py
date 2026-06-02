from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from monet_flow.config import load_config
from monet_flow.data import build_dataloader
from monet_flow.train import build_model, validate
from monet_flow.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/eval.json")
    parser.add_argument("--max-batches", type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    loader = build_dataloader(
        config.data.val_dir,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    metrics = validate(model, loader, config, device, max_batches=args.max_batches)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(metrics)


if __name__ == "__main__":
    main()
