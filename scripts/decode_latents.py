#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from monet_flow.vae import decode_sana_latents, save_image_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .pt file containing a latents tensor.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--key", default="latents")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--vae-model", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu")
    latents = payload[args.key][: args.limit].float()
    images = decode_sana_latents(latents, args.vae_model, args.device)
    save_image_grid(images, Path(args.output), columns=args.columns)
    print(f"wrote {args.output} from {tuple(latents.shape)}")


if __name__ == "__main__":
    main()
