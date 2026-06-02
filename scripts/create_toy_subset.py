#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--text-dim", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latents = torch.randn(
        args.num_samples,
        args.latent_channels,
        args.latent_size,
        args.latent_size,
        dtype=torch.float16,
    )
    text_embeds = torch.randn(args.num_samples, args.text_dim, dtype=torch.float16)
    text_embeds = torch.nn.functional.normalize(text_embeds.float(), dim=-1).half()
    captions = [f"synthetic caption {index}" for index in range(args.num_samples)]
    ids = [f"toy-{index:06d}" for index in range(args.num_samples)]

    shard_name = "shard-000000.pt"
    torch.save(
        {
            "latents": latents,
            "text_embeds": text_embeds,
            "captions": captions,
            "ids": ids,
        },
        output_dir / shard_name,
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": shard_name, "num_samples": args.num_samples}) + "\n")
    print(f"wrote {args.num_samples} toy samples to {output_dir}")


if __name__ == "__main__":
    main()
