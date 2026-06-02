#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from monet_flow.data import LatentShardDataset


def write_shard(output_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = output_dir / f"shard-{shard_index:06d}.pt"
    payload: dict[str, Any] = {
        "latents": torch.stack([row["latents"] for row in rows]),
        "text_embeds": torch.stack([row["text_embeds"] for row in rows]),
    }
    if all("caption" in row for row in rows):
        payload["captions"] = [row["caption"] for row in rows]
    if all("id" in row for row in rows):
        payload["ids"] = [row["id"] for row in rows]
    torch.save(payload, path)
    return {"path": path.name, "num_samples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    dataset = LatentShardDataset(args.input_dir)
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    indices = indices[: args.max_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    current_rows: list[dict[str, Any]] = []
    for index in indices:
        current_rows.append(dataset[index])
        if len(current_rows) == args.shard_size:
            manifest_rows.append(write_shard(output_dir, len(manifest_rows), current_rows))
            current_rows = []
    if current_rows:
        manifest_rows.append(write_shard(output_dir, len(manifest_rows), current_rows))

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")
    captions = [dataset[index].get("caption") for index in indices[:16]]
    captions = [caption for caption in captions if isinstance(caption, str) and caption.strip()]
    if captions:
        (output_dir / "prompts.txt").write_text("\n".join(captions[:8]) + "\n", encoding="utf-8")
    summary = {
        "input_dir": args.input_dir,
        "max_samples": args.max_samples,
        "num_samples": len(indices),
        "num_shards": len(manifest_rows),
        "seed": args.seed,
        "shuffle": args.shuffle,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
