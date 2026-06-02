#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from monet_flow.gcs import gcloud_rsync, is_gcs_uri
from monet_flow.text_encoder import FrozenCLIPTextEncoder


DEFAULT_LATENT_COLUMN = "embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512"


def finite_float(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if finite_float(row, "aesthetic_jasperai", 0.0) < args.min_aesthetic:
        return False
    if finite_float(row, "nsfw_jasperai", 1.0) > args.max_nsfw:
        return False
    if finite_float(row, "wk_jasperai", 1.0) > args.max_watermark:
        return False
    aspect = finite_float(row, "aspect_ratio", 1.0)
    if aspect < args.min_aspect or aspect > args.max_aspect:
        return False
    if int(row.get("least_dimension") or 0) < args.min_least_dimension:
        return False
    if not row.get(args.caption_column):
        return False
    if not row.get(args.latent_column):
        return False
    if args.conditioning_column and not row.get(args.conditioning_column):
        return False
    return True


def latent_to_tensor(values: Any, channels: int, size: int) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    expected = channels * size * size
    if array.size != expected:
        raise ValueError(f"latent has {array.size} values, expected {expected}")
    return torch.from_numpy(array.reshape(channels, size, size))


def embedding_to_tensor(values: Any, dimension: int, normalize: bool) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.size != dimension:
        raise ValueError(f"embedding has {array.size} values, expected {dimension}")
    tensor = torch.from_numpy(array.reshape(dimension))
    if normalize:
        tensor = torch.nn.functional.normalize(tensor, dim=0)
    return tensor


def write_shard(
    output_dir: Path,
    shard_index: int,
    latents: list[torch.Tensor],
    text_embeds: torch.Tensor,
    captions: list[str],
    ids: list[str],
) -> dict[str, Any]:
    shard_name = f"shard-{shard_index:06d}.pt"
    payload = {
        "latents": torch.stack(latents).half(),
        "text_embeds": text_embeds.half(),
        "captions": captions,
        "ids": ids,
    }
    torch.save(payload, output_dir / shard_name)
    return {"path": shard_name, "num_samples": len(captions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="jasperai/monet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name", default="parquet")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-scanned", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--caption-column", default="caption_gemini-2.5-flash-lite")
    parser.add_argument("--latent-column", default=DEFAULT_LATENT_COLUMN)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument(
        "--conditioning-column",
        help=(
            "Optional MONET embedding column to store as text_embeds. "
            "For example, embedding_clip-vit-base-patch32 uses MONET's precomputed CLIP image embedding."
        ),
    )
    parser.add_argument(
        "--no-normalize-conditioning",
        action="store_true",
        help="Do not L2-normalize vectors loaded from --conditioning-column.",
    )
    parser.add_argument("--min-aesthetic", type=float, default=5.5)
    parser.add_argument("--max-nsfw", type=float, default=0.2)
    parser.add_argument("--max-watermark", type=float, default=0.2)
    parser.add_argument("--min-aspect", type=float, default=0.75)
    parser.add_argument("--max-aspect", type=float, default=1.33)
    parser.add_argument("--min-least-dimension", type=int, default=512)
    parser.add_argument("--zero-text-embeds", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to stream MONET.") from exc

    target_is_gcs = is_gcs_uri(args.output_dir)
    if target_is_gcs:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
    else:
        temp_dir = None
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = None
    if not args.zero_text_embeds and not args.conditioning_column:
        encoder = FrozenCLIPTextEncoder(args.text_encoder)

    dataset = load_dataset(
        args.dataset,
        args.config_name,
        split=args.split,
        streaming=True,
    )

    shard_index = 0
    accepted = 0
    scanned = 0
    pending_latents: list[torch.Tensor] = []
    pending_conditioning: list[torch.Tensor] = []
    pending_captions: list[str] = []
    pending_ids: list[str] = []
    manifest_rows = []
    progress = tqdm(total=args.max_samples, desc="accepted")

    def flush() -> None:
        nonlocal shard_index, pending_latents, pending_conditioning, pending_captions, pending_ids
        if not pending_captions:
            return
        if args.zero_text_embeds:
            text_embeds = torch.zeros(len(pending_captions), 512, dtype=torch.float32)
        elif args.conditioning_column:
            text_embeds = torch.stack(pending_conditioning)
        else:
            assert encoder is not None
            text_embeds = encoder.encode(pending_captions, batch_size=args.text_batch_size)
        row = write_shard(
            output_dir,
            shard_index,
            pending_latents,
            text_embeds,
            pending_captions,
            pending_ids,
        )
        manifest_rows.append(row)
        shard_index += 1
        pending_latents = []
        pending_conditioning = []
        pending_captions = []
        pending_ids = []

    for row in dataset:
        scanned += 1
        if args.max_scanned and scanned > args.max_scanned:
            break
        if not keep_row(row, args):
            continue
        try:
            latent = latent_to_tensor(row[args.latent_column], args.latent_channels, args.latent_size)
        except ValueError:
            continue
        if args.conditioning_column:
            try:
                conditioning = embedding_to_tensor(
                    row[args.conditioning_column],
                    512,
                    normalize=not args.no_normalize_conditioning,
                )
            except ValueError:
                continue
            pending_conditioning.append(conditioning)
        pending_latents.append(latent)
        pending_captions.append(str(row[args.caption_column]))
        pending_ids.append(str(row.get("id") or row.get("__key__") or f"row-{scanned}"))
        accepted += 1
        progress.update(1)
        if len(pending_captions) >= args.shard_size:
            flush()
        if accepted >= args.max_samples:
            break

    flush()
    progress.close()

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "scanned": scanned,
        "accepted": accepted,
        "max_scanned": args.max_scanned,
        "caption_column": args.caption_column,
        "conditioning_column": args.conditioning_column,
        "normalize_conditioning": bool(args.conditioning_column and not args.no_normalize_conditioning),
        "latent_column": args.latent_column,
        "filters": {
            "min_aesthetic": args.min_aesthetic,
            "max_nsfw": args.max_nsfw,
            "max_watermark": args.max_watermark,
            "min_aspect": args.min_aspect,
            "max_aspect": args.max_aspect,
            "min_least_dimension": args.min_least_dimension,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if target_is_gcs:
        gcloud_rsync(output_dir, args.output_dir)
        assert temp_dir is not None
        temp_dir.cleanup()
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
