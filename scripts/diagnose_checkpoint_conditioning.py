#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from monet_flow.config import load_config
from monet_flow.gcs import is_gcs_uri
from monet_flow.sample import sample_latents
from monet_flow.text_encoder import FrozenCLIPTextEncoder
from monet_flow.train import build_model
from monet_flow.utils import ensure_dir, get_device, seed_everything
from monet_flow.vae import decode_sana_latents, save_image_grid


def materialize_checkpoint(uri: str, cache_dir: Path) -> Path:
    if not is_gcs_uri(uri):
        return Path(uri)
    local_path = cache_dir / uri.replace("gs://", "").replace("/", "__")
    subprocess.run(["gcloud", "storage", "cp", uri, str(local_path)], check=True)
    return local_path


def summarize_pairwise_distance(values: torch.Tensor) -> dict[str, float]:
    rows = values.flatten(1).float()
    distances = torch.pdist(rows)
    if distances.numel() == 0:
        return {"mean": 0.0, "max": 0.0}
    return {
        "mean": float(distances.mean()),
        "max": float(distances.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--vae-model", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "a red car on a road",
            "a cat sitting on grass",
            "a mountain landscape at sunset",
            "a bowl of fruit on a table",
            "a portrait of a woman wearing sunglasses",
            "an old wooden chair in a sunlit room",
            "a city street after rain at night",
            "a small robot holding a flower",
        ],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = materialize_checkpoint(args.checkpoint, Path(temp_dir))
        model = build_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

    encoder = FrozenCLIPTextEncoder(args.text_encoder, device=device)
    text_embeds = encoder.encode(args.prompts).to(device)

    latent_shape = (
        len(args.prompts),
        config.data.latent_channels,
        config.data.latent_size,
        config.data.latent_size,
    )
    same_noise = torch.randn(
        1,
        config.data.latent_channels,
        config.data.latent_size,
        config.data.latent_size,
        device=device,
    ).expand(latent_shape)
    varied_noise = torch.randn(latent_shape, device=device)

    same_noise_samples = sample_latents(
        model,
        text_embeds,
        config.data.latent_channels,
        config.data.latent_size,
        args.steps,
        device,
        initial_latents=same_noise,
    )
    varied_noise_samples = sample_latents(
        model,
        text_embeds,
        config.data.latent_channels,
        config.data.latent_size,
        args.steps,
        device,
        initial_latents=varied_noise,
    )

    with torch.inference_mode():
        probe_latents = torch.randn(latent_shape, device=device)
        probe_timesteps = torch.full(
            (len(args.prompts), 1, config.data.latent_size, config.data.latent_size),
            0.5,
            device=device,
        )
        same_latent = probe_latents[:1].expand_as(probe_latents)
        text_velocity = model(same_latent, probe_timesteps, text_embeds)["velocity"]
        time_values = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device)
        time_embeds = text_embeds[:1].expand(len(time_values), -1)
        time_latents = probe_latents[:1].expand(len(time_values), -1, -1, -1)
        time_timesteps = time_values[:, None, None, None].expand(
            len(time_values),
            1,
            config.data.latent_size,
            config.data.latent_size,
        )
        time_velocity = model(time_latents, time_timesteps, time_embeds)["velocity"]

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "prompts": args.prompts,
        "sample_steps": args.steps,
        "same_noise_latent_stats": {
            "mean": float(same_noise_samples.mean()),
            "std": float(same_noise_samples.std()),
            "pairwise_l2": summarize_pairwise_distance(same_noise_samples.cpu()),
        },
        "varied_noise_latent_stats": {
            "mean": float(varied_noise_samples.mean()),
            "std": float(varied_noise_samples.std()),
            "pairwise_l2": summarize_pairwise_distance(varied_noise_samples.cpu()),
        },
        "conditioning_sensitivity": {
            "text_velocity_pairwise_l2": summarize_pairwise_distance(text_velocity.cpu()),
            "time_velocity_pairwise_l2": summarize_pairwise_distance(time_velocity.cpu()),
            "text_velocity_std": float(text_velocity.std()),
            "time_velocity_std": float(time_velocity.std()),
        },
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "same_noise_latents": same_noise_samples.cpu(),
            "varied_noise_latents": varied_noise_samples.cpu(),
            "prompts": args.prompts,
            "diagnostics": payload,
        },
        output_dir / "latents.pt",
    )

    if args.decode:
        same_noise_images = decode_sana_latents(same_noise_samples, args.vae_model, device)
        varied_noise_images = decode_sana_latents(varied_noise_samples, args.vae_model, device)
        save_image_grid(same_noise_images, output_dir / "same_noise_grid.png", columns=4)
        save_image_grid(varied_noise_images, output_dir / "varied_noise_grid.png", columns=4)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
