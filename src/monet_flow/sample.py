from __future__ import annotations

import argparse
from pathlib import Path

import torch

from monet_flow.config import load_config
from monet_flow.text_encoder import FrozenCLIPTextEncoder
from monet_flow.train import build_model
from monet_flow.utils import ensure_dir, get_device, seed_everything
from monet_flow.vae import decode_sana_latents, save_image_grid


@torch.inference_mode()
def sample_latents(
    model: torch.nn.Module,
    text_embeds: torch.Tensor,
    latent_channels: int,
    latent_size: int,
    steps: int,
    device: torch.device,
    initial_latents: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    batch = text_embeds.shape[0]
    if initial_latents is None:
        latents = torch.randn(batch, latent_channels, latent_size, latent_size, device=device)
    else:
        latents = initial_latents.to(device=device, dtype=text_embeds.dtype).clone()
        if latents.shape != (batch, latent_channels, latent_size, latent_size):
            raise ValueError(
                f"initial_latents shape {tuple(latents.shape)} does not match "
                f"{(batch, latent_channels, latent_size, latent_size)}"
            )
    dt = -1.0 / steps
    for index in range(steps):
        t = 1.0 - index / steps
        timesteps = torch.full((batch, 1, latent_size, latent_size), t, device=device)
        if guidance_scale == 1.0:
            velocity = model(latents, timesteps, text_embeds, return_aux=False)["velocity"]
        else:
            uncond_embeds = torch.zeros_like(text_embeds)
            uncond_velocity = model(latents, timesteps, uncond_embeds, return_aux=False)["velocity"]
            cond_velocity = model(latents, timesteps, text_embeds, return_aux=False)["velocity"]
            velocity = uncond_velocity + float(guidance_scale) * (cond_velocity - uncond_velocity)
        latents = latents + dt * velocity
    return latents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--vae-model", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    encoder = FrozenCLIPTextEncoder(args.text_encoder, device=device)
    text_embeds = encoder.encode(args.prompts).to(device)
    latents = sample_latents(
        model,
        text_embeds,
        config.data.latent_channels,
        config.data.latent_size,
        args.steps,
        device,
        guidance_scale=args.guidance_scale,
    )
    torch.save(
        {"latents": latents.cpu(), "prompts": args.prompts, "guidance_scale": args.guidance_scale},
        output_dir / "latents.pt",
    )

    if args.decode:
        images = decode_sana_latents(latents, args.vae_model, device)
        save_image_grid(images, Path(output_dir) / "grid.png")


if __name__ == "__main__":
    main()
