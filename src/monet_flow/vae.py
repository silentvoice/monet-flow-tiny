from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


@torch.inference_mode()
def decode_sana_latents(
    latents: torch.Tensor,
    vae_model: str,
    device: str | torch.device,
) -> torch.Tensor:
    try:
        from diffusers import AutoencoderDC
    except ImportError as exc:
        raise ImportError("Install diffusers to decode SANA DC-AE latents.") from exc

    vae = AutoencoderDC.from_pretrained(vae_model, torch_dtype=torch.float32).to(device)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    decoded = vae.decode(latents.float() / scaling_factor).sample
    return decoded.clamp(-1, 1)


def _to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach().cpu().float() + 1.0) / 2.0
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def save_image_grid(images: torch.Tensor, path: str | Path, columns: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [_to_pil(image) for image in images]
    if not pil_images:
        raise ValueError("No images to save")
    width, height = pil_images[0].size
    rows = (len(pil_images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for index, image in enumerate(pil_images):
        grid.paste(image, ((index % columns) * width, (index // columns) * height))
    grid.save(path)
