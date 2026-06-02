#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
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


STEP_PATTERN = re.compile(r"step-(\d+)\.pt$")
DECODE_VERSION = "sana-autoencoderdc-scaling-factor-v1"


def checkpoint_step(path: str) -> int | None:
    match = STEP_PATTERN.search(Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def find_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    if is_gcs_uri(run_dir):
        result = subprocess.run(
            ["gcloud", "storage", "ls", f"{run_dir.rstrip('/')}/checkpoints/step-*.pt"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        candidates = [str(path) for path in Path(run_dir).joinpath("checkpoints").glob("step-*.pt")]
    checkpoints = []
    for candidate in candidates:
        step = checkpoint_step(candidate)
        if step is not None:
            checkpoints.append((step, candidate))
    return sorted(checkpoints)


def materialize_checkpoint(uri: str, cache_dir: Path) -> Path:
    if not is_gcs_uri(uri):
        return Path(uri)
    local_path = cache_dir / uri.replace("gs://", "").replace("/", "__")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", uri, str(local_path)], check=True)
    return local_path


def existing_sample_is_reusable(
    latents_path: Path,
    grid_path: Path | None,
    prompts: list[str],
    step: int,
    steps: int,
    guidance_scale: float,
    noise_seed: int,
) -> bool:
    if not latents_path.exists():
        return False
    if grid_path is not None and not grid_path.exists():
        return False
    try:
        payload = torch.load(latents_path, map_location="cpu")
    except Exception:
        return False
    if payload.get("prompts") != prompts or payload.get("step") != step:
        return False
    sample_steps = payload.get("sample_steps")
    if sample_steps is not None and sample_steps != steps:
        return False
    if float(payload.get("guidance_scale", 1.0)) != float(guidance_scale):
        return False
    if grid_path is not None and payload.get("decode_version") != DECODE_VERSION:
        return False
    return bool(payload.get("fixed_initial_noise")) and int(payload.get("noise_seed", -1)) == noise_seed


def load_or_create_initial_latents(
    path: Path,
    batch: int,
    latent_channels: int,
    latent_size: int,
    noise_seed: int,
) -> torch.Tensor:
    expected_shape = (batch, latent_channels, latent_size, latent_size)
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu")
            latents = payload["initial_latents"].float()
            if tuple(latents.shape) == expected_shape and int(payload.get("noise_seed", -1)) == noise_seed:
                return latents
        except Exception:
            pass

    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    latents = torch.randn(expected_shape, generator=generator)
    torch.save(
        {
            "initial_latents": latents,
            "noise_seed": noise_seed,
            "shape": expected_shape,
        },
        path,
    )
    return latents


def write_gallery(output_dir: Path, prompts: list[str], checkpoint_rows: list[dict[str, object]]) -> None:
    rows = []
    for row in checkpoint_rows:
        step = row["step"]
        grid = row.get("grid")
        checkpoint = row.get("checkpoint")
        image_html = ""
        if isinstance(grid, str):
            image_html = f'<img src="{html.escape(str(Path(grid).relative_to(output_dir)))}" alt="step {step} sample grid">'
        rows.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>Step {step}</h2>",
                    image_html,
                    f"<p><code>{html.escape(str(checkpoint))}</code></p>",
                    "</section>",
                ]
            )
        )

    prompt_items = "\n".join(f"<li>{html.escape(prompt)}</li>" for prompt in prompts)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MONET sample progress</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1b1b1f;
      background: #f8f8f5;
    }}
    header, main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
    }}
    header {{
      padding: 32px 0 16px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 28px;
      line-height: 1.2;
    }}
    ol {{
      columns: 2;
      padding-left: 24px;
      margin: 0;
    }}
    section {{
      padding: 24px 0 36px;
      border-top: 1px solid #d9d6ca;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    img {{
      display: block;
      width: min(100%, 1024px);
      height: auto;
      border: 1px solid #d9d6ca;
      background: white;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
    @media (max-width: 720px) {{
      ol {{
        columns: 1;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MONET Sample Progress</h1>
    <ol>
{prompt_items}
    </ol>
  </header>
  <main>
{"".join(rows)}
  </main>
</body>
</html>
"""
    (output_dir / "gallery.html").write_text(document, encoding="utf-8")


def merge_reusable_existing_samples(
    output_dir: Path,
    checkpoint_rows: list[dict[str, object]],
    run_dir: str,
    prompts: list[str],
    steps: int,
    guidance_scale: float,
    noise_seed: int,
    decode: bool,
    upload_dir: str | None,
) -> list[dict[str, object]]:
    rows_by_step = {int(row["step"]): row for row in checkpoint_rows}
    upload_base = upload_dir.rstrip("/") if upload_dir else None
    for step_dir in sorted(output_dir.glob("step-*")):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.removeprefix("step-"))
        except ValueError:
            continue
        latents_path = step_dir / "latents.pt"
        grid_path = step_dir / "grid.png" if decode else None
        if not existing_sample_is_reusable(
            latents_path,
            grid_path,
            prompts,
            step,
            steps,
            guidance_scale,
            noise_seed,
        ):
            continue
        row: dict[str, object] = {
            "step": step,
            "checkpoint": f"{run_dir.rstrip('/')}/checkpoints/step-{step:07d}.pt",
            "latents": str(latents_path),
            "guidance_scale": guidance_scale,
            "reused": True,
        }
        if grid_path is not None:
            row["grid"] = str(grid_path)
        if upload_base:
            row["uploaded_latents"] = f"{upload_base}/step-{step:07d}/latents.pt"
            if grid_path is not None:
                row["uploaded_grid"] = f"{upload_base}/step-{step:07d}/grid.png"
        rows_by_step.setdefault(step, row)
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="Run root with a checkpoints/ directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upload-dir", help="Optional gs:// destination for generated sample grids.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=1701)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--only-step", type=int, help="Sample one specific checkpoint step.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth checkpoint after sorting.")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--vae-model", default="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    parser.add_argument("--prompts-file", help="Optional newline-delimited prompt file.")
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
    if args.prompts_file:
        args.prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not args.prompts:
            raise ValueError(f"No prompts found in {args.prompts_file}")

    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)
    checkpoint_rows = find_checkpoints(args.run_dir)
    if args.only_step is not None:
        checkpoint_rows = [(step, checkpoint) for step, checkpoint in checkpoint_rows if step == args.only_step]
    checkpoint_rows = checkpoint_rows[:: max(args.stride, 1)]
    if args.max_checkpoints:
        checkpoint_rows = checkpoint_rows[-args.max_checkpoints :]
    if not checkpoint_rows:
        raise FileNotFoundError(f"No step checkpoints found under {args.run_dir}/checkpoints")

    encoder = FrozenCLIPTextEncoder(args.text_encoder, device=device)
    text_embeds = encoder.encode(args.prompts).to(device)
    initial_latents = load_or_create_initial_latents(
        output_dir / "initial_noise.pt",
        len(args.prompts),
        config.data.latent_channels,
        config.data.latent_size,
        args.noise_seed,
    )
    index_rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        for step, checkpoint_uri in checkpoint_rows:
            step_dir = ensure_dir(output_dir / f"step-{step:07d}")
            latents_path = step_dir / "latents.pt"
            grid_path = step_dir / "grid.png" if args.decode else None
            row = {
                "step": step,
                "checkpoint": checkpoint_uri,
                "latents": str(latents_path),
                "guidance_scale": args.guidance_scale,
            }
            if grid_path is not None:
                row["grid"] = str(grid_path)
            if args.upload_dir:
                upload_dir = args.upload_dir.rstrip("/")
                row["uploaded_latents"] = f"{upload_dir}/step-{step:07d}/latents.pt"
                if grid_path is not None:
                    row["uploaded_grid"] = f"{upload_dir}/step-{step:07d}/grid.png"
            if existing_sample_is_reusable(
                latents_path,
                grid_path,
                args.prompts,
                step,
                args.steps,
                args.guidance_scale,
                args.noise_seed,
            ):
                row["reused"] = True
                index_rows.append(row)
                print(row)
                continue

            checkpoint_path = materialize_checkpoint(checkpoint_uri, cache_dir)
            model = build_model(config).to(device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.eval()
            latents = sample_latents(
                model,
                text_embeds,
                config.data.latent_channels,
                config.data.latent_size,
                args.steps,
                device,
                initial_latents=initial_latents,
                guidance_scale=args.guidance_scale,
            )
            torch.save(
                {
                    "latents": latents.cpu(),
                    "prompts": args.prompts,
                    "step": step,
                    "sample_steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "fixed_initial_noise": True,
                    "noise_seed": args.noise_seed,
                    "decode_version": DECODE_VERSION if args.decode else None,
                },
                latents_path,
            )
            if args.decode:
                images = decode_sana_latents(latents, args.vae_model, device)
                save_image_grid(images, grid_path)
            row["reused"] = False
            index_rows.append(row)
            print(row)

    index_rows = merge_reusable_existing_samples(
        output_dir,
        index_rows,
        args.run_dir,
        args.prompts,
        args.steps,
        args.guidance_scale,
        args.noise_seed,
        args.decode,
        args.upload_dir,
    )

    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "prompts": args.prompts,
                "fixed_initial_noise": True,
                "noise_seed": args.noise_seed,
                "checkpoints": index_rows,
            },
            handle,
            indent=2,
        )
    write_gallery(output_dir, args.prompts, index_rows)

    if args.upload_dir:
        subprocess.run(["gcloud", "storage", "rsync", "-r", str(output_dir), args.upload_dir], check=True)


if __name__ == "__main__":
    main()
