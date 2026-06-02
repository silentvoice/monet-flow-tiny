from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from monet_flow.config import Config, load_config, to_dict
from monet_flow.data import build_dataloader
from monet_flow.flow import build_flow_batch, compute_losses
from monet_flow.gcs import download_file, is_gcs_uri, upload_file
from monet_flow.model import LatentFlowTransformer
from monet_flow.utils import (
    append_jsonl,
    autocast_dtype,
    count_parameters,
    cycle,
    ensure_dir,
    format_count,
    get_device,
    seed_everything,
)


def build_model(config: Config) -> LatentFlowTransformer:
    return LatentFlowTransformer(
        latent_channels=config.data.latent_channels,
        latent_size=config.data.latent_size,
        text_dim=config.data.text_dim,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        aux_layer=config.model.aux_layer,
        conditioning_mode=config.model.conditioning_mode,
        timestep_scale=config.model.timestep_scale,
        use_text_token=config.model.use_text_token,
        use_time_token=config.model.use_time_token,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": to_dict(config),
            "step": step,
        },
        path,
    )


def resolve_checkpoint(path: str) -> str:
    if not is_gcs_uri(path):
        return path
    cache_root = Path(os.environ.get("MONET_FLOW_GCS_CACHE", "data/cache/gcs"))
    safe_name = path.replace("gs://", "").replace("/", "__")
    return str(download_file(path, cache_root / "checkpoints" / safe_name))


def apply_text_dropout(text: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return text
    keep = torch.rand(text.shape[0], 1, device=text.device) >= float(probability)
    return text * keep.to(dtype=text.dtype)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    device: torch.device,
    max_batches: int = 16,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(val_loader):
        if batch_index >= max_batches:
            break
        clean = batch["latents"].to(device, non_blocking=True)
        text = batch["text_embeds"].to(device, non_blocking=True)
        flow_batch = build_flow_batch(clean, config.self_flow_lite)
        output = model(
            flow_batch["noised"],
            flow_batch["timesteps"],
            text,
            return_aux=config.self_flow_lite.aux_loss_weight > 0,
        )
        losses = compute_losses(
            output,
            flow_batch["target_velocity"],
            clean,
            flow_batch["mask"],
            config.self_flow_lite,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config_path: str, resume: str | None = None) -> None:
    config = load_config(config_path)
    seed_everything(config.experiment.seed)
    device = get_device()
    gcs_output_dir = None
    if is_gcs_uri(config.experiment.output_dir):
        gcs_output_dir = config.experiment.output_dir
        local_root = Path(os.environ.get("MONET_FLOW_LOCAL_OUTPUT", "outputs"))
        output_dir = ensure_dir(local_root / config.experiment.name)
    else:
        output_dir = ensure_dir(config.experiment.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = output_dir / "metrics.jsonl"
    resolved_config_path = output_dir / "config.resolved.json"

    def upload_output_file(path: Path) -> None:
        if not gcs_output_dir:
            return
        relative_path = path.relative_to(output_dir).as_posix()
        upload_file(path, f"{gcs_output_dir.rstrip('/')}/{relative_path}")

    if gcs_output_dir and not metrics_path.exists():
        try:
            download_file(f"{gcs_output_dir.rstrip('/')}/metrics.jsonl", metrics_path)
        except Exception as exc:
            if exc.__class__.__name__ != "NotFound":
                raise

    with resolved_config_path.open("w", encoding="utf-8") as handle:
        json.dump(to_dict(config), handle, indent=2, sort_keys=True)
    upload_output_file(resolved_config_path)

    train_loader = build_dataloader(
        config.data.train_dir,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = build_dataloader(
        config.data.val_dir,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_step = 0
    if resume:
        checkpoint = torch.load(resolve_checkpoint(resume), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.training.learning_rate
            group["weight_decay"] = config.training.weight_decay
        start_step = int(checkpoint.get("step", 0))
        print(
            "resumed "
            f"step={start_step} lr={config.training.learning_rate} "
            f"weight_decay={config.training.weight_decay}"
        )

    print(f"device={device} parameters={format_count(count_parameters(model))}")
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.training.precision == "fp16")
    dtype = autocast_dtype(config.training.precision)
    loader_iter = cycle(train_loader)

    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, config.training.max_steps + 1):
        running: dict[str, float] = {}
        for _ in range(config.training.grad_accum_steps):
            batch = next(loader_iter)
            clean = batch["latents"].to(device, non_blocking=True)
            text = batch["text_embeds"].to(device, non_blocking=True)
            text = apply_text_dropout(text, config.training.text_dropout_prob)
            flow_batch = build_flow_batch(clean, config.self_flow_lite)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                output = model(
                    flow_batch["noised"],
                    flow_batch["timesteps"],
                    text,
                    return_aux=config.self_flow_lite.aux_loss_weight > 0,
                )
                losses = compute_losses(
                    output,
                    flow_batch["target_velocity"],
                    clean,
                    flow_batch["mask"],
                    config.self_flow_lite,
                )
                loss = losses["loss"] / config.training.grad_accum_steps
            scaler.scale(loss).backward()
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % config.training.log_every == 0 or step == 1:
            record = {
                "step": step,
                "split": "train",
                **{key: value / config.training.grad_accum_steps for key, value in running.items()},
            }
            append_jsonl(metrics_path, record)
            print(record)

        if step % config.training.val_every == 0:
            metrics = validate(model, val_loader, config, device)
            record = {"step": step, "split": "val", **metrics}
            append_jsonl(metrics_path, record)
            print(record)

        if step % config.training.save_every == 0 or step == config.training.max_steps:
            step_checkpoint_path = checkpoint_dir / f"step-{step:07d}.pt"
            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            save_checkpoint(step_checkpoint_path, model, optimizer, config, step)
            save_checkpoint(latest_checkpoint_path, model, optimizer, config, step)
            upload_output_file(step_checkpoint_path)
            upload_output_file(latest_checkpoint_path)
            upload_output_file(metrics_path)

    if gcs_output_dir:
        upload_output_file(resolved_config_path)
        if metrics_path.exists():
            upload_output_file(metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
