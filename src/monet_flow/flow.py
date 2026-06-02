from __future__ import annotations

import torch
import torch.nn.functional as F

from monet_flow.config import SelfFlowLiteConfig


def build_flow_batch(
    clean_latents: torch.Tensor,
    self_flow_lite: SelfFlowLiteConfig,
) -> dict[str, torch.Tensor]:
    """Create rectified-flow inputs.

    We use x_t = (1 - t) * x0 + t * x1 and train the model to predict x1 - x0.
    Sampling starts at x1 and integrates backward from t=1 to t=0.
    """

    batch, _, height, width = clean_latents.shape
    noise = torch.randn_like(clean_latents)

    if self_flow_lite.enabled and self_flow_lite.per_token_timesteps:
        timesteps = torch.rand(batch, 1, height, width, device=clean_latents.device)
    else:
        timesteps = torch.rand(batch, 1, 1, 1, device=clean_latents.device).expand(
            batch, 1, height, width
        )

    mask = torch.zeros(batch, 1, height, width, dtype=torch.bool, device=clean_latents.device)
    if self_flow_lite.enabled and self_flow_lite.mask_ratio > 0:
        mask = torch.rand(batch, 1, height, width, device=clean_latents.device) < float(
            self_flow_lite.mask_ratio
        )
        masked_t = torch.full_like(timesteps, float(self_flow_lite.masked_token_t))
        timesteps = torch.where(mask, masked_t, timesteps)

    noised = (1.0 - timesteps) * clean_latents + timesteps * noise
    target_velocity = noise - clean_latents
    return {
        "noised": noised,
        "timesteps": timesteps,
        "target_velocity": target_velocity,
        "noise": noise,
        "mask": mask,
    }


def compute_losses(
    model_output: dict[str, torch.Tensor],
    target_velocity: torch.Tensor,
    clean_latents: torch.Tensor,
    mask: torch.Tensor,
    self_flow_lite: SelfFlowLiteConfig,
) -> dict[str, torch.Tensor]:
    velocity_loss = F.mse_loss(model_output["velocity"], target_velocity)
    total = velocity_loss
    losses = {"loss": total, "velocity_loss": velocity_loss}

    if self_flow_lite.enabled and self_flow_lite.aux_loss_weight > 0:
        aux_latents = model_output["aux_latents"]
        if self_flow_lite.aux_masked_only and mask.any():
            expanded_mask = mask.expand_as(clean_latents)
            aux_loss = F.mse_loss(aux_latents[expanded_mask], clean_latents[expanded_mask])
        else:
            aux_loss = F.mse_loss(aux_latents, clean_latents)
        total = total + float(self_flow_lite.aux_loss_weight) * aux_loss
        losses["loss"] = total
        losses["aux_loss"] = aux_loss
    return losses
