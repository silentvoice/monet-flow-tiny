from __future__ import annotations

import torch

from monet_flow.config import SelfFlowLiteConfig
from monet_flow.flow import build_flow_batch, compute_losses
from monet_flow.model import LatentFlowTransformer


def test_baseline_flow_shapes() -> None:
    clean = torch.randn(2, 32, 16, 16)
    cfg = SelfFlowLiteConfig(enabled=False)
    batch = build_flow_batch(clean, cfg)
    assert batch["noised"].shape == clean.shape
    assert batch["target_velocity"].shape == clean.shape
    assert batch["timesteps"].shape == (2, 1, 16, 16)


def test_self_flow_lite_aux_loss() -> None:
    clean = torch.randn(2, 32, 16, 16)
    text = torch.randn(2, 512)
    cfg = SelfFlowLiteConfig(
        enabled=True,
        per_token_timesteps=True,
        mask_ratio=0.25,
        aux_loss_weight=0.1,
    )
    model = LatentFlowTransformer(
        latent_channels=32,
        latent_size=16,
        text_dim=512,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        aux_layer=1,
        timestep_scale=1.0,
        use_text_token=True,
        use_time_token=True,
    )
    batch = build_flow_batch(clean, cfg)
    output = model(batch["noised"], batch["timesteps"], text, return_aux=True)
    losses = compute_losses(output, batch["target_velocity"], clean, batch["mask"], cfg)
    assert output["velocity"].shape == clean.shape
    assert output["aux_latents"].shape == clean.shape
    assert losses["loss"].ndim == 0
    assert "aux_loss" in losses


def test_adaln_conditioning_shapes() -> None:
    clean = torch.randn(2, 32, 16, 16)
    text = torch.randn(2, 512)
    cfg = SelfFlowLiteConfig(enabled=True, per_token_timesteps=True, mask_ratio=0.1)
    model = LatentFlowTransformer(
        latent_channels=32,
        latent_size=16,
        text_dim=512,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        aux_layer=1,
        conditioning_mode="adaln",
        timestep_scale=1000.0,
        use_text_token=True,
        use_time_token=True,
    )
    batch = build_flow_batch(clean, cfg)
    output = model(batch["noised"], batch["timesteps"], text, return_aux=True)
    assert output["velocity"].shape == clean.shape
    assert output["aux_latents"].shape == clean.shape
