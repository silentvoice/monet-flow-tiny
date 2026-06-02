from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def make_2d_sincos_pos_embed(size: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("hidden size must be divisible by 4 for 2D sin/cos positions")
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1.0 / (10_000**omega)
    x = x.flatten().float()[:, None] * omega[None]
    y = y.flatten().float()[:, None] * omega[None]
    return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        ff_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        attn_tokens = modulate(self.norm1(tokens), shift_attn, scale_attn)
        attn_output = self.attn(attn_tokens, attn_tokens, attn_tokens, need_weights=False)[0]
        tokens = tokens + torch.tanh(gate_attn).unsqueeze(1) * attn_output

        mlp_tokens = modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        tokens = tokens + torch.tanh(gate_mlp).unsqueeze(1) * self.mlp(mlp_tokens)
        return tokens


class LatentFlowTransformer(nn.Module):
    """Small DiT-style transformer over SANA latent tokens."""

    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        text_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        aux_layer: int,
        conditioning_mode: str = "additive",
        timestep_scale: float = 1.0,
        use_text_token: bool = False,
        use_time_token: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.aux_layer = aux_layer
        self.conditioning_mode = conditioning_mode
        self.timestep_scale = timestep_scale
        self.use_text_token = use_text_token
        self.use_time_token = use_time_token
        if self.conditioning_mode not in {"additive", "adaln"}:
            raise ValueError(
                f"Unknown conditioning_mode={self.conditioning_mode!r}; "
                "expected 'additive' or 'adaln'."
            )

        self.input_proj = nn.Conv2d(latent_channels, hidden_size, kernel_size=1)
        pos = make_2d_sincos_pos_embed(latent_size, hidden_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.text_token_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.time_token_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        if self.conditioning_mode == "adaln":
            self.condition_proj = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.layers = nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            ff_dim = int(hidden_size * mlp_ratio)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=ff_dim,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(depth)
                ]
            )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, latent_channels)
        self.aux_output = nn.Linear(hidden_size, latent_channels)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, channels, height, width = latents.shape
        if channels != self.latent_channels or height != self.latent_size or width != self.latent_size:
            raise ValueError(
                "Unexpected latent shape "
                f"{tuple(latents.shape)}; expected (*, {self.latent_channels}, "
                f"{self.latent_size}, {self.latent_size})"
            )

        tokens = self.input_proj(latents).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(tokens.dtype)

        if timesteps.ndim == 1:
            timesteps = timesteps[:, None, None, None].expand(batch, 1, height, width)
        elif timesteps.shape[-2:] != (height, width):
            timesteps = timesteps.expand(batch, 1, height, width)

        timestep_tokens = timesteps.flatten(1)
        time_features = timestep_embedding(
            timestep_tokens.reshape(-1) * float(self.timestep_scale),
            self.hidden_size,
        )
        time_features = self.time_mlp(time_features.to(tokens.dtype)).reshape(
            batch, height * width, self.hidden_size
        )
        text_features = self.text_proj(text_embeds.to(tokens.dtype))
        pooled_time = time_features.mean(dim=1)
        condition_tokens = []
        if self.use_text_token:
            condition_tokens.append(self.text_token_norm(text_features).unsqueeze(1))
        if self.use_time_token:
            time_token = self.time_token_norm(pooled_time).unsqueeze(1)
            condition_tokens.append(time_token)
        if self.conditioning_mode == "adaln":
            condition = self.condition_proj(torch.cat([text_features, pooled_time], dim=-1))
            tokens = tokens + time_features
        else:
            condition = None
            tokens = tokens + time_features + text_features.unsqueeze(1)
        latent_token_count = tokens.shape[1]
        if condition_tokens:
            tokens = torch.cat([*condition_tokens, tokens], dim=1)

        aux_tokens = None
        for layer_index, layer in enumerate(self.layers, start=1):
            if self.conditioning_mode == "adaln":
                assert condition is not None
                tokens = layer(tokens, condition)
            else:
                tokens = layer(tokens)
            if return_aux and layer_index == self.aux_layer:
                aux_tokens = tokens[:, -latent_token_count:]

        latent_tokens = tokens[:, -latent_token_count:]
        hidden = self.final_norm(latent_tokens)
        velocity = self.output(hidden).transpose(1, 2).reshape(batch, channels, height, width)
        output = {"velocity": velocity}
        if return_aux:
            if aux_tokens is None:
                aux_tokens = hidden
            aux_latents = self.aux_output(aux_tokens).transpose(1, 2).reshape(
                batch, channels, height, width
            )
            output["aux_latents"] = aux_latents
        return output
