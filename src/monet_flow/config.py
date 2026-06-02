from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from monet_flow.gcs import is_gcs_uri, split_gcs_uri


@dataclass
class ExperimentConfig:
    name: str = "monet_flow"
    output_dir: str = "outputs/monet_flow"
    seed: int = 17


@dataclass
class DataConfig:
    train_dir: str = "data/processed/monet_train"
    val_dir: str = "data/processed/monet_val"
    latent_channels: int = 32
    latent_size: int = 16
    text_dim: int = 512
    num_workers: int = 4


@dataclass
class ModelConfig:
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    aux_layer: int = 4
    conditioning_mode: str = "additive"
    timestep_scale: float = 1.0
    use_text_token: bool = False
    use_time_token: bool = False


@dataclass
class TrainingConfig:
    max_steps: int = 100_000
    batch_size: int = 64
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    precision: str = "bf16"
    log_every: int = 25
    save_every: int = 2_500
    val_every: int = 1_000
    max_grad_norm: float = 1.0
    text_dropout_prob: float = 0.0


@dataclass
class SelfFlowLiteConfig:
    enabled: bool = False
    per_token_timesteps: bool = False
    mask_ratio: float = 0.0
    masked_token_t: float = 1.0
    aux_loss_weight: float = 0.0
    aux_masked_only: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_flow_lite: SelfFlowLiteConfig = field(default_factory=SelfFlowLiteConfig)


T = TypeVar("T")


def _update_dataclass(cls: type[T], values: dict[str, Any] | None) -> T:
    obj = cls()
    if not values:
        return obj
    allowed = {field.name for field in fields(cls)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"Unknown config key for {cls.__name__}: {key}")
        setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> Config:
    path = str(path)
    if is_gcs_uri(path):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError("Install google-cloud-storage to load gs:// configs.") from exc
        bucket_name, blob_name = split_gcs_uri(path)
        raw_text = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
        raw = yaml.safe_load(raw_text) or {}
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return Config(
        experiment=_update_dataclass(ExperimentConfig, raw.get("experiment")),
        data=_update_dataclass(DataConfig, raw.get("data")),
        model=_update_dataclass(ModelConfig, raw.get("model")),
        training=_update_dataclass(TrainingConfig, raw.get("training")),
        self_flow_lite=_update_dataclass(SelfFlowLiteConfig, raw.get("self_flow_lite")),
    )


def to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)
