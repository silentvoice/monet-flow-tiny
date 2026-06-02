from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np
import torch


T = TypeVar("T")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def append_jsonl(path: str | Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def cycle(loader: Iterable[T]) -> Iterator[T]:
    while True:
        for item in loader:
            yield item


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(f"Unsupported precision: {precision}")


def format_count(value: int | float) -> str:
    if value == 0:
        return "0"
    magnitude = int(math.log10(abs(value)) // 3)
    suffix = ["", "K", "M", "B", "T"][min(magnitude, 4)]
    scaled = value / (1000 ** magnitude)
    return f"{scaled:.2f}{suffix}"
