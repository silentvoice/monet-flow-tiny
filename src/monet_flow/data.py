from __future__ import annotations

import bisect
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from monet_flow.gcs import download_prefix, is_gcs_uri


def resolve_data_dir(data_dir: str | Path) -> Path:
    data_dir = str(data_dir)
    if not is_gcs_uri(data_dir):
        return Path(data_dir)
    cache_root = Path(os.environ.get("MONET_FLOW_GCS_CACHE", "data/cache/gcs"))
    safe_name = data_dir.replace("gs://", "").replace("/", "__")
    return download_prefix(data_dir, cache_root / safe_name)


class LatentShardDataset(Dataset):
    """Lazy dataset for shards produced by scripts/prepare_monet_subset.py."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = resolve_data_dir(data_dir)
        self.entries = self._read_manifest()
        if not self.entries:
            raise FileNotFoundError(
                f"No MONET latent shards found in {self.data_dir}. "
                "Run scripts/create_toy_subset.py or scripts/prepare_monet_subset.py first."
            )
        self.cumulative: list[int] = []
        total = 0
        for entry in self.entries:
            total += int(entry["num_samples"])
            self.cumulative.append(total)
        self._cache_path: Path | None = None
        self._cache: dict[str, Any] | None = None

    def _read_manifest(self) -> list[dict[str, Any]]:
        manifest = self.data_dir / "manifest.jsonl"
        if manifest.exists():
            entries = []
            with manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    row["path"] = str(self.data_dir / row["path"])
                    entries.append(row)
            return entries

        entries = []
        for shard in sorted(self.data_dir.glob("shard-*.pt")):
            payload = torch.load(shard, map_location="cpu")
            entries.append({"path": str(shard), "num_samples": len(payload["latents"])})
        return entries

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _load_shard(self, path: Path) -> dict[str, Any]:
        if self._cache_path != path:
            self._cache = torch.load(path, map_location="cpu")
            self._cache_path = path
        assert self._cache is not None
        return self._cache

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = len(self) + index
        shard_index = bisect.bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - shard_start
        path = Path(self.entries[shard_index]["path"])
        shard = self._load_shard(path)
        item = {
            "latents": shard["latents"][local_index].float(),
            "text_embeds": shard["text_embeds"][local_index].float(),
        }
        if "captions" in shard:
            item["caption"] = shard["captions"][local_index]
        if "ids" in shard:
            item["id"] = shard["ids"][local_index]
        return item


def build_dataloader(
    data_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = LatentShardDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
