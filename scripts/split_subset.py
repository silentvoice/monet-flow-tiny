#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--val-shards", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    with (input_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) <= args.val_shards:
        raise ValueError("Not enough shards to split")

    val_rows = rows[-args.val_shards :]
    train_rows = rows[: -args.val_shards]

    for destination, selected_rows in [(train_dir, train_rows), (val_dir, val_rows)]:
        with (destination / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                source = input_dir / row["path"]
                target = destination / row["path"]
                shutil.copy2(source, target)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"train_shards={len(train_rows)} val_shards={len(val_rows)}")


if __name__ == "__main__":
    main()
