from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse


def is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Not a valid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_prefix(uri: str, local_dir: str | Path) -> Path:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("Install google-cloud-storage to read gs:// data in Python.") from exc

    bucket_name, prefix = split_gcs_uri(uri)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No blobs found under {uri}")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative = Path(blob.name).relative_to(prefix) if prefix else Path(blob.name)
        target = local_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == blob.size:
            continue
        blob.download_to_filename(target)
    return local_dir


def download_file(uri: str, local_path: str | Path) -> Path:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("Install google-cloud-storage to read gs:// files in Python.") from exc

    bucket_name, blob_name = split_gcs_uri(uri)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path


def list_files(uri: str) -> list[str]:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("Install google-cloud-storage to list gs:// files in Python.") from exc

    bucket_name, prefix = split_gcs_uri(uri)
    client = storage.Client()
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in client.list_blobs(bucket_name, prefix=prefix)
        if not blob.name.endswith("/")
    ]


def upload_directory(local_dir: str | Path, uri: str) -> None:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("Install google-cloud-storage to upload outputs to gs://.") from exc

    bucket_name, prefix = split_gcs_uri(uri)
    local_dir = Path(local_dir)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(local_dir)
        blob_name = str(Path(prefix) / relative) if prefix else str(relative)
        bucket.blob(blob_name).upload_from_filename(path)


def upload_file(local_path: str | Path, uri: str) -> None:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("Install google-cloud-storage to upload outputs to gs://.") from exc

    bucket_name, blob_name = split_gcs_uri(uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


def gcloud_cp(source: str | Path, destination: str | Path, recursive: bool = False) -> None:
    command = ["gcloud", "storage", "cp"]
    if recursive:
        command.append("-r")
    command.extend([str(source), str(destination)])
    subprocess.run(command, check=True)


def gcloud_rsync(source: str | Path, destination: str | Path) -> None:
    subprocess.run(["gcloud", "storage", "rsync", "-r", str(source), str(destination)], check=True)
