"""MinIO client for document synchronization."""

import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

import config


def get_client() -> Minio:
    """Create and return a MinIO client."""
    return Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    """Ensure the bucket exists, create if not."""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def sync_documents(client: Minio | None = None) -> list[str]:
    """
    Sync documents from MinIO bucket to local directory.

    Returns:
        List of downloaded file paths.
    """
    if client is None:
        client = get_client()

    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    try:
        objects = client.list_objects(config.MINIO_BUCKET, recursive=True)

        for obj in objects:
            if obj.object_name.endswith((".md", ".csv")):
                local_path = docs_dir / obj.object_name
                local_path.parent.mkdir(parents=True, exist_ok=True)

                client.fget_object(
                    config.MINIO_BUCKET,
                    obj.object_name,
                    str(local_path),
                )
                downloaded.append(str(local_path))

    except S3Error as e:
        print(f"MinIO error: {e}")
        raise

    return downloaded


def list_remote_files(client: Minio | None = None) -> list[str]:
    """List all document files in the MinIO bucket."""
    if client is None:
        client = get_client()

    files = []
    try:
        objects = client.list_objects(config.MINIO_BUCKET, recursive=True)
        for obj in objects:
            if obj.object_name.endswith((".md", ".csv")):
                files.append(obj.object_name)
    except S3Error as e:
        print(f"MinIO error: {e}")
        raise

    return files
