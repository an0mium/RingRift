# Storage Module

Storage abstraction layer for cloud and local file operations. Provides a unified interface for uploading, downloading, and managing artifacts across local filesystem, AWS S3, and Google Cloud Storage.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Storage Backends](#storage-backends)
   - [LocalStorage](#localstorage)
   - [S3Storage](#s3storage)
   - [GCSStorage](#gcsstorage)
4. [StorageBackend Interface](#storagebackend-interface)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Integration](#integration)

---

## Overview

The storage module provides a consistent API for file operations regardless of the underlying storage system.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Module                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   StorageBackend (ABC)                   │   │
│   │  • upload(local_path, remote_key)                       │   │
│   │  • download(remote_key, local_path)                     │   │
│   │  • list(prefix)                                          │   │
│   │  • exists(remote_key)                                    │   │
│   │  • delete(remote_key)                                    │   │
│   │  • download_if_newer(remote_key, local_path)            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │LocalStorage │    │  S3Storage  │    │ GCSStorage  │        │
│   │             │    │             │    │             │        │
│   │ Filesystem  │    │   AWS S3    │    │   GCS       │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Factory Functions                           │   │
│   │  • get_storage_backend(backend, bucket, prefix, ...)    │   │
│   │  • get_storage_from_uri("s3://bucket/prefix")           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

| File          | Lines | Description                     |
| ------------- | ----- | ------------------------------- |
| `backends.py` | ~453  | Storage backend implementations |
| `__init__.py` | ~26   | Public API exports              |

---

## Quick Start

```python
from app.storage import get_storage_backend

# Local storage (default)
storage = get_storage_backend("local", base_path="/data")

# AWS S3
storage = get_storage_backend("s3", bucket="my-bucket", prefix="ringrift-ai")

# Google Cloud Storage
storage = get_storage_backend("gcs", bucket="my-bucket", prefix="ringrift-ai")

# Upload a file
storage.upload("/local/model.pth", "models/v1/model.pth")

# Download a file
storage.download("models/v1/model.pth", "/local/model.pth")

# List files
files = storage.list("models/")

# Check existence
if storage.exists("models/v1/model.pth"):
    ...

# Delete
storage.delete("models/v1/old_model.pth")
```

---

## Storage Backends

### LocalStorage

Filesystem-based storage:

```python
from app.storage import LocalStorage

storage = LocalStorage(base_path="/data/ringrift")

# Upload (copies file)
storage.upload("/tmp/model.pth", "models/new_model.pth")
# Result: /data/ringrift/models/new_model.pth

# Download (copies file)
storage.download("models/new_model.pth", "/tmp/downloaded.pth")

# List files recursively
files = storage.list("models/")
# ["models/v1/model.pth", "models/v2/model.pth", ...]

# List with prefix pattern
files = storage.list("models/v1")
# ["models/v1/model.pth", "models/v1/config.json"]
```

Features:

- Preserves file metadata (`shutil.copy2`)
- Creates parent directories automatically
- Recursive listing with `rglob`

### S3Storage

AWS S3 storage:

```python
from app.storage import S3Storage

storage = S3Storage(
    bucket="ringrift-training",
    prefix="ai-service",
    region="us-west-2",  # Optional
)

# Upload to S3
storage.upload("/local/model.pth", "models/v1/model.pth")
# Result: s3://ringrift-training/ai-service/models/v1/model.pth

# Download from S3
storage.download("models/v1/model.pth", "/local/model.pth")

# List objects
files = storage.list("models/")
# ["models/v1/model.pth", "models/v2/model.pth"]
```

Requirements:

- `pip install boto3`
- AWS credentials configured (environment, ~/.aws/credentials, or IAM role)

### GCSStorage

Google Cloud Storage:

```python
from app.storage import GCSStorage

storage = GCSStorage(
    bucket="ringrift-training",
    prefix="ai-service",
)

# Upload to GCS
storage.upload("/local/model.pth", "models/v1/model.pth")
# Result: gs://ringrift-training/ai-service/models/v1/model.pth

# Download from GCS
storage.download("models/v1/model.pth", "/local/model.pth")

# List objects
files = storage.list("models/")
```

Requirements:

- `pip install google-cloud-storage`
- GCP credentials configured (GOOGLE_APPLICATION_CREDENTIALS or default credentials)

---

## StorageBackend Interface

All backends implement this interface:

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):

    @abstractmethod
    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Upload a local file to remote storage."""

    @abstractmethod
    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Download a file from remote storage."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List files with the given prefix."""

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if a file exists."""

    @abstractmethod
    def delete(self, remote_key: str) -> None:
        """Delete a file from storage."""

    def download_if_newer(self, remote_key: str, local_path: str | Path) -> bool:
        """Download file only if remote is newer than local.
        Returns True if downloaded, False if local is up-to-date."""
```

---

## Configuration

### Environment Variables

| Variable            | Description                  | Default              |
| ------------------- | ---------------------------- | -------------------- |
| `STORAGE_BACKEND`   | Backend type: local, s3, gcs | `local`              |
| `STORAGE_BUCKET`    | S3/GCS bucket name           | (required for cloud) |
| `STORAGE_PREFIX`    | Key prefix for cloud storage | `ringrift-ai`        |
| `STORAGE_BASE_PATH` | Base path for local storage  | `.`                  |

### Factory Functions

```python
from app.storage import get_storage_backend

# Uses environment variables
storage = get_storage_backend()

# Explicit configuration
storage = get_storage_backend(
    backend="s3",
    bucket="my-bucket",
    prefix="my-prefix",
)

# Local with custom path
storage = get_storage_backend(
    backend="local",
    base_path="/data/storage",
)
```

### URI-Based Factory

```python
from app.storage.backends import get_storage_from_uri

# Local storage
storage = get_storage_from_uri("/path/to/data")
storage = get_storage_from_uri("file:///path/to/data")

# S3 storage
storage = get_storage_from_uri("s3://my-bucket/prefix")

# GCS storage
storage = get_storage_from_uri("gs://my-bucket/prefix")
```

---

## Usage Examples

### Model Checkpoint Storage

```python
from app.storage import get_storage_backend
from pathlib import Path

storage = get_storage_backend()

def save_checkpoint(model, epoch: int, board_type: str):
    """Save model checkpoint to storage."""
    local_path = Path(f"/tmp/checkpoint_{epoch}.pth")
    torch.save(model.state_dict(), local_path)

    remote_key = f"checkpoints/{board_type}/epoch_{epoch:04d}.pth"
    storage.upload(local_path, remote_key)

    # Cleanup local temp file
    local_path.unlink()

def load_latest_checkpoint(board_type: str) -> dict:
    """Load the latest checkpoint."""
    # List all checkpoints
    checkpoints = storage.list(f"checkpoints/{board_type}/")

    if not checkpoints:
        return None

    # Get latest (sorted by name)
    latest = sorted(checkpoints)[-1]

    # Download to temp
    local_path = Path(f"/tmp/{Path(latest).name}")
    storage.download(latest, local_path)

    state_dict = torch.load(local_path)
    local_path.unlink()

    return state_dict
```

### Training Data Sync

```python
from app.storage import get_storage_backend

storage = get_storage_backend("s3", bucket="ringrift-data")

def sync_training_data(local_dir: str, remote_prefix: str):
    """Sync local training data to cloud storage."""
    local_path = Path(local_dir)

    for file_path in local_path.rglob("*.npz"):
        relative = file_path.relative_to(local_path)
        remote_key = f"{remote_prefix}/{relative}"

        if not storage.exists(remote_key):
            print(f"Uploading {file_path.name}...")
            storage.upload(file_path, remote_key)

def download_training_data(remote_prefix: str, local_dir: str):
    """Download training data from cloud."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    files = storage.list(remote_prefix)
    for remote_key in files:
        if not remote_key.endswith(".npz"):
            continue

        relative = remote_key[len(remote_prefix):].lstrip("/")
        local_file = local_path / relative

        if not local_file.exists():
            print(f"Downloading {relative}...")
            storage.download(remote_key, local_file)
```

### Multi-Backend Support

```python
import os
from app.storage import get_storage_backend

def get_artifact_storage():
    """Get storage based on environment."""
    # Development: local storage
    if os.getenv("RINGRIFT_ENV") == "development":
        return get_storage_backend("local", base_path="./artifacts")

    # Production: cloud storage
    backend = os.getenv("STORAGE_BACKEND", "s3")
    return get_storage_backend(backend)

# Usage
storage = get_artifact_storage()
storage.upload("model.pth", "models/latest.pth")
```

### Conditional Download

```python
from app.storage import get_storage_backend
from pathlib import Path

storage = get_storage_backend()

def ensure_model_exists(model_key: str, local_path: Path) -> Path:
    """Ensure model exists locally, downloading if needed."""
    if local_path.exists():
        # Use download_if_newer when implemented with metadata
        return local_path

    if storage.exists(model_key):
        storage.download(model_key, local_path)
        return local_path

    raise FileNotFoundError(f"Model not found: {model_key}")
```

---

## Integration

### With Model Loader

```python
from app.storage import get_storage_backend
from app.models.loader import ModelLoader

storage = get_storage_backend("s3", bucket="ringrift-models")

def download_production_model(board_type: str, num_players: int):
    """Download production model from cloud storage."""
    remote_key = f"production/{board_type}_{num_players}p/model.pth"
    local_path = Path(f"models/{board_type}_{num_players}p.pth")

    if not local_path.exists():
        storage.download(remote_key, local_path)

    loader = ModelLoader()
    return loader.load_nnue(board_type, num_players)
```

### With Training Pipeline

```python
from app.storage import get_storage_backend

storage = get_storage_backend()

class CheckpointManager:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.prefix = f"training_runs/{run_id}"

    def save(self, model, optimizer, epoch: int, metrics: dict):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }

        local_path = Path(f"/tmp/checkpoint_{epoch}.pth")
        torch.save(checkpoint, local_path)

        storage.upload(local_path, f"{self.prefix}/checkpoint_{epoch:04d}.pth")
        local_path.unlink()

    def load_latest(self):
        checkpoints = storage.list(f"{self.prefix}/checkpoint_")
        if not checkpoints:
            return None

        latest = sorted(checkpoints)[-1]
        local_path = Path("/tmp/latest_checkpoint.pth")
        storage.download(latest, local_path)

        return torch.load(local_path)
```

---

## Error Handling

```python
from app.storage import get_storage_backend

storage = get_storage_backend("s3", bucket="my-bucket")

try:
    storage.download("models/missing.pth", "/local/path.pth")
except FileNotFoundError:
    print("Model not found in storage")

try:
    storage.upload("/nonexistent/file.pth", "models/new.pth")
except FileNotFoundError:
    print("Local file not found")

# Check before download
if storage.exists("models/v1/model.pth"):
    storage.download("models/v1/model.pth", local_path)
else:
    print("Model not available")
```

---

## See Also

- `app/models/loader.py` - Model loading with storage integration
- `app/training/checkpointing.py` - Checkpoint management
- `scripts/sync_models.py` - Model synchronization utilities

---

_Last updated: December 2025_
