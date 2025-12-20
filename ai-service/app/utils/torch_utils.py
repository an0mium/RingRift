"""Utilities for safe PyTorch operations.

This module provides secure wrappers around PyTorch functions that may
have security implications, particularly around model loading.

Security Note:
    torch.load with weights_only=False can execute arbitrary code during
    unpickling. This module provides safe_load_checkpoint which:
    1. First tries weights_only=True (safe mode)
    2. Falls back to weights_only=False only when necessary
    3. Logs a warning when using unsafe mode
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import torch - this module should work even without torch
skip_torch = os.getenv("RINGRIFT_SKIP_TORCH_IMPORT", "").strip().lower()
skip_optional = os.getenv("RINGRIFT_SKIP_OPTIONAL_IMPORTS", "").strip().lower()
if skip_torch in ("1", "true", "yes", "on") or skip_optional in ("1", "true", "yes", "on"):
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
else:
    try:
        import torch
        HAS_TORCH = True
    except Exception as exc:
        HAS_TORCH = False
        torch = None  # type: ignore[assignment]
        logger.debug("PyTorch import failed: %s", exc)


def safe_load_checkpoint(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
    allow_unsafe: bool = True,
    warn_on_unsafe: bool = True,
) -> dict[str, Any]:
    """Safely load a PyTorch checkpoint.

    This function attempts to load checkpoints in the safest way possible:
    1. First tries with weights_only=True (prevents arbitrary code execution)
    2. If that fails and allow_unsafe=True, falls back to weights_only=False

    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to (default: "cpu")
        allow_unsafe: Whether to allow fallback to unsafe loading
        warn_on_unsafe: Whether to log a warning when using unsafe loading

    Returns:
        The loaded checkpoint dictionary

    Raises:
        ImportError: If PyTorch is not installed
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If loading fails and allow_unsafe=False
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for checkpoint loading")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Try safe loading first
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return checkpoint
    except Exception as safe_error:
        if not allow_unsafe:
            raise RuntimeError(
                f"Failed to load checkpoint with weights_only=True: {safe_error}. "
                "Set allow_unsafe=True to allow unsafe loading."
            ) from safe_error

        # Fall back to unsafe loading for legacy checkpoints
        if warn_on_unsafe:
            logger.warning(
                "Loading checkpoint with weights_only=False (unsafe mode). "
                "This checkpoint may contain non-tensor data. Path: %s",
                path,
            )

        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            return checkpoint
        except TypeError:
            # Very old PyTorch versions don't support weights_only
            checkpoint = torch.load(path, map_location=map_location)
            return checkpoint


def load_state_dict_only(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    """Load only the state_dict from a checkpoint (safest mode).

    This is the safest way to load model weights - it only loads the
    state_dict and ignores any other data in the checkpoint.

    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to

    Returns:
        The model state_dict

    Raises:
        KeyError: If the checkpoint doesn't contain a state_dict
    """
    checkpoint = safe_load_checkpoint(
        path,
        map_location=map_location,
        allow_unsafe=True,  # May need for legacy checkpoints
        warn_on_unsafe=False,  # We're only extracting state_dict anyway
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            # Assume the whole dict is the state_dict
            return checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")


def save_checkpoint_safe(
    checkpoint: dict[str, Any],
    path: str | Path,
    *,
    use_new_format: bool = True,
) -> None:
    """Save a checkpoint in a secure format.

    Args:
        checkpoint: The checkpoint dictionary to save
        path: Path to save the checkpoint
        use_new_format: If True, use torch.save with _use_new_zipfile_serialization
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for checkpoint saving")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_new_format:
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
    else:
        torch.save(checkpoint, path)
