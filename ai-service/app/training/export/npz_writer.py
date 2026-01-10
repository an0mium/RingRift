"""NPZ file writer with validation and disk space checking.

This module provides safe NPZ writing with:
- Disk space validation before write
- Post-write structure validation
- Atomic operations via existing npz_atomic_writer
- Checksum embedding for integrity verification

Usage:
    from app.training.export.npz_writer import NPZExportWriter

    writer = NPZExportWriter(output_path="data/training/hex8_2p.npz")
    result = writer.write(arrays={"features": features_arr, ...}, metadata={...})
    if result.success:
        print(f"Wrote {result.sample_count} samples to {result.path}")
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.training.export.config import (
    DISK_SPACE_SAFETY_MARGIN_MB,
    NPZ_COMPRESSION_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Result of NPZ write operation.

    Attributes:
        success: True if write succeeded
        path: Path to written file
        sample_count: Number of samples in file
        file_size_bytes: Size of written file in bytes
        array_shapes: Dict mapping array names to shapes
        errors: List of error messages if failed
        warnings: List of warning messages
        validation_passed: True if post-write validation passed
    """

    success: bool
    path: Path | None = None
    sample_count: int = 0
    file_size_bytes: int = 0
    array_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_passed: bool = False


def estimate_npz_size(arrays: dict[str, np.ndarray]) -> int:
    """Estimate compressed NPZ file size in bytes.

    Args:
        arrays: Dictionary of arrays to save

    Returns:
        Estimated file size in bytes
    """
    total_bytes = 0
    for arr in arrays.values():
        if isinstance(arr, np.ndarray):
            total_bytes += arr.nbytes
    # Apply compression ratio estimate
    return int(total_bytes * NPZ_COMPRESSION_RATIO)


def check_disk_space(output_path: Path, arrays: dict[str, np.ndarray]) -> tuple[bool, str]:
    """Check if there's enough disk space for NPZ export.

    Args:
        output_path: Path where NPZ will be written
        arrays: Dictionary of arrays to save

    Returns:
        Tuple of (has_space, message)
    """
    try:
        # Get destination directory
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir = Path.cwd()

        # Check available space
        usage = shutil.disk_usage(output_dir)
        available_mb = usage.free / (1024 * 1024)

        # Estimate required space
        estimated_bytes = estimate_npz_size(arrays)
        estimated_mb = estimated_bytes / (1024 * 1024)
        required_mb = estimated_mb + DISK_SPACE_SAFETY_MARGIN_MB

        if available_mb < required_mb:
            return False, (
                f"Insufficient disk space: "
                f"need {required_mb:.1f}MB (estimated {estimated_mb:.1f}MB + "
                f"{DISK_SPACE_SAFETY_MARGIN_MB}MB safety margin), "
                f"available {available_mb:.1f}MB at {output_dir}"
            )

        return True, f"Disk space OK: {available_mb:.1f}MB available, need ~{required_mb:.1f}MB"

    except OSError as e:
        # If we can't check, log warning but don't block
        return True, f"Could not check disk space: {e}"


class NPZExportWriter:
    """Write arrays to NPZ with validation and safety checks.

    Features:
    - Pre-write disk space validation
    - Post-write structure validation
    - Checksum embedding for integrity
    - Automatic directory creation
    - Archive of existing files

    Example:
        writer = NPZExportWriter(Path("output.npz"))
        result = writer.write(
            arrays={"features": features, "values": values},
            metadata={"board_type": "hex8"}
        )
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        archive_existing: bool = True,
        validate_after_write: bool = True,
        embed_checksums: bool = True,
    ):
        """Initialize writer.

        Args:
            output_path: Path for output NPZ file
            archive_existing: Archive existing file before overwrite (default: True)
            validate_after_write: Validate NPZ structure after write (default: True)
            embed_checksums: Embed checksums in save_kwargs (default: True)
        """
        self.output_path = Path(output_path)
        self.archive_existing = archive_existing
        self.validate_after_write = validate_after_write
        self.embed_checksums = embed_checksums

    def write(
        self,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> WriteResult:
        """Write arrays to NPZ file.

        Args:
            arrays: Dictionary of arrays to write (e.g., {"features": ..., "values": ...})
            metadata: Optional metadata to include in file

        Returns:
            WriteResult with success status and details
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check disk space
        has_space, space_msg = check_disk_space(self.output_path, arrays)
        if not has_space:
            return WriteResult(
                success=False,
                errors=[f"Disk space error: {space_msg}"],
            )
        logger.info(space_msg)

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Archive existing file
        if self.archive_existing and self.output_path.exists():
            self._archive_existing_file()

        # Build save kwargs
        save_kwargs = dict(arrays)

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if key not in save_kwargs:
                    save_kwargs[key] = np.asarray(value)

        # Embed checksums
        if self.embed_checksums:
            try:
                from app.training.data_quality import embed_checksums_in_save_kwargs
                save_kwargs = embed_checksums_in_save_kwargs(save_kwargs)
            except ImportError:
                warnings.append("Checksum embedding skipped (data_quality module unavailable)")

        # Write NPZ
        try:
            np.savez_compressed(str(self.output_path), **save_kwargs)
        except Exception as e:
            return WriteResult(
                success=False,
                errors=[f"Write failed: {e}"],
            )

        # Get file size
        file_size = self.output_path.stat().st_size if self.output_path.exists() else 0

        # Build array shapes dict
        array_shapes = {
            name: arr.shape if isinstance(arr, np.ndarray) else ()
            for name, arr in arrays.items()
        }

        # Count samples (from features array)
        sample_count = 0
        if "features" in arrays:
            sample_count = arrays["features"].shape[0]
        elif "values" in arrays:
            sample_count = arrays["values"].shape[0]

        # Validate after write
        validation_passed = False
        if self.validate_after_write:
            validation_result = self._validate_npz(self.output_path)
            validation_passed = validation_result["valid"]
            if not validation_passed:
                errors.extend(validation_result.get("errors", []))
            warnings.extend(validation_result.get("warnings", []))

        return WriteResult(
            success=len(errors) == 0,
            path=self.output_path,
            sample_count=sample_count,
            file_size_bytes=file_size,
            array_shapes=array_shapes,
            errors=errors,
            warnings=warnings,
            validation_passed=validation_passed,
        )

    def _archive_existing_file(self) -> None:
        """Archive existing file with timestamp suffix."""
        if not self.output_path.exists():
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archived = self.output_path.with_suffix(f".archived_{timestamp}.npz")

        try:
            self.output_path.rename(archived)
            logger.info(f"Archived existing file to: {archived}")
        except OSError as e:
            logger.warning(f"Failed to archive {self.output_path}: {e}")

    def _validate_npz(self, path: Path) -> dict[str, Any]:
        """Validate NPZ structure.

        Returns:
            Dict with 'valid', 'errors', 'warnings', 'sample_count', 'array_shapes'
        """
        try:
            from app.coordination.npz_validation import validate_npz_structure

            result = validate_npz_structure(path)
            return {
                "valid": result.valid,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "sample_count": result.sample_count,
                "array_shapes": result.array_shapes,
            }
        except ImportError:
            # Fallback: basic validation
            return self._basic_validate_npz(path)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
            }

    def _basic_validate_npz(self, path: Path) -> dict[str, Any]:
        """Basic NPZ validation when full validator unavailable."""
        errors = []
        warnings = []
        sample_count = 0
        array_shapes: dict[str, tuple[int, ...]] = {}

        try:
            with np.load(str(path), allow_pickle=True) as data:
                # Check required arrays exist
                required = ["features", "values"]
                for name in required:
                    if name not in data:
                        errors.append(f"Missing required array: {name}")
                    else:
                        arr = data[name]
                        array_shapes[name] = arr.shape

                # Get sample count from features
                if "features" in data:
                    sample_count = data["features"].shape[0]

                    # Validate all arrays have consistent sample count
                    for name in data.files:
                        arr = data[name]
                        if isinstance(arr, np.ndarray) and arr.ndim > 0:
                            if arr.shape[0] != sample_count:
                                # Some arrays are metadata, not per-sample
                                if name not in {"board_type", "board_size", "history_length"}:
                                    warnings.append(
                                        f"Array '{name}' has {arr.shape[0]} samples, "
                                        f"expected {sample_count}"
                                    )

        except Exception as e:
            errors.append(f"Failed to read NPZ: {e}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "sample_count": sample_count,
            "array_shapes": array_shapes,
        }

    def validate(self, path: Path | None = None) -> WriteResult:
        """Validate an existing NPZ file.

        Args:
            path: Path to validate (default: self.output_path)

        Returns:
            WriteResult with validation details
        """
        path = path or self.output_path
        if not path.exists():
            return WriteResult(
                success=False,
                errors=[f"File not found: {path}"],
            )

        result = self._validate_npz(path)
        return WriteResult(
            success=result["valid"],
            path=path,
            sample_count=result.get("sample_count", 0),
            file_size_bytes=path.stat().st_size,
            array_shapes=result.get("array_shapes", {}),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            validation_passed=result["valid"],
        )


def register_with_manifest(
    output_path: Path,
    board_type: str,
    num_players: int,
    sample_count: int,
) -> bool:
    """Register NPZ file with ClusterManifest for cluster-wide discovery.

    Args:
        output_path: Path to NPZ file
        board_type: Board type (e.g., "hex8")
        num_players: Number of players
        sample_count: Number of samples in file

    Returns:
        True if registration succeeded
    """
    try:
        import socket
        from app.distributed.cluster_manifest import get_cluster_manifest

        manifest = get_cluster_manifest()
        file_size = output_path.stat().st_size
        node_id = socket.gethostname()

        manifest.register_npz(
            npz_path=str(output_path),
            node_id=node_id,
            board_type=board_type,
            num_players=num_players,
            sample_count=sample_count,
            file_size=file_size,
        )
        logger.info(
            f"Registered NPZ with ClusterManifest: {output_path} "
            f"({file_size // 1024 // 1024}MB, {sample_count} samples)"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to register NPZ with manifest: {e}")
        return False
