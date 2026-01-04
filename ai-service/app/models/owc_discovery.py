"""OWCModelDiscovery - Discover model files on OWC external drive.

Sprint 15 (January 3, 2026): Part of OWC Model Evaluation Automation.

This module provides a standalone model discovery service for the OWC drive.
Unlike OWCModelImportDaemon which discovers AND imports models, this class
focuses purely on discovery and integrates with EvaluationStatusTracker.

Key features:
- Discovers .pth files across multiple directories on OWC
- Extracts board_type, num_players, architecture from filenames
- Computes SHA256 hashes for deduplication
- Registers discovered models with EvaluationStatusTracker
- Supports both local and remote (SSH) access modes

Environment Variables:
    OWC_HOST: OWC host (default: mac-studio)
    OWC_USER: SSH user for OWC host
    OWC_BASE_PATH: OWC mount path (default: /Volumes/RingRift-Data)
    OWC_SSH_KEY: Path to SSH key (default: ~/.ssh/id_ed25519)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.ssh import SSHClient

logger = logging.getLogger(__name__)

__all__ = [
    "OWCModelDiscovery",
    "OWCDiscoveryConfig",
    "DiscoveredModel",
    "DiscoverySummary",
    "get_owc_discovery",
]


# ============================================================================
# Configuration
# ============================================================================

OWC_HOST = os.getenv("OWC_HOST", "mac-studio")
OWC_USER = os.getenv("OWC_USER", "armand")
OWC_BASE_PATH = os.getenv("OWC_BASE_PATH", "/Volumes/RingRift-Data")
OWC_SSH_KEY = os.getenv("OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

# Model directories to scan on OWC drive
OWC_MODEL_PATHS = [
    "models/archived",
    "models/training_runs",
    "models/checkpoints",
    "selfplay_repository/models",
    "training_data/models",
    "models",  # Root models directory
]

# Minimum model file size (1MB) - smaller files are likely corrupt
MIN_MODEL_SIZE_BYTES = 1_000_000

# Pattern to extract board_type, num_players, and version from model filenames
MODEL_NAME_PATTERN = re.compile(
    r"(?:canonical_)?(?:ringrift_)?(?:best_)?"
    r"(hex8|hexagonal|square8|square19)"
    r"[_-]?"
    r"([234])p"
    r"(?:[_-](v\d+\w*))?"
    r"(?:[_-](\d{8}))?"
    r"\.pth$",
    re.IGNORECASE,
)


def _is_running_on_owc_host(owc_host: str) -> bool:
    """Check if we're running on the OWC host itself."""
    hostname = socket.gethostname().lower()
    owc_host_lower = owc_host.lower()

    local_patterns = [
        owc_host_lower,
        f"{owc_host_lower}.local",
        owc_host_lower.replace("-", ""),
    ]

    hostname_normalized = hostname.replace("-", "").replace(".", "").replace("_", "")

    for pattern in local_patterns:
        pattern_normalized = pattern.replace("-", "").replace(".", "").replace("_", "")
        if hostname_normalized.startswith(pattern_normalized):
            return True

    if owc_host_lower in ("localhost", "127.0.0.1", "::1"):
        return True

    return False


@dataclass
class OWCDiscoveryConfig:
    """Configuration for OWC Model Discovery."""

    # OWC connection
    owc_host: str = OWC_HOST
    owc_user: str = OWC_USER
    owc_base_path: str = OWC_BASE_PATH
    owc_ssh_key: str = OWC_SSH_KEY

    # Discovery settings
    model_paths: list[str] = field(default_factory=lambda: list(OWC_MODEL_PATHS))
    min_model_size_bytes: int = MIN_MODEL_SIZE_BYTES

    # Timeouts
    ssh_timeout: int = 60
    hash_timeout: int = 120  # Timeout for computing SHA256 hashes

    # Options
    compute_hashes: bool = True  # Compute SHA256 for each model
    register_with_tracker: bool = True  # Auto-register with EvaluationStatusTracker

    @classmethod
    def from_env(cls) -> "OWCDiscoveryConfig":
        """Load configuration from environment."""
        return cls(
            owc_host=os.getenv("OWC_HOST", OWC_HOST),
            owc_user=os.getenv("OWC_USER", OWC_USER),
            owc_base_path=os.getenv("OWC_BASE_PATH", OWC_BASE_PATH),
            owc_ssh_key=os.getenv("OWC_SSH_KEY", OWC_SSH_KEY),
            compute_hashes=os.getenv("RINGRIFT_OWC_COMPUTE_HASHES", "true").lower() == "true",
        )


@dataclass
class DiscoveredModel:
    """Model discovered on OWC drive.

    Extended from OWCModelInfo with additional fields for evaluation tracking.
    """

    path: str  # Relative path on OWC
    file_name: str  # Just the filename
    board_type: str | None  # Extracted from filename
    num_players: int | None  # Extracted from filename
    architecture_version: str | None  # Extracted from filename (v2, v4, v5-heavy, etc.)
    file_size: int  # File size in bytes
    sha256: str | None = None  # Content hash for deduplication
    modified_at: float | None = None  # File modification time (Unix timestamp)
    source: str = "owc"  # Source identifier

    @property
    def config_key(self) -> str | None:
        """Get config key if board_type and num_players are known."""
        if self.board_type and self.num_players:
            return f"{self.board_type}_{self.num_players}p"
        return None

    @property
    def is_canonical(self) -> bool:
        """Check if this is a canonical model."""
        return self.file_name.startswith("canonical_")

    @property
    def is_best(self) -> bool:
        """Check if this is a 'best' model."""
        return "best" in self.file_name.lower()

    @property
    def full_path(self) -> str:
        """Get full path on OWC."""
        return self.path


@dataclass
class DiscoverySummary:
    """Summary of a discovery operation."""

    total_models: int = 0
    models_by_config: dict[str, int] = field(default_factory=dict)
    models_by_architecture: dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    canonical_count: int = 0
    with_hash_count: int = 0
    discovery_duration: float = 0.0
    errors: list[str] = field(default_factory=list)


# ============================================================================
# Discovery Implementation
# ============================================================================


def _extract_model_info(file_path: str, file_size: int) -> DiscoveredModel | None:
    """Extract model information from file path.

    Args:
        file_path: Path to model file (relative or absolute)
        file_size: Size of the file in bytes

    Returns:
        DiscoveredModel if parsing successful, None for invalid files
    """
    file_name = Path(file_path).name

    # Skip obvious non-model files
    if not file_name.endswith(".pth"):
        return None

    # Skip temp/partial files
    if file_name.startswith(".") or file_name.startswith("_tmp"):
        return None

    match = MODEL_NAME_PATTERN.search(file_name)
    if match:
        board_type = match.group(1).lower()
        num_players = int(match.group(2))
        version = match.group(3)

        return DiscoveredModel(
            path=file_path,
            file_name=file_name,
            board_type=board_type,
            num_players=num_players,
            architecture_version=version,
            file_size=file_size,
        )

    # Fallback: try to extract at least partial info
    file_name_lower = file_name.lower()

    board_type = None
    for bt in ["hexagonal", "hex8", "square19", "square8"]:
        if bt in file_name_lower:
            board_type = bt
            break

    num_players = None
    for np in [4, 3, 2]:
        if f"{np}p" in file_name_lower:
            num_players = np
            break

    return DiscoveredModel(
        path=file_path,
        file_name=file_name,
        board_type=board_type,
        num_players=num_players,
        architecture_version=None,
        file_size=file_size,
    )


class OWCModelDiscovery:
    """Discover model files on OWC external drive.

    This class provides model discovery functionality that can be used
    independently of the import daemon. It integrates with EvaluationStatusTracker
    to register discovered models for backlog evaluation.

    Usage:
        discovery = OWCModelDiscovery()
        models = await discovery.discover_all_models()
        summary = await discovery.get_discovery_summary()
    """

    _instance: "OWCModelDiscovery | None" = None

    def __init__(self, config: OWCDiscoveryConfig | None = None):
        """Initialize OWC Model Discovery.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        self._config = config or OWCDiscoveryConfig.from_env()
        self._is_local = _is_running_on_owc_host(self._config.owc_host)

        # SSH client for remote operations (lazy-loaded)
        self._ssh_client: "SSHClient | None" = None

        # Cache of discovered models (path -> DiscoveredModel)
        self._model_cache: dict[str, DiscoveredModel] = {}
        self._last_discovery_time: float = 0.0

        if self._is_local:
            logger.info(
                f"[OWCDiscovery] Running on OWC host '{self._config.owc_host}', "
                f"using local file access"
            )

    @classmethod
    def get_instance(cls, config: OWCDiscoveryConfig | None = None) -> "OWCModelDiscovery":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    @property
    def config(self) -> OWCDiscoveryConfig:
        """Get discovery configuration."""
        return self._config

    @property
    def is_local(self) -> bool:
        """Check if running on OWC host."""
        return self._is_local

    # =========================================================================
    # SSH/Command Operations
    # =========================================================================

    def _get_ssh_client(self) -> "SSHClient":
        """Get or create SSH client for remote operations."""
        if self._ssh_client is None:
            from app.core.ssh import SSHClient, SSHConfig

            self._ssh_client = SSHClient(
                SSHConfig(
                    host=self._config.owc_host,
                    user=self._config.owc_user,
                    key_path=self._config.owc_ssh_key,
                    connect_timeout=10,
                    command_timeout=self._config.ssh_timeout,
                )
            )
        return self._ssh_client

    async def _run_command(self, command: str) -> tuple[bool, str]:
        """Run command on OWC host (locally or via SSH).

        Args:
            command: Shell command to run

        Returns:
            Tuple of (success, output_or_error)
        """
        if self._is_local:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self._config.ssh_timeout,
                )
                if result.returncode == 0:
                    return True, result.stdout.strip()
                else:
                    return False, result.stderr.strip() or "Command failed"
            except subprocess.TimeoutExpired:
                return False, "Command timed out"
            except OSError as e:
                return False, str(e)
        else:
            ssh_client = self._get_ssh_client()
            result = await ssh_client.run_async(
                command, timeout=self._config.ssh_timeout
            )
            if result.success:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip() or result.error or "Unknown error"

    async def check_available(self) -> bool:
        """Check if OWC drive is accessible.

        Returns:
            True if OWC is available, False otherwise
        """
        if self._is_local:
            owc_path = Path(self._config.owc_base_path)
            return owc_path.exists() and owc_path.is_dir()

        success, _ = await self._run_command(
            f"ls -d '{self._config.owc_base_path}' 2>/dev/null"
        )
        return success

    # =========================================================================
    # Discovery Operations
    # =========================================================================

    async def discover_all_models(
        self,
        force_refresh: bool = False,
        include_hashes: bool | None = None,
    ) -> list[DiscoveredModel]:
        """Find all model files on OWC drive.

        Args:
            force_refresh: Force re-discovery even if cache exists
            include_hashes: Override config for hash computation

        Returns:
            List of DiscoveredModel for all discovered models
        """
        import time

        start_time = time.time()

        # Check cache validity (5 minute TTL)
        if not force_refresh and self._model_cache:
            cache_age = time.time() - self._last_discovery_time
            if cache_age < 300:  # 5 minutes
                logger.debug(
                    f"[OWCDiscovery] Using cached results ({len(self._model_cache)} models, "
                    f"age: {cache_age:.0f}s)"
                )
                return list(self._model_cache.values())

        models: list[DiscoveredModel] = []
        base_path = self._config.owc_base_path

        # Build find command for all model directories
        search_paths = " ".join(
            f"'{base_path}/{p}'" for p in self._config.model_paths
        )

        min_size_k = self._config.min_model_size_bytes // 1024

        # Detect platform for correct stat format
        # macOS uses BSD stat (-f), Linux uses GNU stat (-c)
        # Use subshell to try GNU first, fallback to BSD if output is empty
        find_cmd = (
            f"output=$(find {search_paths} -name '*.pth' -type f "
            f"-size +{min_size_k}k "
            f"-exec stat -c '%s %Y %n' {{}} \\; 2>/dev/null); "
            f"if [ -n \"$output\" ]; then echo \"$output\"; else "
            f"find {search_paths} -name '*.pth' -type f "
            f"-size +{min_size_k}k "
            f"-exec stat -f '%z %m %N' {{}} \\; 2>/dev/null; fi"
        )

        success, output = await self._run_command(find_cmd)

        if not success or not output.strip():
            logger.warning("[OWCDiscovery] No models found or discovery failed")
            return models

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse size, mtime, and path
            parts = line.split(" ", 2)
            if len(parts) < 2:
                continue

            try:
                file_size = int(parts[0])
                if len(parts) == 3:
                    modified_at = float(parts[1])
                    file_path = parts[2]
                else:
                    # Fallback if mtime not available
                    modified_at = None
                    file_path = parts[1]

                # Convert absolute path to relative
                if file_path.startswith(base_path):
                    rel_path = file_path[len(base_path) :].lstrip("/")
                else:
                    rel_path = file_path

                # Extract model info
                model_info = _extract_model_info(rel_path, file_size)
                if model_info:
                    model_info.modified_at = modified_at
                    models.append(model_info)

            except (ValueError, IndexError):
                continue

        logger.info(f"[OWCDiscovery] Discovered {len(models)} models on OWC")

        # Compute hashes if requested
        compute_hashes = include_hashes if include_hashes is not None else self._config.compute_hashes
        if compute_hashes and models:
            await self._compute_hashes(models)

        # Update cache
        self._model_cache = {m.path: m for m in models}
        self._last_discovery_time = time.time()

        duration = time.time() - start_time
        logger.info(
            f"[OWCDiscovery] Discovery completed in {duration:.1f}s: "
            f"{len(models)} models, {sum(m.sha256 is not None for m in models)} with hashes"
        )

        return models

    async def _compute_hashes(self, models: list[DiscoveredModel]) -> None:
        """Compute SHA256 hashes for models.

        Args:
            models: List of models to compute hashes for
        """
        base_path = self._config.owc_base_path

        # For local mode, compute hashes directly
        if self._is_local:
            for model in models:
                try:
                    full_path = Path(base_path) / model.path
                    if full_path.exists():
                        model.sha256 = await asyncio.to_thread(
                            self._compute_file_hash, full_path
                        )
                except OSError as e:
                    logger.debug(f"[OWCDiscovery] Failed to hash {model.file_name}: {e}")
            return

        # For remote mode, batch compute hashes via SSH
        # Process in batches of 10 to avoid command line length limits
        batch_size = 10
        for i in range(0, len(models), batch_size):
            batch = models[i : i + batch_size]
            paths = " ".join(f"'{base_path}/{m.path}'" for m in batch)

            hash_cmd = f"sha256sum {paths} 2>/dev/null || shasum -a 256 {paths} 2>/dev/null"
            success, output = await self._run_command(hash_cmd)

            if success and output:
                for line in output.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 2:
                        hash_value = parts[0]
                        file_path = " ".join(parts[1:])
                        # Match to model
                        for model in batch:
                            if model.path in file_path or model.file_name in file_path:
                                model.sha256 = hash_value
                                break

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal SHA256 hash string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def get_model_count(self) -> int:
        """Get total count of models on OWC.

        Returns:
            Number of models, or 0 if unavailable
        """
        base_path = self._config.owc_base_path
        search_paths = " ".join(
            f"'{base_path}/{p}'" for p in self._config.model_paths
        )

        count_cmd = (
            f"find {search_paths} -name '*.pth' -type f "
            f"-size +{self._config.min_model_size_bytes // 1024}k 2>/dev/null | wc -l"
        )

        success, output = await self._run_command(count_cmd)

        if success and output.strip().isdigit():
            return int(output.strip())
        return 0

    async def get_discovery_summary(
        self, force_refresh: bool = False
    ) -> DiscoverySummary:
        """Get summary statistics of discovered models.

        Args:
            force_refresh: Force re-discovery

        Returns:
            DiscoverySummary with statistics
        """
        import time

        start_time = time.time()
        models = await self.discover_all_models(force_refresh=force_refresh)
        duration = time.time() - start_time

        summary = DiscoverySummary(
            total_models=len(models),
            total_size_bytes=sum(m.file_size for m in models),
            canonical_count=sum(1 for m in models if m.is_canonical),
            with_hash_count=sum(1 for m in models if m.sha256),
            discovery_duration=duration,
        )

        # Count by config
        for model in models:
            if model.config_key:
                summary.models_by_config[model.config_key] = (
                    summary.models_by_config.get(model.config_key, 0) + 1
                )

            # Count by architecture
            arch = model.architecture_version or "unknown"
            summary.models_by_architecture[arch] = (
                summary.models_by_architecture.get(arch, 0) + 1
            )

        return summary

    # =========================================================================
    # Integration with EvaluationStatusTracker
    # =========================================================================

    async def register_with_tracker(
        self,
        models: list[DiscoveredModel] | None = None,
        force_refresh: bool = False,
    ) -> int:
        """Register discovered models with EvaluationStatusTracker.

        Args:
            models: Optional list of models. If None, discovers first.
            force_refresh: Force re-discovery

        Returns:
            Number of models registered (newly added only)
        """
        try:
            from app.training.evaluation_status import (
                get_evaluation_status_tracker,
                ModelSource,
            )
        except ImportError:
            logger.warning(
                "[OWCDiscovery] EvaluationStatusTracker not available, skipping registration"
            )
            return 0

        if models is None:
            models = await self.discover_all_models(force_refresh=force_refresh)

        if not models:
            return 0

        tracker = get_evaluation_status_tracker()
        registered = 0

        # Batch registration for efficiency
        batch = []
        for model in models:
            if model.config_key and model.sha256:
                batch.append({
                    "model_path": model.path,
                    "model_sha256": model.sha256,
                    "board_type": model.board_type,
                    "num_players": model.num_players,
                    "source": ModelSource.OWC,
                })

        if batch:
            registered = await asyncio.to_thread(
                tracker.register_models_batch, batch
            )

        logger.info(
            f"[OWCDiscovery] Registered {registered} models with EvaluationStatusTracker"
        )

        return registered

    async def get_unevaluated_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 100,
    ) -> list[DiscoveredModel]:
        """Get models that haven't been evaluated yet.

        Uses EvaluationStatusTracker to filter to unevaluated models.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            limit: Maximum number to return

        Returns:
            List of unevaluated DiscoveredModel
        """
        try:
            from app.training.evaluation_status import get_evaluation_status_tracker
        except ImportError:
            # Fall back to all discovered models
            models = await self.discover_all_models()
            result = []
            for m in models:
                if board_type and m.board_type != board_type:
                    continue
                if num_players and m.num_players != num_players:
                    continue
                result.append(m)
                if len(result) >= limit:
                    break
            return result

        tracker = get_evaluation_status_tracker()

        # Get unevaluated model paths from tracker
        unevaluated = tracker.get_unevaluated_models(
            board_type=board_type,
            num_players=num_players,
            limit=limit,
            source="owc",
        )

        # Match to our cache
        unevaluated_paths = {m.model_path for m in unevaluated}

        if not self._model_cache:
            await self.discover_all_models()

        return [
            m for m in self._model_cache.values()
            if m.path in unevaluated_paths
        ][:limit]


# ============================================================================
# Module-level helpers
# ============================================================================


def get_owc_discovery(config: OWCDiscoveryConfig | None = None) -> OWCModelDiscovery:
    """Get singleton OWCModelDiscovery instance.

    Args:
        config: Optional configuration

    Returns:
        OWCModelDiscovery instance
    """
    return OWCModelDiscovery.get_instance(config)
