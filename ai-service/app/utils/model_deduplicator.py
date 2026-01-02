"""Model Deduplicator - Identify unique models by SHA256 checksum.

This module provides deduplication for large model archives, ensuring that
identical models (regardless of filename or location) are evaluated only once
in tournaments.

Usage:
    from app.utils.model_deduplicator import ModelDeduplicator, UniqueModel

    deduplicator = ModelDeduplicator()
    unique_models = await deduplicator.scan_directory(Path("/Volumes/RingRift-Data"))
    print(f"Found {len(unique_models)} unique models")

    # Group by config
    for config_key, models in deduplicator.group_by_config(unique_models).items():
        print(f"  {config_key}: {len(models)} models")

January 2, 2026: Created for massive tournament support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache database location
DEFAULT_CACHE_DB = Path("data/model_checksums.db")

# Batch size for parallel checksum computation
CHECKSUM_BATCH_SIZE = 50

# Board type aliases (for filename parsing)
BOARD_ALIASES = {
    "sq8": "square8",
    "sq19": "square19",
    "hex8": "hex8",
    "hex": "hexagonal",
    "hexagonal": "hexagonal",
    "square8": "square8",
    "square19": "square19",
}


@dataclass
class UniqueModel:
    """Represents a deduplicated model."""

    sha256: str
    canonical_path: Path  # First discovered path
    all_paths: list[Path] = field(default_factory=list)  # All locations
    board_type: str = "unknown"
    num_players: int = 0
    model_family: str = "unknown"
    file_size: int = 0
    architecture: str = "unknown"
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sha256": self.sha256,
            "canonical_path": str(self.canonical_path),
            "all_paths": [str(p) for p in self.all_paths],
            "board_type": self.board_type,
            "num_players": self.num_players,
            "model_family": self.model_family,
            "file_size": self.file_size,
            "architecture": self.architecture,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniqueModel:
        """Create from dictionary."""
        return cls(
            sha256=data["sha256"],
            canonical_path=Path(data["canonical_path"]),
            all_paths=[Path(p) for p in data.get("all_paths", [])],
            board_type=data.get("board_type", "unknown"),
            num_players=data.get("num_players", 0),
            model_family=data.get("model_family", "unknown"),
            file_size=data.get("file_size", 0),
            architecture=data.get("architecture", "unknown"),
            first_seen=data.get("first_seen", time.time()),
        )


class ModelDeduplicator:
    """Deduplicate models by SHA256 checksum.

    Uses an SQLite cache to avoid recomputing checksums for files that
    haven't changed (based on file size and mtime).
    """

    def __init__(self, cache_db: Path | None = None):
        """Initialize deduplicator with optional cache database."""
        self._cache_db = cache_db or DEFAULT_CACHE_DB
        self._checksums: dict[str, UniqueModel] = {}
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initialize the checksum cache database."""
        self._cache_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_checksums (
                    sha256 TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL,
                    board_type TEXT,
                    num_players INTEGER,
                    model_family TEXT,
                    file_size INTEGER,
                    architecture TEXT,
                    first_seen REAL,
                    all_paths TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    file_size INTEGER,
                    mtime REAL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_config "
                "ON model_checksums(board_type, num_players)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_family "
                "ON model_checksums(model_family)"
            )
            conn.commit()

    def _get_cached_checksum(
        self, file_path: Path, file_size: int, mtime: float
    ) -> str | None:
        """Get cached checksum if file hasn't changed."""
        with sqlite3.connect(self._cache_db) as conn:
            cursor = conn.execute(
                "SELECT sha256 FROM file_cache "
                "WHERE file_path = ? AND file_size = ? AND mtime = ?",
                (str(file_path), file_size, mtime),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def _cache_checksum(
        self, file_path: Path, sha256: str, file_size: int, mtime: float
    ) -> None:
        """Cache a computed checksum."""
        with sqlite3.connect(self._cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO file_cache "
                "(file_path, sha256, file_size, mtime) VALUES (?, ?, ?, ?)",
                (str(file_path), sha256, file_size, mtime),
            )
            conn.commit()

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _compute_checksum_async(self, file_path: Path) -> tuple[Path, str, int]:
        """Compute checksum asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        stat = file_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime

        # Check cache first
        cached = self._get_cached_checksum(file_path, file_size, mtime)
        if cached:
            return file_path, cached, file_size

        # Compute in thread pool
        sha256 = await loop.run_in_executor(None, self._compute_sha256, file_path)

        # Cache result
        self._cache_checksum(file_path, sha256, file_size, mtime)

        return file_path, sha256, file_size

    def extract_config_from_filename(
        self, path: Path
    ) -> tuple[str, int, str, str]:
        """Extract board_type, num_players, family, architecture from filename.

        Returns:
            Tuple of (board_type, num_players, model_family, architecture)
        """
        name = path.stem.lower()

        # Default values
        board_type = "unknown"
        num_players = 0
        model_family = "unknown"
        architecture = "v2"  # Default architecture

        # Extract player count (e.g., "2p", "3p", "4p")
        player_match = re.search(r"(\d)p", name)
        if player_match:
            num_players = int(player_match.group(1))

        # Extract board type
        for alias, canonical in BOARD_ALIASES.items():
            if alias in name:
                board_type = canonical
                break

        # Extract architecture version
        if "v5_heavy_large" in name or "v5-heavy-large" in name:
            architecture = "v5-heavy-large"
        elif "v5_heavy" in name or "v5-heavy" in name or "v5heavy" in name:
            architecture = "v5-heavy"
        elif "v5" in name:
            architecture = "v5"
        elif "v4" in name:
            architecture = "v4"
        elif "v2" in name:
            architecture = "v2"
        elif "nnue" in name:
            architecture = "nnue"

        # Extract model family
        # canonical_* models
        if name.startswith("canonical_"):
            match = re.match(r"canonical_([a-z0-9]+)_(\d)p", name)
            if match:
                model_family = "canonical"
            else:
                model_family = "canonical"
        # ringrift_* models
        elif name.startswith("ringrift_"):
            match = re.match(r"ringrift_(v\d+)_", name)
            if match:
                model_family = f"ringrift_{match.group(1)}"
            else:
                model_family = "ringrift"
        # policy_* models
        elif name.startswith("policy_"):
            model_family = "policy"
        # heuristic_* models
        elif name.startswith("heuristic_"):
            model_family = "heuristic"
        # *_nn_baseline* models
        elif "_nn_baseline" in name:
            model_family = "nn_baseline"
        # *_trained* models
        elif "_trained" in name:
            model_family = "trained"
        # *_cluster* models
        elif "_cluster" in name:
            model_family = "cluster"
        # *_improved* models
        elif "_improved" in name:
            model_family = "improved"
        # *_retrained* models
        elif "_retrained" in name:
            model_family = "retrained"
        # Checkpoint models
        elif name.startswith("checkpoint_"):
            model_family = "checkpoint"
        else:
            # Extract base name (remove timestamp if present)
            # Pattern: name_YYYYMMDD_HHMMSS or name_YYYYMMDD
            base = re.sub(r"_\d{8}(_\d{6})?$", "", name)
            if base != name:
                model_family = base
            else:
                model_family = name

        return board_type, num_players, model_family, architecture

    async def scan_directory(
        self,
        directory: Path,
        pattern: str = "**/*.pth",
        progress_callback: Any | None = None,
    ) -> list[UniqueModel]:
        """Scan directory for unique models.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for model files
            progress_callback: Optional callback(processed, total)

        Returns:
            List of unique models (deduplicated by SHA256)
        """
        logger.info(f"Scanning {directory} with pattern {pattern}...")

        # Find all model files
        files = list(directory.glob(pattern))
        total = len(files)
        logger.info(f"Found {total} .pth files")

        if total == 0:
            return []

        # Compute checksums in batches
        checksum_map: dict[str, list[tuple[Path, int]]] = {}  # sha256 -> [(path, size)]
        processed = 0

        for batch_start in range(0, total, CHECKSUM_BATCH_SIZE):
            batch_end = min(batch_start + CHECKSUM_BATCH_SIZE, total)
            batch = files[batch_start:batch_end]

            tasks = [self._compute_checksum_async(f) for f in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error computing checksum: {result}")
                    continue
                file_path, sha256, file_size = result
                if sha256 not in checksum_map:
                    checksum_map[sha256] = []
                checksum_map[sha256].append((file_path, file_size))

            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total)

            if processed % 500 == 0:
                logger.info(
                    f"Progress: {processed}/{total} files "
                    f"({len(checksum_map)} unique)"
                )

        # Create UniqueModel for each unique checksum
        unique_models = []
        for sha256, paths_and_sizes in checksum_map.items():
            # Sort by path to ensure consistent canonical path selection
            paths_and_sizes.sort(key=lambda x: str(x[0]))
            canonical_path, file_size = paths_and_sizes[0]
            all_paths = [p for p, _ in paths_and_sizes]

            # Extract metadata from canonical path
            board_type, num_players, model_family, architecture = (
                self.extract_config_from_filename(canonical_path)
            )

            unique_model = UniqueModel(
                sha256=sha256,
                canonical_path=canonical_path,
                all_paths=all_paths,
                board_type=board_type,
                num_players=num_players,
                model_family=model_family,
                file_size=file_size,
                architecture=architecture,
            )
            unique_models.append(unique_model)

            # Store in cache
            self._checksums[sha256] = unique_model

        # Persist to database
        self._save_to_db(unique_models)

        logger.info(
            f"Scan complete: {len(unique_models)} unique models "
            f"from {total} files (dedup ratio: {total / len(unique_models):.1f}x)"
        )

        return unique_models

    def _save_to_db(self, models: list[UniqueModel]) -> None:
        """Save unique models to the cache database."""
        with sqlite3.connect(self._cache_db) as conn:
            for model in models:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO model_checksums
                    (sha256, canonical_path, board_type, num_players,
                     model_family, file_size, architecture, first_seen, all_paths)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model.sha256,
                        str(model.canonical_path),
                        model.board_type,
                        model.num_players,
                        model.model_family,
                        model.file_size,
                        model.architecture,
                        model.first_seen,
                        json.dumps([str(p) for p in model.all_paths]),
                    ),
                )
            conn.commit()

    def load_from_db(self) -> list[UniqueModel]:
        """Load cached unique models from database."""
        models = []
        with sqlite3.connect(self._cache_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM model_checksums")
            for row in cursor:
                model = UniqueModel(
                    sha256=row["sha256"],
                    canonical_path=Path(row["canonical_path"]),
                    all_paths=[Path(p) for p in json.loads(row["all_paths"] or "[]")],
                    board_type=row["board_type"] or "unknown",
                    num_players=row["num_players"] or 0,
                    model_family=row["model_family"] or "unknown",
                    file_size=row["file_size"] or 0,
                    architecture=row["architecture"] or "v2",
                    first_seen=row["first_seen"] or time.time(),
                )
                models.append(model)
                self._checksums[model.sha256] = model
        return models

    def get_unique_models_for_config(
        self, board_type: str, num_players: int
    ) -> list[UniqueModel]:
        """Get deduplicated models for a specific config."""
        return [
            m
            for m in self._checksums.values()
            if m.board_type == board_type and m.num_players == num_players
        ]

    def group_by_config(
        self, models: list[UniqueModel]
    ) -> dict[str, list[UniqueModel]]:
        """Group models by config key (board_type_num_players)."""
        grouped: dict[str, list[UniqueModel]] = {}
        for model in models:
            if model.board_type == "unknown" or model.num_players == 0:
                continue  # Skip models without valid config
            config_key = f"{model.board_type}_{model.num_players}p"
            if config_key not in grouped:
                grouped[config_key] = []
            grouped[config_key].append(model)
        return grouped

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about cached models."""
        models = list(self._checksums.values())
        if not models:
            return {"total": 0}

        by_config = self.group_by_config(models)
        by_family: dict[str, int] = {}
        by_arch: dict[str, int] = {}
        total_size = 0
        total_copies = 0

        for model in models:
            by_family[model.model_family] = by_family.get(model.model_family, 0) + 1
            by_arch[model.architecture] = by_arch.get(model.architecture, 0) + 1
            total_size += model.file_size
            total_copies += len(model.all_paths)

        return {
            "total_unique": len(models),
            "total_copies": total_copies,
            "dedup_ratio": total_copies / len(models) if models else 0,
            "total_size_gb": total_size / (1024**3),
            "by_config": {k: len(v) for k, v in by_config.items()},
            "by_family": dict(sorted(by_family.items(), key=lambda x: -x[1])[:20]),
            "by_architecture": by_arch,
        }

    def print_report(self, models: list[UniqueModel] | None = None) -> None:
        """Print a human-readable deduplication report."""
        if models is None:
            models = list(self._checksums.values())

        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("MODEL DEDUPLICATION REPORT")
        print("=" * 60)
        print(f"\nTotal unique models: {stats['total_unique']:,}")
        print(f"Total file copies: {stats['total_copies']:,}")
        print(f"Deduplication ratio: {stats['dedup_ratio']:.1f}x")
        print(f"Total storage: {stats['total_size_gb']:.2f} GB")

        print("\n--- By Configuration ---")
        for config_key, count in sorted(stats["by_config"].items()):
            print(f"  {config_key}: {count} models")

        print("\n--- By Architecture ---")
        for arch, count in sorted(stats["by_architecture"].items(), key=lambda x: -x[1]):
            print(f"  {arch}: {count} models")

        print("\n--- Top Model Families ---")
        for family, count in list(stats["by_family"].items())[:10]:
            print(f"  {family}: {count} models")

        print("=" * 60 + "\n")
