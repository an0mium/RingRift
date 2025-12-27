"""Aria2 Transport - High-performance multi-connection download transport.

This module provides aria2-based data synchronization for the unified data sync
service. It uses aria2c for resilient, multi-connection downloads that work well
over unstable network connections.

Key features:
1. Multi-connection parallel downloads (16 connections per server by default)
2. Multi-source downloads when multiple peers have the same file
3. Auto-resume on connection drops
4. Metalink support for efficient multi-source coordination
5. Integration with the node inventory system

Usage:
    transport = Aria2Transport()

    # Sync from multiple sources
    result = await transport.sync_from_sources(
        sources=["http://node1:8766", "http://node2:8766"],
        local_dir=Path("data/games/synced"),
        patterns=["*.db"],
    )
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import (
    LARGE_CHUNK_SIZE,
    compute_file_checksum,
    verify_file_checksum,
)

# Import torrent generator and cluster manifest for P2P sync
try:
    from app.distributed.torrent_generator import TorrentGenerator, get_torrent_generator
    from app.distributed.cluster_manifest import (
        ClusterManifest,
        TorrentMetadata,
        get_cluster_manifest,
    )
    HAS_TORRENT_SUPPORT = True
except ImportError:
    HAS_TORRENT_SUPPORT = False
    TorrentGenerator = None  # type: ignore
    TorrentMetadata = None  # type: ignore

logger = logging.getLogger(__name__)

# Import circuit breaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        get_adaptive_timeout,
        get_operation_breaker,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False

# Maximum total timeout for batch operations (prevent hour-long stalls)
MAX_BATCH_TIMEOUT = 1800  # 30 minutes max for any batch operation
MAX_PER_FILE_TIMEOUT = 120  # 2 minutes per file max


@dataclass
class Aria2Config:
    """Configuration for aria2 transport."""
    connections_per_server: int = 16
    split: int = 16  # Number of parallel segments per file
    min_split_size: str = "1M"  # Minimum size to split
    max_concurrent_downloads: int = 5
    connect_timeout: int = 10
    timeout: int = 300
    retry_wait: int = 3
    max_tries: int = 5
    continue_download: bool = True
    check_integrity: bool = True
    allow_overwrite: bool = True
    # Data server port (matches aria2_data_sync.py)
    data_server_port: int = 8766
    # Checksum verification after download
    verify_checksum: bool = True
    # BitTorrent support for P2P swarm downloads (resilient for flaky connections)
    enable_bittorrent: bool = True
    bt_enable_dht: bool = True  # Distributed Hash Table for peer discovery (trackerless)
    bt_enable_lpd: bool = True  # Local Peer Discovery
    bt_enable_pex: bool = True  # Peer Exchange
    bt_max_peers: int = 55
    bt_tracker_timeout: int = 60
    bt_listen_port: int = 51413  # BitTorrent listen port
    bt_dht_listen_port: int = 6881  # DHT discovery port
    # Seed ratio (2.0 = seed until 2:1 upload ratio, helps cluster resilience)
    seed_ratio: float = 2.0
    # Minimum seed time in seconds (seed at least this long regardless of ratio)
    seed_time: int = 3600  # 1 hour
    # Cache directory for .torrent files
    torrent_cache_dir: str = "data/torrents"
    # DHT routing table persistence (enables faster peer discovery on restart)
    dht_file_path: str = ".dht.dat"
    dht_save_interval: int = 30  # Save DHT routing table every 30 seconds
    # December 2025: Automatic BitTorrent preference for large files
    # Files above this threshold will prefer BitTorrent over HTTP (50MB default)
    # BitTorrent provides piece-level verification which prevents corruption
    # issues seen with rsync --partial on flaky connections
    prefer_torrent_for_large_files: bool = True
    large_file_threshold_bytes: int = 50_000_000  # 50MB - use BitTorrent above this


@dataclass
class FileInfo:
    """Information about a remote file."""
    name: str
    path: str
    size_bytes: int
    mtime: float
    category: str  # 'games', 'models', 'training', 'elo'
    checksum: str | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class NodeInventory:
    """Inventory from a remote node."""
    url: str
    hostname: str = ""
    files: dict[str, FileInfo] = field(default_factory=dict)
    reachable: bool = False
    total_size_mb: float = 0


@dataclass
class Aria2SyncResult:
    """Result of an aria2 sync operation.

    Dec 2025: Renamed from SyncResult to avoid confusion with
    app.coordination.sync_constants.SyncResult (general-purpose sync result).
    """
    success: bool
    files_synced: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    method: str = "aria2"


# Backwards compatibility alias
SyncResult = Aria2SyncResult


def check_aria2_available() -> bool:
    """Check if aria2c is available."""
    return shutil.which("aria2c") is not None


class Aria2Transport:
    """High-performance aria2-based data transport."""

    def __init__(self, config: Aria2Config | None = None):
        self.config = config or Aria2Config()
        self._aria2_available: bool | None = None
        self._session: Any | None = None

    def is_available(self) -> bool:
        """Check if aria2 transport is available."""
        if self._aria2_available is None:
            self._aria2_available = check_aria2_available()
        return self._aria2_available

    async def _get_session(self):
        """Get or create aiohttp session for inventory fetches."""
        if self._session is None:
            try:
                import aiohttp
                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout,
                    connect=self.config.connect_timeout,
                )
                self._session = aiohttp.ClientSession(timeout=timeout)
            except ImportError:
                logger.warning("aiohttp not available, using requests fallback")
                return None
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_inventory(
        self,
        source_url: str,
        timeout: int = 10,
    ) -> NodeInventory | None:
        """Fetch inventory from a data server node.

        Args:
            source_url: Base URL of the data server (e.g., http://node1:8766)
            timeout: Request timeout in seconds

        Returns:
            NodeInventory if successful, None otherwise
        """
        # Extract host for circuit breaker tracking
        host = source_url.split("://")[-1].split(":")[0].split("/")[0]

        # Circuit breaker check
        if HAS_CIRCUIT_BREAKER:
            breaker = get_operation_breaker("aria2")
            if not breaker.can_execute(host):
                logger.debug(f"Circuit breaker open for {host}, skipping inventory fetch")
                return None
            timeout = int(get_adaptive_timeout("aria2", host, float(timeout)))

        inventory_url = f"{source_url.rstrip('/')}/inventory.json"

        try:
            session = await self._get_session()
            if session:
                async with session.get(inventory_url, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Record success
                        if HAS_CIRCUIT_BREAKER:
                            get_operation_breaker("aria2").record_success(host)
                        return self._parse_inventory(source_url, data)
            else:
                # Fallback to requests
                import requests
                resp = await asyncio.to_thread(
                    requests.get,
                    inventory_url,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    # Record success
                    if HAS_CIRCUIT_BREAKER:
                        get_operation_breaker("aria2").record_success(host)
                    return self._parse_inventory(source_url, resp.json())

            # Non-200 response is a failure
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("aria2").record_failure(host)

        except Exception as e:
            logger.debug(f"Failed to fetch inventory from {source_url}: {e}")
            # Record failure
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("aria2").record_failure(host, e)

        return None

    def _parse_inventory(self, source_url: str, data: dict) -> NodeInventory:
        """Parse inventory JSON into NodeInventory object."""
        files = {}
        total_size = 0
        base_url = source_url.rstrip("/")

        def add_file(path: str, file_data: dict[str, Any], category_hint: str | None = None) -> None:
            nonlocal total_size
            if not path:
                return
            path = path.lstrip("/")
            name = file_data.get("name") or Path(path).name
            category = file_data.get("category") or category_hint or "unknown"
            file_info = FileInfo(
                name=name,
                path=path,
                size_bytes=file_data.get("size_bytes", 0),
                mtime=file_data.get("mtime", 0),
                category=category,
                checksum=file_data.get("checksum"),
                sources=[f"{base_url}/{path}"],
            )
            if path not in files:
                files[path] = file_info
                total_size += file_info.size_bytes

        files_map = data.get("files", {}) or {}
        for path, file_data in files_map.items():
            add_file(path, file_data, file_data.get("category"))

        for category in ["games", "models", "training", "elo"]:
            for file_data in data.get(category, []):
                path = file_data.get("path") or f"{category}/{file_data.get('name', '')}"
                add_file(path, file_data, category)

        return NodeInventory(
            url=source_url,
            hostname=data.get("hostname", ""),
            files=files,
            reachable=True,
            total_size_mb=total_size / (1024 * 1024),
        )

    async def discover_sources(
        self,
        source_urls: list[str],
        parallel: int = 10,
    ) -> tuple[list[NodeInventory], dict[str, list[str]]]:
        """Discover all available sources and aggregate file information.

        Returns:
            Tuple of (list of NodeInventory, dict mapping filename to list of source URLs)
        """
        # Fetch inventories in parallel
        tasks = [self.fetch_inventory(url) for url in source_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        inventories = []
        file_sources: dict[str, list[str]] = {}

        for result in results:
            if isinstance(result, NodeInventory) and result.reachable:
                inventories.append(result)
                for file_info in result.files.values():
                    if file_info.name not in file_sources:
                        file_sources[file_info.name] = []
                    file_sources[file_info.name].extend(file_info.sources)

        logger.info(
            f"Discovered {len(inventories)} reachable nodes with {len(file_sources)} unique files"
        )
        return inventories, file_sources

    def _build_aria2_command(
        self,
        output_dir: Path,
        url_file: Path | None = None,
        urls: list[str] | None = None,
        torrent_file: Path | None = None,
    ) -> list[str]:
        """Build aria2c command with optimal settings."""
        cmd = [
            "aria2c",
            f"--max-connection-per-server={self.config.connections_per_server}",
            f"--split={self.config.split}",
            f"--min-split-size={self.config.min_split_size}",
            f"--max-concurrent-downloads={self.config.max_concurrent_downloads}",
            f"--connect-timeout={self.config.connect_timeout}",
            f"--timeout={self.config.timeout}",
            f"--retry-wait={self.config.retry_wait}",
            f"--max-tries={self.config.max_tries}",
            f"--dir={output_dir}",
            "--file-allocation=falloc",
            "--console-log-level=warn",
            "--summary-interval=0",
        ]

        if self.config.continue_download:
            cmd.append("--continue=true")
        if self.config.allow_overwrite:
            cmd.append("--allow-overwrite=true")
        if self.config.check_integrity:
            cmd.append("--check-integrity=true")

        # BitTorrent options for P2P swarm downloads (resilient for flaky connections)
        if self.config.enable_bittorrent:
            if self.config.bt_enable_dht:
                cmd.append("--enable-dht=true")
                cmd.append(f"--dht-listen-port={self.config.bt_dht_listen_port}")
                # Persist DHT routing table for faster peer discovery on restart
                cmd.append(f"--dht-file-path={self.config.dht_file_path}")
                cmd.append(f"--dht-save-interval={self.config.dht_save_interval}")
            if self.config.bt_enable_lpd:
                cmd.append("--bt-enable-lpd=true")
            if self.config.bt_enable_pex:
                cmd.append("--enable-peer-exchange=true")
            cmd.append(f"--bt-max-peers={self.config.bt_max_peers}")
            cmd.append(f"--bt-tracker-timeout={self.config.bt_tracker_timeout}")
            cmd.append(f"--listen-port={self.config.bt_listen_port}")
            # Seeding configuration for cluster resilience
            cmd.append(f"--seed-ratio={self.config.seed_ratio}")
            if self.config.seed_time > 0:
                cmd.append(f"--seed-time={self.config.seed_time // 60}")  # aria2 uses minutes

        if torrent_file:
            cmd.append(str(torrent_file))
        elif url_file:
            cmd.append(f"--input-file={url_file}")
        elif urls:
            cmd.extend(urls)

        return cmd

    def _resolve_category_dir(self, local_dir: Path, category: str) -> Path:
        if local_dir.name == "synced" and local_dir.parent.name == category:
            return local_dir
        if category == "games" and local_dir.name in ("games", "selfplay"):
            return local_dir
        if category == "models" and local_dir.name == "models":
            return local_dir
        if category == "training" and local_dir.name in ("training", "training_data"):
            return local_dir
        if category == "elo" and local_dir.name == "elo":
            return local_dir
        return local_dir / category

    async def download_file(
        self,
        sources: list[str],
        output_dir: Path,
        filename: str | None = None,
        expected_checksum: str | None = None,
    ) -> tuple[bool, int, str]:
        """Download a single file from multiple sources using aria2.

        Args:
            sources: List of URLs to download from (aria2 will try all)
            output_dir: Directory to save the file
            filename: Optional filename override
            expected_checksum: Optional SHA256 checksum for verification

        Returns:
            Tuple of (success, bytes_downloaded, error_message)
        """
        if not self.is_available():
            return False, 0, "aria2c not available"

        if not sources:
            return False, 0, "No sources provided"

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_aria2_command(output_dir, urls=sources)
        if filename:
            cmd.append(f"--out={filename}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout,
            )

            if process.returncode == 0:
                # Get file path and size
                if filename:
                    filepath = output_dir / filename
                else:
                    # Extract filename from first URL
                    filepath = output_dir / Path(sources[0]).name

                if not filepath.exists():
                    return False, 0, f"Download completed but file not found: {filepath}"

                size = filepath.stat().st_size

                # Verify checksum if provided and verification is enabled
                if expected_checksum and self.config.verify_checksum:
                    if not verify_file_checksum(filepath, expected_checksum):
                        actual = compute_file_checksum(filepath, chunk_size=LARGE_CHUNK_SIZE, truncate=len(expected_checksum))
                        logger.warning(
                            f"Checksum mismatch for {filepath.name}: "
                            f"expected {expected_checksum[:16]}..., got {actual[:16]}..."
                        )
                        # Delete corrupted file
                        filepath.unlink(missing_ok=True)
                        return False, 0, f"Checksum mismatch: expected {expected_checksum[:16]}, got {actual[:16]}"

                return True, size, ""
            else:
                error = stderr.decode()[:200] if stderr else "Unknown error"
                return False, 0, error

        except asyncio.TimeoutError:
            return False, 0, "Download timeout"
        except Exception as e:
            return False, 0, str(e)[:200]

    async def download_torrent(
        self,
        torrent_source: str | Path,
        output_dir: Path,
        seed_after: bool = True,
        timeout: int | None = None,
    ) -> tuple[bool, int, str]:
        """Download using BitTorrent with automatic seeding.

        BitTorrent is ideal for nodes with flaky connections because:
        - Downloads can resume from any peer
        - Multiple peers provide redundancy
        - DHT enables peer discovery without central tracker
        - Seeding helps other nodes in the cluster

        Args:
            torrent_source: Path to .torrent file or magnet:?xt=urn... URI
            output_dir: Directory to save downloaded files
            seed_after: Continue seeding after download (helps cluster resilience)
            timeout: Optional timeout override (default uses config.timeout)

        Returns:
            Tuple of (success, bytes_downloaded, error_message)
        """
        if not self.is_available():
            return False, 0, "aria2c not available"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine if this is a magnet link or torrent file
        torrent_str = str(torrent_source)
        is_magnet = torrent_str.startswith("magnet:")

        # Build command with BitTorrent-specific settings
        cmd = [
            "aria2c",
            f"--dir={output_dir}",
            f"--max-concurrent-downloads={self.config.max_concurrent_downloads}",
            f"--bt-max-peers={self.config.bt_max_peers}",
            f"--bt-tracker-timeout={self.config.bt_tracker_timeout}",
            f"--listen-port={self.config.bt_listen_port}",
            "--file-allocation=falloc",
            "--console-log-level=warn",
            "--summary-interval=0",
        ]

        # DHT for trackerless peer discovery (critical for flaky connections)
        if self.config.bt_enable_dht:
            cmd.extend([
                "--enable-dht=true",
                f"--dht-listen-port={self.config.bt_dht_listen_port}",
                "--dht-file-path=data/torrents/dht.dat",  # Persist DHT state
            ])

        # Local Peer Discovery (find peers on local network)
        if self.config.bt_enable_lpd:
            cmd.append("--bt-enable-lpd=true")

        # Peer Exchange (learn about more peers from connected peers)
        if self.config.bt_enable_pex:
            cmd.append("--enable-peer-exchange=true")

        # Seeding configuration
        if seed_after:
            cmd.append(f"--seed-ratio={self.config.seed_ratio}")
            if self.config.seed_time > 0:
                cmd.append(f"--seed-time={self.config.seed_time // 60}")
        else:
            cmd.append("--seed-ratio=0.0")  # Don't seed

        # Allow continuing partial downloads
        if self.config.continue_download:
            cmd.append("--continue=true")
        if self.config.allow_overwrite:
            cmd.append("--allow-overwrite=true")

        # Add the torrent source
        if is_magnet:
            cmd.append(torrent_str)
        else:
            # Verify torrent file exists
            torrent_path = Path(torrent_source)
            if not torrent_path.exists():
                return False, 0, f"Torrent file not found: {torrent_path}"
            cmd.append(str(torrent_path))

        effective_timeout = timeout or self.config.timeout

        try:
            logger.info(f"Starting BitTorrent download: {torrent_str[:60]}...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=effective_timeout,
            )

            if process.returncode == 0:
                # Calculate total size of downloaded files
                total_size = sum(
                    f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
                )
                logger.info(f"BitTorrent download complete: {total_size / 1024 / 1024:.1f} MB")
                return True, total_size, ""
            else:
                error = stderr.decode()[:200] if stderr else "Unknown error"
                return False, 0, error

        except asyncio.TimeoutError:
            return False, 0, f"BitTorrent download timeout after {effective_timeout}s"
        except Exception as e:
            return False, 0, str(e)[:200]

    async def download_with_torrent_fallback(
        self,
        file_path: str,
        output_dir: Path,
        fallback_urls: list[str] | None = None,
        min_seeders: int = 1,
        expected_checksum: str | None = None,
        expected_size: int | None = None,
        prefer_torrent: bool | None = None,
    ) -> tuple[bool, int, str]:
        """Download file using BitTorrent if available, HTTP fallback otherwise.

        This method provides resilient downloads by:
        1. Checking if a torrent exists for this file with active seeders
        2. Using BitTorrent swarm if seeders available (handles connection drops)
        3. Falling back to multi-connection HTTP if no torrent/seeders

        December 2025: Added prefer_torrent parameter and automatic preference
        for large files based on config.large_file_threshold_bytes. BitTorrent
        provides piece-level verification which prevents corruption issues seen
        with rsync --partial on flaky connections.

        Args:
            file_path: Path to the file (used to lookup torrent)
            output_dir: Directory to save the file
            fallback_urls: HTTP URLs to use if BitTorrent unavailable
            min_seeders: Minimum seeders required to use BitTorrent
            expected_checksum: Optional SHA256 checksum for verification
            expected_size: Optional expected file size in bytes
            prefer_torrent: Force BitTorrent preference (None=use config)

        Returns:
            Tuple of (success, bytes_downloaded, error_message)
        """
        if not self.is_available():
            return False, 0, "aria2c not available"

        # Determine if we should prefer BitTorrent based on file size
        # Large files (>50MB) benefit from piece-level verification
        use_torrent_preference = prefer_torrent
        if use_torrent_preference is None and self.config.prefer_torrent_for_large_files:
            if expected_size and expected_size > self.config.large_file_threshold_bytes:
                use_torrent_preference = True
                logger.info(
                    f"Large file detected ({expected_size / 1024 / 1024:.1f}MB > "
                    f"{self.config.large_file_threshold_bytes / 1024 / 1024:.0f}MB threshold), "
                    f"preferring BitTorrent"
                )

        # Try BitTorrent first if torrent support available
        if HAS_TORRENT_SUPPORT and (use_torrent_preference or use_torrent_preference is None):
            try:
                manifest = get_cluster_manifest()
                torrent_meta = manifest.get_torrent_for_file(file_path)

                if torrent_meta and len(torrent_meta.seeders) >= min_seeders:
                    torrent_path = Path(torrent_meta.torrent_path)
                    if torrent_path.exists():
                        logger.info(
                            f"Using BitTorrent for {file_path} "
                            f"({len(torrent_meta.seeders)} seeders available)"
                        )
                        success, size, error = await self.download_torrent(
                            torrent_path,
                            output_dir,
                            seed_after=True,
                        )
                        if success:
                            # Register as seeder after successful download
                            await self._register_as_seeder(torrent_meta.info_hash)
                            return success, size, error
                        else:
                            logger.warning(
                                f"BitTorrent download failed, falling back to HTTP: {error}"
                            )
                    else:
                        logger.debug(
                            f"Torrent file not found locally: {torrent_path}, using HTTP"
                        )
                else:
                    if torrent_meta:
                        logger.debug(
                            f"Insufficient seeders ({len(torrent_meta.seeders)}) for {file_path}, using HTTP"
                        )
            except Exception as e:
                logger.debug(f"Error checking torrent availability: {e}")

        # Fall back to HTTP
        if not fallback_urls:
            return False, 0, "No fallback URLs and BitTorrent unavailable"

        filename = Path(file_path).name
        return await self.download_file(
            sources=fallback_urls,
            output_dir=output_dir,
            filename=filename,
            expected_checksum=expected_checksum,
        )

    async def _register_as_seeder(self, info_hash: str) -> None:
        """Register this node as a seeder for a torrent.

        Called after successful download to update ClusterManifest.
        """
        if not HAS_TORRENT_SUPPORT:
            return

        try:
            manifest = get_cluster_manifest()
            manifest.add_seeder(info_hash, manifest.node_id)
            logger.debug(f"Registered as seeder for {info_hash[:16]}...")
        except Exception as e:
            logger.debug(f"Failed to register as seeder: {e}")

    async def seed_file(
        self,
        file_path: Path,
        torrent_path: Path | None = None,
        duration_seconds: int | None = None,
    ) -> tuple[bool, str]:
        """Seed a file for P2P distribution.

        Creates torrent if needed and seeds for specified duration.

        Args:
            file_path: Path to the file to seed
            torrent_path: Optional path to existing .torrent file
            duration_seconds: How long to seed (default: config.seed_time)

        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_available():
            return False, "aria2c not available"

        if not file_path.exists():
            return False, f"File not found: {file_path}"

        duration = duration_seconds or self.config.seed_time

        # Create torrent if needed
        if torrent_path is None and HAS_TORRENT_SUPPORT:
            try:
                generator = get_torrent_generator()
                torrent_path, info_hash = generator.create_torrent(file_path)

                # Register torrent in manifest
                manifest = get_cluster_manifest()
                file_size = file_path.stat().st_size
                manifest.register_torrent(
                    info_hash=info_hash,
                    file_path=str(file_path),
                    torrent_path=str(torrent_path),
                    file_size=file_size,
                )
            except Exception as e:
                return False, f"Failed to create torrent: {e}"

        if torrent_path is None or not torrent_path.exists():
            return False, "No torrent file available"

        # Seed the file
        cmd = [
            "aria2c",
            f"--dir={file_path.parent}",
            f"--seed-time={duration // 60}",  # Convert to minutes
            "--seed-ratio=0.0",  # No ratio limit, use time limit
            "--bt-seed-unverified=true",  # Seed without re-verifying
            "--console-log-level=warn",
            "--summary-interval=0",
        ]

        if self.config.bt_enable_dht:
            cmd.extend([
                "--enable-dht=true",
                f"--dht-listen-port={self.config.bt_dht_listen_port}",
                f"--dht-file-path={self.config.dht_file_path}",
            ])

        if self.config.bt_enable_lpd:
            cmd.append("--bt-enable-lpd=true")

        if self.config.bt_enable_pex:
            cmd.append("--enable-peer-exchange=true")

        cmd.append(str(torrent_path))

        try:
            logger.info(f"Seeding {file_path.name} for {duration // 60} minutes...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Don't wait for completion - let it run in background
            # The user can monitor via aria2 RPC or process list
            await asyncio.sleep(1)  # Let it start

            if process.returncode is not None and process.returncode != 0:
                stderr = (await process.stderr.read()).decode()[:200] if process.stderr else ""
                return False, f"Seeding failed to start: {stderr}"

            return True, ""

        except Exception as e:
            return False, str(e)[:200]

    async def create_and_register_torrent(
        self,
        file_path: Path,
        web_seeds: list[str] | None = None,
    ) -> tuple[Path | None, str | None, str]:
        """Create a torrent for a file and register it in ClusterManifest.

        This is the recommended way to prepare a file for P2P distribution.
        Call this after creating/downloading a large file to enable swarm sync.

        Args:
            file_path: Path to the file
            web_seeds: Optional HTTP URLs for hybrid HTTP+BT downloads

        Returns:
            Tuple of (torrent_path, info_hash, error_message)
        """
        if not HAS_TORRENT_SUPPORT:
            return None, None, "Torrent support not available"

        if not file_path.exists():
            return None, None, f"File not found: {file_path}"

        try:
            generator = get_torrent_generator()
            torrent_path, info_hash = generator.create_torrent(
                file_path,
                web_seeds=web_seeds,
            )

            # Register in manifest
            manifest = get_cluster_manifest()
            file_size = file_path.stat().st_size
            piece_size = generator.piece_size or 262144

            manifest.register_torrent(
                info_hash=info_hash,
                file_path=str(file_path),
                torrent_path=str(torrent_path),
                file_size=file_size,
                piece_size=piece_size,
                piece_count=(file_size + piece_size - 1) // piece_size,
                web_seeds=web_seeds,
            )

            logger.info(f"Created and registered torrent: {info_hash[:16]}... for {file_path.name}")
            return torrent_path, info_hash, ""

        except Exception as e:
            return None, None, str(e)

    async def download_batch(
        self,
        file_sources: dict[str, list[str]],
        output_dir: Path,
        category: str | None = "games",
        expected_checksums: dict[str, str] | None = None,
    ) -> SyncResult:
        """Download multiple files using aria2 with a URL list file.

        Args:
            file_sources: Dict mapping filenames to list of source URLs
            output_dir: Directory to save files
            category: Category subdirectory (games, models, training, elo)
            expected_checksums: Optional dict mapping filenames to SHA256 checksums

        Returns:
            SyncResult with download statistics
        """
        if not self.is_available():
            return SyncResult(
                success=False,
                errors=["aria2c not available"],
            )

        if not file_sources:
            return SyncResult(success=True)

        start_time = time.time()
        category_dir = output_dir if not category else output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        checksums = expected_checksums or {}

        # Create URL list file for aria2
        # Format: URL\n  out=filename\n
        url_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                url_file = Path(f.name)
                for filename, sources in file_sources.items():
                    # Write all sources for this file (aria2 will try them in order)
                    for source in sources:
                        f.write(f"{source}\n")
                    f.write(f"  out={filename}\n")

            cmd = self._build_aria2_command(category_dir, url_file=url_file)

            # Calculate reasonable timeout: base timeout + per-file overhead
            # Cap at MAX_BATCH_TIMEOUT to prevent hour-long stalls
            per_file_timeout = min(self.config.timeout, MAX_PER_FILE_TIMEOUT)
            total_timeout = min(
                self.config.timeout + (per_file_timeout * min(len(file_sources), 50)),
                MAX_BATCH_TIMEOUT
            )

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=total_timeout,
            )

            # Count successful downloads and verify checksums
            files_synced = 0
            files_failed = 0
            bytes_transferred = 0
            errors = []

            for filename in file_sources:
                filepath = category_dir / filename
                if filepath.exists():
                    # Verify checksum if provided and verification is enabled
                    if filename in checksums and self.config.verify_checksum:
                        expected = checksums[filename]
                        if not verify_file_checksum(filepath, expected):
                            actual = compute_file_checksum(
                                filepath, chunk_size=LARGE_CHUNK_SIZE, truncate=len(expected)
                            )
                            logger.warning(
                                f"Checksum mismatch for {filename}: "
                                f"expected {expected[:16]}..., got {actual[:16]}..."
                            )
                            # Delete corrupted file
                            filepath.unlink(missing_ok=True)
                            files_failed += 1
                            errors.append(f"Checksum mismatch: {filename}")
                            continue

                    files_synced += 1
                    bytes_transferred += filepath.stat().st_size
                else:
                    files_failed += 1
                    errors.append(f"Failed to download: {filename}")

            if stderr and files_failed > 0:
                errors.append(stderr.decode()[:500])

            return SyncResult(
                success=files_failed == 0,
                files_synced=files_synced,
                files_failed=files_failed,
                bytes_transferred=bytes_transferred,
                duration_seconds=time.time() - start_time,
                errors=errors,
                method="aria2",
            )

        except asyncio.TimeoutError:
            return SyncResult(
                success=False,
                duration_seconds=time.time() - start_time,
                errors=["Batch download timeout"],
                method="aria2",
            )
        except Exception as e:
            return SyncResult(
                success=False,
                duration_seconds=time.time() - start_time,
                errors=[str(e)],
                method="aria2",
            )
        finally:
            if url_file and url_file.exists():
                url_file.unlink()

    async def sync_games(
        self,
        source_urls: list[str],
        local_dir: Path,
        max_age_hours: float = 168,  # 1 week
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync game databases from multiple sources.

        Args:
            source_urls: List of data server URLs
            local_dir: Local directory to sync to
            max_age_hours: Only sync files newer than this
            dry_run: If True, just report what would be synced

        Returns:
            SyncResult with sync statistics
        """
        start_time = time.time()

        # Discover all sources
        inventories, _file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        # Filter to games category and by age
        cutoff_time = time.time() - (max_age_hours * 3600)
        games_to_sync: dict[str, list[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "games")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "games":
                    continue
                if file_info.mtime < cutoff_time:
                    continue

                # Check if we already have this file
                local_path = category_dir / file_info.name
                if local_path.exists():
                    local_mtime = local_path.stat().st_mtime
                    if local_mtime >= file_info.mtime:
                        continue

                if file_info.name not in games_to_sync:
                    games_to_sync[file_info.name] = []
                games_to_sync[file_info.name].extend(file_info.sources)

        if not games_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(games_to_sync)} game files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_synced=0,
                files_failed=len(games_to_sync),  # Reported as "would sync"
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(games_to_sync, category_dir, None)

    async def sync_models(
        self,
        source_urls: list[str],
        local_dir: Path,
        patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync model checkpoints from multiple sources."""
        start_time = time.time()

        inventories, _file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        models_to_sync: dict[str, list[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "models")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "models":
                    continue
                if patterns and not any(fnmatch.fnmatch(file_info.name, pattern) for pattern in patterns):
                    continue

                local_path = category_dir / file_info.name
                if local_path.exists():
                    # Check if remote is newer or different size
                    local_stat = local_path.stat()
                    if local_stat.st_mtime >= file_info.mtime:
                        continue

                if file_info.name not in models_to_sync:
                    models_to_sync[file_info.name] = []
                models_to_sync[file_info.name].extend(file_info.sources)

        if not models_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(models_to_sync)} model files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_failed=len(models_to_sync),
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(models_to_sync, category_dir, None)

    async def sync_training_data(
        self,
        source_urls: list[str],
        local_dir: Path,
        max_age_hours: float = 24,
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync training data batches from multiple sources."""
        start_time = time.time()

        inventories, _file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        cutoff_time = time.time() - (max_age_hours * 3600)
        training_to_sync: dict[str, list[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "training")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "training":
                    continue
                if file_info.mtime < cutoff_time:
                    continue

                local_path = category_dir / file_info.name
                if local_path.exists():
                    continue

                if file_info.name not in training_to_sync:
                    training_to_sync[file_info.name] = []
                training_to_sync[file_info.name].extend(file_info.sources)

        if not training_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(training_to_sync)} training files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_failed=len(training_to_sync),
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(training_to_sync, category_dir, None)

    async def full_cluster_sync(
        self,
        source_urls: list[str],
        local_dir: Path,
        categories: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, SyncResult]:
        """Sync all data categories from cluster.

        Args:
            source_urls: List of data server URLs
            local_dir: Local base directory
            categories: Categories to sync (default: all)
            dry_run: If True, just report what would be synced

        Returns:
            Dict mapping category to SyncResult
        """
        if categories is None:
            categories = ["games", "models", "training"]

        results = {}

        if "games" in categories:
            results["games"] = await self.sync_games(
                source_urls, local_dir, dry_run=dry_run
            )

        if "models" in categories:
            results["models"] = await self.sync_models(
                source_urls, local_dir, dry_run=dry_run
            )

        if "training" in categories:
            results["training"] = await self.sync_training_data(
                source_urls, local_dir, dry_run=dry_run
            )

        return results


# Factory function for integration with unified_data_sync
def create_aria2_transport(config: dict[str, Any] | None = None) -> Aria2Transport:
    """Create an Aria2Transport instance from config dict."""
    if config:
        aria2_config = Aria2Config(
            connections_per_server=config.get("connections_per_server", 16),
            split=config.get("split", 16),
            max_concurrent_downloads=config.get("max_concurrent_downloads", 5),
            timeout=config.get("timeout", 300),
            data_server_port=config.get("data_server_port", 8766),
        )
        return Aria2Transport(aria2_config)
    return Aria2Transport()
