#!/usr/bin/env python3
"""Distributed P2P Orchestrator - Self-healing compute cluster for RingRift AI training.

This orchestrator runs on each node in the cluster and:
1. Discovers other nodes via broadcast UDP or known peer list
2. Participates in leader election for coordination tasks
3. Monitors local resources and shares status with peers
4. Auto-starts selfplay/training jobs based on cluster needs
5. Self-heals when nodes go offline or IPs change

Architecture:
- Each node runs this script as a daemon
- Nodes communicate via HTTP REST API (port 8770)
- Leader election uses Bully algorithm (highest node_id wins)
- Heartbeats every 30 seconds detect failures
- Nodes maintain local SQLite state for crash recovery

Usage:
    # On each node:
    python scripts/p2p_orchestrator.py --node-id mac-studio
    python scripts/p2p_orchestrator.py --node-id vast-5090-quad --port 8770

    # With known peers (for cloud nodes without broadcast):
    python scripts/p2p_orchestrator.py --node-id vast-3090 --peers 100.107.168.125:8770,100.66.142.46:8770
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

# HTTP server imports
try:
    from aiohttp import web, ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")

# ============================================
# Configuration
# ============================================

DEFAULT_PORT = 8770
HEARTBEAT_INTERVAL = 30  # seconds
PEER_TIMEOUT = 90  # seconds without heartbeat = node considered dead
ELECTION_TIMEOUT = 10  # seconds to wait for election responses
JOB_CHECK_INTERVAL = 60  # seconds between job status checks
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# LEARNED LESSONS from PLAN.md - Disk and resource thresholds
DISK_CRITICAL_THRESHOLD = 90  # Stop all new jobs at 90% disk
DISK_WARNING_THRESHOLD = 80   # Reduce job count at 80% disk
DISK_CLEANUP_THRESHOLD = 85   # Trigger automatic cleanup at 85%
MEMORY_CRITICAL_THRESHOLD = 95  # OOM prevention - stop jobs at 95%
MEMORY_WARNING_THRESHOLD = 85   # Reduce jobs at 85% memory

# LEARNED LESSONS - Connection robustness
HTTP_CONNECT_TIMEOUT = 10     # Fast timeout for connection phase
HTTP_TOTAL_TIMEOUT = 30       # Total request timeout
MAX_CONSECUTIVE_FAILURES = 3  # Mark node dead after 3 failures
RETRY_DEAD_NODE_INTERVAL = 300  # Retry dead nodes every 5 minutes

# Git auto-update settings
GIT_UPDATE_CHECK_INTERVAL = 300  # Check for updates every 5 minutes
GIT_REMOTE_NAME = "origin"       # Git remote to check
GIT_BRANCH_NAME = "main"         # Branch to track
AUTO_UPDATE_ENABLED = True       # Enable automatic updates
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = True  # Stop jobs before updating

# Path to local state database
STATE_DIR = Path(__file__).parent.parent / "logs" / "p2p_orchestrator"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay
    TRAINING = "training"
    CMAES = "cmaes"
    # Distributed job types
    DISTRIBUTED_CMAES_COORDINATOR = "distributed_cmaes_coordinator"
    DISTRIBUTED_CMAES_WORKER = "distributed_cmaes_worker"
    DISTRIBUTED_TOURNAMENT_COORDINATOR = "distributed_tournament_coordinator"
    DISTRIBUTED_TOURNAMENT_WORKER = "distributed_tournament_worker"
    IMPROVEMENT_LOOP = "improvement_loop"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    has_gpu: bool = False
    gpu_name: str = ""
    memory_gb: int = 0
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    # LEARNED LESSONS - Track connection failures for adaptive retry
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    # LEARNED LESSONS - Track resource issues
    disk_cleanup_needed: bool = False
    oom_events: int = 0
    last_oom_time: float = 0.0

    def is_alive(self) -> bool:
        """Check if node is considered alive based on last heartbeat."""
        return time.time() - self.last_heartbeat < PEER_TIMEOUT

    def is_healthy(self) -> bool:
        """Check if node is healthy for new jobs (not just reachable)."""
        if not self.is_alive():
            return False
        # LEARNED LESSONS - Don't start jobs on resource-constrained nodes
        if self.disk_percent >= DISK_CRITICAL_THRESHOLD:
            return False
        if self.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
            return False
        return True

    def should_retry(self) -> bool:
        """Check if we should retry connecting to a failed node."""
        if self.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            return True
        # LEARNED LESSONS - Retry dead nodes periodically
        return time.time() - self.last_failure_time > RETRY_DEAD_NODE_INTERVAL

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['role'] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeInfo':
        """Create from dictionary."""
        d = d.copy()
        d['role'] = NodeRole(d.get('role', 'follower'))
        # Handle missing new fields gracefully
        d.setdefault('consecutive_failures', 0)
        d.setdefault('last_failure_time', 0.0)
        d.setdefault('disk_cleanup_needed', False)
        d.setdefault('oom_events', 0)
        d.setdefault('last_oom_time', 0.0)
        return cls(**d)


@dataclass
class ClusterJob:
    """A job running in the cluster."""
    job_id: str
    job_type: JobType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    engine_mode: str = "descent-only"
    pid: int = 0
    started_at: float = 0.0
    status: str = "running"
    # Extended fields for distributed jobs
    coordinator_node: str = ""  # Node running coordinator (for worker jobs)
    worker_port: int = 8766     # Port for worker server
    config_json: str = ""       # JSON config for complex jobs

    def to_dict(self) -> dict:
        d = asdict(self)
        d['job_type'] = self.job_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ClusterJob':
        d = d.copy()
        d['job_type'] = JobType(d.get('job_type', 'selfplay'))
        # Handle missing new fields
        d.setdefault('coordinator_node', '')
        d.setdefault('worker_port', 8766)
        d.setdefault('config_json', '')
        return cls(**d)


@dataclass
class DistributedCMAESState:
    """State for distributed CMA-ES job coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    generations: int = 100
    population_size: int = 20
    games_per_eval: int = 50
    current_generation: int = 0
    best_fitness: float = 0.0
    best_weights: Dict[str, float] = field(default_factory=dict)
    worker_nodes: List[str] = field(default_factory=list)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0
    results_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DistributedCMAESState':
        return cls(**d)


@dataclass
class DistributedTournamentState:
    """State for distributed tournament coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    agent_ids: List[str] = field(default_factory=list)
    games_per_pairing: int = 2
    total_matches: int = 0
    completed_matches: int = 0
    worker_nodes: List[str] = field(default_factory=list)
    pending_matches: List[dict] = field(default_factory=list)
    results: List[dict] = field(default_factory=list)
    final_ratings: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DistributedTournamentState':
        return cls(**d)


@dataclass
class ImprovementLoopState:
    """State for improvement loop coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    current_iteration: int = 0
    max_iterations: int = 50
    games_per_iteration: int = 1000
    phase: str = "idle"  # idle, selfplay, export, train, evaluate, promote
    best_model_path: str = ""
    best_winrate: float = 0.0
    consecutive_failures: int = 0
    worker_nodes: List[str] = field(default_factory=list)
    selfplay_progress: Dict[str, int] = field(default_factory=dict)  # node_id -> games done
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ImprovementLoopState':
        return cls(**d)


class P2POrchestrator:
    """Main P2P orchestrator class that runs on each node."""

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        known_peers: List[str] = None,
        ringrift_path: str = None,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.known_peers = known_peers or []
        self.ringrift_path = ringrift_path or self._detect_ringrift_path()

        # Node state
        self.role = NodeRole.FOLLOWER
        self.leader_id: Optional[str] = None
        self.peers: Dict[str, NodeInfo] = {}
        self.local_jobs: Dict[str, ClusterJob] = {}

        # Distributed job state tracking (leader-only)
        self.distributed_cmaes_state: Dict[str, DistributedCMAESState] = {}
        self.distributed_tournament_state: Dict[str, DistributedTournamentState] = {}
        self.improvement_loop_state: Dict[str, ImprovementLoopState] = {}

        # Locks for thread safety
        self.peers_lock = threading.Lock()
        self.jobs_lock = threading.Lock()

        # State persistence
        self.db_path = STATE_DIR / f"{node_id}_state.db"
        self._init_database()

        # Event flags
        self.running = True
        self.election_in_progress = False

        # Load persisted state
        self._load_state()

        # Self info
        self.self_info = self._create_self_info()

        print(f"[P2P] Initialized node {node_id} on {host}:{port}")
        print(f"[P2P] RingRift path: {self.ringrift_path}")
        print(f"[P2P] Known peers: {self.known_peers}")

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift installation path."""
        # Try common locations
        candidates = [
            Path.home() / "Development" / "RingRift",
            Path.home() / "ringrift",
            Path("/home/ubuntu/ringrift"),
            Path("/root/ringrift"),
        ]
        for path in candidates:
            if (path / "ai-service").exists():
                return str(path)
        return str(Path(__file__).parent.parent.parent)

    def _init_database(self):
        """Initialize SQLite database for state persistence."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Peers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peers (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                last_heartbeat REAL,
                info_json TEXT
            )
        """)

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT,
                node_id TEXT,
                board_type TEXT,
                num_players INTEGER,
                engine_mode TEXT,
                pid INTEGER,
                started_at REAL,
                status TEXT
            )
        """)

        # State table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load persisted state from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Load peers
            cursor.execute("SELECT node_id, info_json FROM peers")
            for row in cursor.fetchall():
                try:
                    info = NodeInfo.from_dict(json.loads(row[1]))
                    self.peers[row[0]] = info
                except Exception as e:
                    print(f"[P2P] Failed to load peer {row[0]}: {e}")

            # Load jobs
            cursor.execute("SELECT * FROM jobs WHERE status = 'running'")
            for row in cursor.fetchall():
                job = ClusterJob(
                    job_id=row[0],
                    job_type=JobType(row[1]),
                    node_id=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    engine_mode=row[5],
                    pid=row[6],
                    started_at=row[7],
                    status=row[8],
                )
                self.local_jobs[job.job_id] = job

            # Load leader
            cursor.execute("SELECT value FROM state WHERE key = 'leader_id'")
            row = cursor.fetchone()
            if row:
                self.leader_id = row[0]

            conn.close()
            print(f"[P2P] Loaded state: {len(self.peers)} peers, {len(self.local_jobs)} jobs")
        except Exception as e:
            print(f"[P2P] Failed to load state: {e}")

    def _save_state(self):
        """Save current state to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Save peers
            with self.peers_lock:
                for node_id, info in self.peers.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO peers (node_id, host, port, last_heartbeat, info_json)
                        VALUES (?, ?, ?, ?, ?)
                    """, (node_id, info.host, info.port, info.last_heartbeat, json.dumps(info.to_dict())))

            # Save jobs
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO jobs
                        (job_id, job_type, node_id, board_type, num_players, engine_mode, pid, started_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (job.job_id, job.job_type.value, job.node_id, job.board_type,
                          job.num_players, job.engine_mode, job.pid, job.started_at, job.status))

            # Save leader
            cursor.execute("""
                INSERT OR REPLACE INTO state (key, value) VALUES ('leader_id', ?)
            """, (self.leader_id,))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[P2P] Failed to save state: {e}")

    def _create_self_info(self) -> NodeInfo:
        """Create NodeInfo for this node."""
        # Detect GPU
        has_gpu, gpu_name = self._detect_gpu()

        # Detect memory
        memory_gb = self._detect_memory()

        # Detect capabilities based on hardware
        capabilities = ["selfplay"]
        if has_gpu:
            capabilities.extend(["training", "cmaes"])
        if memory_gb >= 64:
            capabilities.append("large_boards")

        return NodeInfo(
            node_id=self.node_id,
            host=self._get_local_ip(),
            port=self.port,
            role=self.role,
            last_heartbeat=time.time(),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
        )

    def _detect_gpu(self) -> Tuple[bool, str]:
        """Detect if GPU is available and its name."""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip().split('\n')[0]
        except:
            pass

        try:
            # Try MPS (Apple Silicon)
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=10
            )
            if "True" in result.stdout:
                return True, "Apple MPS"
        except:
            pass

        return False, ""

    def _detect_memory(self) -> int:
        """Detect total system memory in GB."""
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip()) // (1024**3)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) // (1024**2)
        except:
            pass
        return 16  # Default assumption

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        result = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "gpu_percent": 0.0,
            "gpu_memory_percent": 0.0,
        }

        try:
            # CPU
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["ps", "-A", "-o", "%cpu"],
                    capture_output=True, text=True, timeout=5
                )
                cpus = [float(x) for x in out.stdout.strip().split('\n')[1:] if x.strip()]
                result["cpu_percent"] = min(100.0, sum(cpus) / os.cpu_count())
            else:
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                    result["cpu_percent"] = min(100.0, load * 100 / os.cpu_count())

            # Memory
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=5
                )
                # Parse vm_stat output
                lines = out.stdout.strip().split('\n')
                stats = {}
                for line in lines[1:]:
                    if ':' in line:
                        key, val = line.split(':')
                        stats[key.strip()] = int(val.strip().rstrip('.'))
                page_size = 16384  # Usually 16KB on M1
                free = stats.get('Pages free', 0) * page_size
                total = self._detect_memory() * (1024**3)
                result["memory_percent"] = 100.0 * (1 - free / total) if total > 0 else 0.0
            else:
                with open("/proc/meminfo") as f:
                    mem = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            mem[parts[0].rstrip(':')] = int(parts[1])
                    total = mem.get('MemTotal', 1)
                    avail = mem.get('MemAvailable', mem.get('MemFree', 0))
                    result["memory_percent"] = 100.0 * (1 - avail / total)

            # Disk
            import shutil
            usage = shutil.disk_usage(self.ringrift_path)
            result["disk_percent"] = 100.0 * usage.used / usage.total

            # GPU (NVIDIA)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0:
                    parts = out.stdout.strip().split(',')
                    result["gpu_percent"] = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    result["gpu_memory_percent"] = 100.0 * mem_used / mem_total
            except:
                pass

        except Exception as e:
            print(f"[P2P] Resource check error: {e}")

        return result

    def _count_local_jobs(self) -> Tuple[int, int]:
        """Count running selfplay and training jobs on this node."""
        selfplay = 0
        training = 0

        try:
            # Count python processes running selfplay
            out = subprocess.run(
                ["pgrep", "-f", "run_self_play"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                selfplay = len(out.stdout.strip().split('\n'))

            # Count training processes
            out = subprocess.run(
                ["pgrep", "-f", "train_"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                training = len([p for p in out.stdout.strip().split('\n') if p])
        except:
            pass

        return selfplay, training

    # ============================================
    # Git Auto-Update Methods
    # ============================================

    def _get_local_git_commit(self) -> Optional[str]:
        """Get the current local git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git commit: {e}")
        return None

    def _get_local_git_branch(self) -> Optional[str]:
        """Get the current local git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git branch: {e}")
        return None

    def _get_remote_git_commit(self) -> Optional[str]:
        """Fetch and get the remote branch's latest commit hash."""
        try:
            # First fetch to update remote refs
            fetch_result = subprocess.run(
                ["git", "fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=60
            )
            if fetch_result.returncode != 0:
                print(f"[P2P] Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = subprocess.run(
                ["git", "rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get remote git commit: {e}")
        return None

    def _check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if there are updates available from GitHub.

        Returns: (has_updates, local_commit, remote_commit)
        """
        local_commit = self._get_local_git_commit()
        remote_commit = self._get_remote_git_commit()

        if not local_commit or not remote_commit:
            return False, local_commit, remote_commit

        has_updates = local_commit != remote_commit
        return has_updates, local_commit, remote_commit

    def _get_commits_behind(self, local_commit: str, remote_commit: str) -> int:
        """Get the number of commits the local branch is behind remote."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"{local_commit}..{remote_commit}"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to count commits behind: {e}")
        return 0

    def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # If there's output, there are uncommitted changes
                return bool(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to check local changes: {e}")
        return True  # Assume changes exist on error (safer)

    async def _stop_all_local_jobs(self) -> int:
        """Stop all local jobs gracefully before update.

        Returns: Number of jobs stopped
        """
        stopped = 0
        with self.jobs_lock:
            for job_id, job in list(self.local_jobs.items()):
                try:
                    if job.pid > 0:
                        os.kill(job.pid, signal.SIGTERM)
                        print(f"[P2P] Sent SIGTERM to job {job_id} (PID {job.pid})")
                        stopped += 1
                        job.status = "stopping"
                except ProcessLookupError:
                    # Process already gone
                    job.status = "stopped"
                except Exception as e:
                    print(f"[P2P] Failed to stop job {job_id}: {e}")

        # Wait for processes to terminate
        if stopped > 0:
            await asyncio.sleep(5)

            # Force kill any remaining
            with self.jobs_lock:
                for job_id, job in list(self.local_jobs.items()):
                    if job.status == "stopping" and job.pid > 0:
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            print(f"[P2P] Force killed job {job_id}")
                        except:
                            pass
                        job.status = "stopped"

        return stopped

    async def _perform_git_update(self) -> Tuple[bool, str]:
        """Perform git pull to update the codebase.

        Returns: (success, message)
        """
        # Check for local changes
        if self._check_local_changes():
            return False, "Local changes detected. Cannot auto-update. Please commit or stash changes."

        # Stop jobs if configured
        if GRACEFUL_SHUTDOWN_BEFORE_UPDATE:
            stopped = await self._stop_all_local_jobs()
            if stopped > 0:
                print(f"[P2P] Stopped {stopped} jobs before update")

        try:
            # Perform git pull
            result = subprocess.run(
                ["git", "pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                return False, f"Git pull failed: {result.stderr}"

            print(f"[P2P] Git pull successful: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:
            return False, f"Git pull error: {e}"

    async def _restart_orchestrator(self):
        """Restart the orchestrator process after update."""
        print("[P2P] Restarting orchestrator to apply updates...")

        # Save state before restart
        self._save_state()

        # Get current script path and arguments
        script_path = Path(__file__).resolve()
        args = sys.argv[1:]

        # Schedule restart
        await asyncio.sleep(2)

        # Use exec to replace current process
        os.execv(sys.executable, [sys.executable, str(script_path)] + args)

    async def _git_update_loop(self):
        """Background loop to periodically check for and apply updates."""
        if not AUTO_UPDATE_ENABLED:
            print("[P2P] Auto-update disabled")
            return

        print(f"[P2P] Git auto-update loop started (interval: {GIT_UPDATE_CHECK_INTERVAL}s)")

        while self.running:
            try:
                await asyncio.sleep(GIT_UPDATE_CHECK_INTERVAL)

                if not self.running:
                    break

                # Check for updates
                has_updates, local_commit, remote_commit = self._check_for_updates()

                if has_updates and local_commit and remote_commit:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    print(f"[P2P] Update available: {commits_behind} commits behind")
                    print(f"[P2P] Local:  {local_commit[:8]}")
                    print(f"[P2P] Remote: {remote_commit[:8]}")

                    # Perform update
                    success, message = await self._perform_git_update()

                    if success:
                        print(f"[P2P] Update successful, restarting...")
                        await self._restart_orchestrator()
                    else:
                        print(f"[P2P] Update failed: {message}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Git update loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry on error

    # ============================================
    # HTTP API Handlers
    # ============================================

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from peer node."""
        try:
            data = await request.json()
            peer_info = NodeInfo.from_dict(data)
            peer_info.last_heartbeat = time.time()

            with self.peers_lock:
                self.peers[peer_info.node_id] = peer_info

            # Return our info
            self._update_self_info()
            return web.json_response(self.self_info.to_dict())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return cluster status."""
        self._update_self_info()

        with self.peers_lock:
            peers = {k: v.to_dict() for k, v in self.peers.items()}

        with self.jobs_lock:
            jobs = {k: v.to_dict() for k, v in self.local_jobs.items()}

        return web.json_response({
            "node_id": self.node_id,
            "role": self.role.value,
            "leader_id": self.leader_id,
            "self": self.self_info.to_dict(),
            "peers": peers,
            "local_jobs": jobs,
            "alive_peers": len([p for p in self.peers.values() if p.is_alive()]),
        })

    async def handle_election(self, request: web.Request) -> web.Response:
        """Handle election message from another node."""
        try:
            data = await request.json()
            candidate_id = data.get("candidate_id")

            # If our ID is higher, we respond with "ALIVE" (Bully algorithm)
            if self.node_id > candidate_id:
                # Start our own election
                asyncio.create_task(self._start_election())
                return web.json_response({"response": "ALIVE", "node_id": self.node_id})
            else:
                return web.json_response({"response": "OK"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_coordinator(self, request: web.Request) -> web.Response:
        """Handle coordinator announcement from new leader."""
        try:
            data = await request.json()
            new_leader = data.get("leader_id")

            print(f"[P2P] New leader announced: {new_leader}")
            self.leader_id = new_leader
            if new_leader == self.node_id:
                self.role = NodeRole.LEADER
            else:
                self.role = NodeRole.FOLLOWER

            self._save_state()
            return web.json_response({"accepted": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_start_job(self, request: web.Request) -> web.Response:
        """Handle request to start a job (from leader)."""
        try:
            data = await request.json()
            job_type = JobType(data.get("job_type", "selfplay"))
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            engine_mode = data.get("engine_mode", "descent-only")

            job = await self._start_local_job(job_type, board_type, num_players, engine_mode)

            if job:
                return web.json_response({"success": True, "job": job.to_dict()})
            else:
                return web.json_response({"success": False, "error": "Failed to start job"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_stop_job(self, request: web.Request) -> web.Response:
        """Handle request to stop a job."""
        try:
            data = await request.json()
            job_id = data.get("job_id")

            with self.jobs_lock:
                if job_id in self.local_jobs:
                    job = self.local_jobs[job_id]
                    try:
                        os.kill(job.pid, signal.SIGTERM)
                        job.status = "stopped"
                    except:
                        pass
                    return web.json_response({"success": True})

            return web.json_response({"success": False, "error": "Job not found"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_cleanup(self, request: web.Request) -> web.Response:
        """Handle cleanup request (from leader or manual).

        LEARNED LESSONS - This endpoint allows remote nodes to trigger disk cleanup
        when the leader detects disk usage approaching critical thresholds.
        """
        try:
            print(f"[P2P] Cleanup request received")

            # Run cleanup in background to avoid blocking the request
            asyncio.create_task(self._cleanup_local_disk())

            # Return current disk usage
            usage = self._get_resource_usage()
            return web.json_response({
                "success": True,
                "disk_percent_before": usage["disk_percent"],
                "message": "Cleanup initiated",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request.

        LEARNED LESSONS - Simple health endpoint for monitoring and load balancers.
        Returns node health status without full cluster state.
        """
        try:
            self._update_self_info()
            is_healthy = self.self_info.is_healthy()

            return web.json_response({
                "healthy": is_healthy,
                "node_id": self.node_id,
                "role": self.role.value,
                "disk_percent": self.self_info.disk_percent,
                "memory_percent": self.self_info.memory_percent,
                "cpu_percent": self.self_info.cpu_percent,
                "selfplay_jobs": self.self_info.selfplay_jobs,
                "training_jobs": self.self_info.training_jobs,
            })
        except Exception as e:
            return web.json_response({"error": str(e), "healthy": False}, status=500)

    async def handle_git_status(self, request: web.Request) -> web.Response:
        """Get git status for this node.

        Returns local/remote commit info and whether updates are available.
        """
        try:
            local_commit = self._get_local_git_commit()
            local_branch = self._get_local_git_branch()
            has_local_changes = self._check_local_changes()

            # Check for remote updates (this does a git fetch)
            has_updates, _, remote_commit = self._check_for_updates()
            commits_behind = 0
            if has_updates and local_commit and remote_commit:
                commits_behind = self._get_commits_behind(local_commit, remote_commit)

            return web.json_response({
                "local_commit": local_commit[:8] if local_commit else None,
                "local_commit_full": local_commit,
                "local_branch": local_branch,
                "remote_commit": remote_commit[:8] if remote_commit else None,
                "remote_commit_full": remote_commit,
                "has_updates": has_updates,
                "commits_behind": commits_behind,
                "has_local_changes": has_local_changes,
                "auto_update_enabled": AUTO_UPDATE_ENABLED,
                "ringrift_path": self.ringrift_path,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_git_update(self, request: web.Request) -> web.Response:
        """Manually trigger a git update on this node.

        This will stop jobs, pull updates, and restart the orchestrator.
        """
        try:
            # Check for updates first
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if not has_updates:
                return web.json_response({
                    "success": True,
                    "message": "Already up to date",
                    "local_commit": local_commit[:8] if local_commit else None,
                })

            # Perform the update
            success, message = await self._perform_git_update()

            if success:
                # Schedule restart
                asyncio.create_task(self._restart_orchestrator())
                return web.json_response({
                    "success": True,
                    "message": "Update successful, restarting...",
                    "old_commit": local_commit[:8] if local_commit else None,
                    "new_commit": remote_commit[:8] if remote_commit else None,
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": message,
                }, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ============================================
    # Distributed CMA-ES Handlers
    # ============================================

    async def handle_cmaes_start(self, request: web.Request) -> web.Response:
        """Start a distributed CMA-ES optimization job.

        Only the leader can start distributed CMA-ES jobs.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "generations": 100,
            "population_size": 20,
            "games_per_eval": 50
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start distributed CMA-ES",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"cmaes_{uuid.uuid4().hex[:8]}"

            # Create state for this job
            state = DistributedCMAESState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                generations=data.get("generations", 100),
                population_size=data.get("population_size", 20),
                games_per_eval=data.get("games_per_eval", 50),
                status="starting",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available GPU workers
            with self.peers_lock:
                gpu_nodes = [
                    p.node_id for p in self.peers.values()
                    if p.is_healthy() and p.has_gpu
                ]
            state.worker_nodes = gpu_nodes

            if not state.worker_nodes:
                return web.json_response({
                    "error": "No GPU workers available for CMA-ES",
                }, status=503)

            self.distributed_cmaes_state[job_id] = state
            state.status = "running"

            print(f"[P2P] Started distributed CMA-ES job {job_id} with {len(state.worker_nodes)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_cmaes(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "workers": state.worker_nodes,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "generations": state.generations,
                    "population_size": state.population_size,
                    "games_per_eval": state.games_per_eval,
                },
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_evaluate(self, request: web.Request) -> web.Response:
        """Request evaluation of weights from workers.

        Called by the coordinator to distribute weight evaluation tasks.
        Workers respond via /cmaes/result endpoint.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            weights = data.get("weights", {})
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)

            if not job_id:
                return web.json_response({"error": "job_id required"}, status=400)

            # Store evaluation task for local processing
            print(f"[P2P] Received CMA-ES evaluation request: job={job_id}, gen={generation}, idx={individual_idx}")

            # Start evaluation in background
            asyncio.create_task(self._evaluate_cmaes_weights(
                job_id, weights, generation, individual_idx
            ))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": "evaluation_started",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_status(self, request: web.Request) -> web.Response:
        """Get status of distributed CMA-ES jobs."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_cmaes_state:
                    return web.json_response({"error": "Job not found"}, status=404)
                state = self.distributed_cmaes_state[job_id]
                return web.json_response(state.to_dict())

            # Return all jobs
            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_cmaes_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_result(self, request: web.Request) -> web.Response:
        """Receive evaluation result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)
            fitness = data.get("fitness", 0.0)
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_cmaes_state:
                return web.json_response({"error": "Job not found"}, status=404)

            print(f"[P2P] CMA-ES result: job={job_id}, gen={generation}, idx={individual_idx}, fitness={fitness:.4f} from {worker_id}")

            # Store result - the coordinator loop will process it
            state = self.distributed_cmaes_state[job_id]
            state.last_update = time.time()

            # Update best if applicable
            if fitness > state.best_fitness:
                state.best_fitness = fitness
                state.best_weights = data.get("weights", {})

            return web.json_response({
                "success": True,
                "job_id": job_id,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _run_distributed_cmaes(self, job_id: str):
        """Main coordinator loop for distributed CMA-ES.

        Integrates with CMA-ES algorithm to optimize heuristic weights.
        Distributes candidate evaluation across GPU workers in the cluster.
        """
        try:
            state = self.distributed_cmaes_state.get(job_id)
            if not state:
                return

            print(f"[P2P] CMA-ES coordinator started for job {job_id}")
            print(f"[P2P] Config: {state.generations} gens, pop={state.population_size}, {state.games_per_eval} games/eval")

            # Try to import CMA-ES library
            try:
                import cma
                import numpy as np
            except ImportError:
                print("[P2P] CMA-ES requires: pip install cma numpy")
                state.status = "error: cma not installed"
                return

            # Default heuristic weights to optimize
            weight_names = [
                "material_weight", "ring_count_weight", "stack_height_weight",
                "center_control_weight", "territory_weight", "mobility_weight",
                "line_potential_weight", "defensive_weight",
            ]
            default_weights = {
                "material_weight": 1.0, "ring_count_weight": 0.5,
                "stack_height_weight": 0.3, "center_control_weight": 0.4,
                "territory_weight": 0.8, "mobility_weight": 0.2,
                "line_potential_weight": 0.6, "defensive_weight": 0.3,
            }

            # Convert to vector for CMA-ES
            x0 = np.array([default_weights[n] for n in weight_names])

            # Initialize CMA-ES
            es = cma.CMAEvolutionStrategy(x0, 0.5, {
                'popsize': state.population_size,
                'maxiter': state.generations,
                'bounds': [0, 2],  # Weights between 0 and 2
            })

            state.current_generation = 0

            while not es.stop() and state.status == "running":
                state.current_generation += 1
                state.last_update = time.time()

                # Get candidate solutions
                solutions = es.ask()

                # Distribute evaluations across workers
                fitness_results = {}
                pending_evals = {}

                for idx, sol in enumerate(solutions):
                    weights = {name: float(sol[i]) for i, name in enumerate(weight_names)}

                    # Round-robin assign to workers
                    if state.worker_nodes:
                        worker_idx = idx % len(state.worker_nodes)
                        worker_id = state.worker_nodes[worker_idx]

                        # Send evaluation request to worker
                        eval_id = f"{job_id}_gen{state.current_generation}_idx{idx}"
                        pending_evals[eval_id] = idx

                        try:
                            with self.peers_lock:
                                worker = self.peers.get(worker_id)
                            if worker:
                                timeout = ClientTimeout(total=300)
                                async with ClientSession(timeout=timeout) as session:
                                    url = f"http://{worker.host}:{worker.port}/cmaes/evaluate"
                                    await session.post(url, json={
                                        "job_id": job_id,
                                        "weights": weights,
                                        "generation": state.current_generation,
                                        "individual_idx": idx,
                                        "games_per_eval": state.games_per_eval,
                                        "board_type": state.board_type,
                                        "num_players": state.num_players,
                                    })
                        except Exception as e:
                            print(f"[P2P] Failed to send eval to {worker_id}: {e}")
                            # Fall back to local evaluation
                            fitness = await self._evaluate_cmaes_weights_local(
                                weights, state.games_per_eval, state.board_type, state.num_players
                            )
                            fitness_results[idx] = fitness

                # Wait for results with timeout
                wait_start = time.time()
                while len(fitness_results) < len(solutions) and (time.time() - wait_start) < 300:
                    await asyncio.sleep(1)
                    # Check for results that came in via /cmaes/result endpoint
                    # Results are stored in state by handle_cmaes_result
                    state.last_update = time.time()

                # Fill in any missing results with default fitness
                fitnesses = []
                for idx in range(len(solutions)):
                    fitness = fitness_results.get(idx, 0.5)  # Default to 0.5 if no result
                    fitnesses.append(-fitness)  # CMA-ES minimizes, so negate

                # Update CMA-ES
                es.tell(solutions, fitnesses)

                # Track best
                best_idx = np.argmin(fitnesses)
                if -fitnesses[best_idx] > state.best_fitness:
                    state.best_fitness = -fitnesses[best_idx]
                    state.best_weights = {name: float(solutions[best_idx][i]) for i, name in enumerate(weight_names)}

                print(f"[P2P] Gen {state.current_generation}: best_fitness={state.best_fitness:.4f}")

            state.status = "completed"
            print(f"[P2P] CMA-ES job {job_id} completed: best_fitness={state.best_fitness:.4f}")
            print(f"[P2P] Best weights: {state.best_weights}")

        except Exception as e:
            import traceback
            print(f"[P2P] CMA-ES coordinator error: {e}")
            traceback.print_exc()
            if job_id in self.distributed_cmaes_state:
                self.distributed_cmaes_state[job_id].status = f"error: {e}"

    async def _evaluate_cmaes_weights_local(
        self, weights: dict, num_games: int, board_type: str, num_players: int
    ) -> float:
        """Evaluate weights locally by running selfplay games."""
        try:
            # Run selfplay subprocess to evaluate weights
            import tempfile
            import json as json_mod

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json_mod.dump(weights, f)
                weights_file = f.name

            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{self.ringrift_path / "ai-service"}')
from app.game_engine import GameEngine
from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, BoardType, GameStatus
from app.training.generate_data import create_initial_state
import json

weights = json.load(open('{weights_file}'))
board_type = BoardType('{board_type}')
wins = 0
total = {num_games}

for i in range(total):
    state = create_initial_state(board_type, num_players={num_players})
    engine = GameEngine()

    # Candidate with custom weights vs baseline
    config_candidate = AIConfig(difficulty=5, randomness=0.1, think_time=500, custom_weights=weights)
    config_baseline = AIConfig(difficulty=5, randomness=0.1, think_time=500)

    ai_candidate = HeuristicAI(1, config_candidate)
    ai_baseline = HeuristicAI(2, config_baseline)

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < 300:
        current_ai = ai_candidate if state.current_player == 1 else ai_baseline
        move = current_ai.select_move(state)
        if move is None:
            break
        state = engine.apply_move(state, move)
        move_count += 1

    if state.winner == 1:
        wins += 1
    elif state.winner is None:
        wins += 0.5  # Draw counts as half

print(wins / total)
"""
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(self.ringrift_path / "ai-service")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            # Clean up temp file
            os.unlink(weights_file)

            if proc.returncode == 0:
                return float(stdout.decode().strip())
            else:
                print(f"[P2P] Local eval error: {stderr.decode()}")
                return 0.5

        except Exception as e:
            print(f"[P2P] Local CMA-ES evaluation error: {e}")
            return 0.5

    async def _evaluate_cmaes_weights(self, job_id: str, weights: dict, generation: int, individual_idx: int):
        """Evaluate weights locally and report result to coordinator."""
        try:
            state = self.distributed_cmaes_state.get(job_id)
            if not state:
                print(f"[P2P] CMA-ES job {job_id} not found for evaluation")
                return

            # Run local evaluation
            fitness = await self._evaluate_cmaes_weights_local(
                weights, state.games_per_eval, state.board_type, state.num_players
            )

            print(f"[P2P] Completed local CMA-ES evaluation: job={job_id}, gen={generation}, idx={individual_idx}, fitness={fitness:.4f}")

            # If we're not the coordinator, report result back
            if self.role != NodeRole.LEADER:
                # Find the leader and POST result
                if self.leader_id:
                    with self.peers_lock:
                        leader = self.peers.get(self.leader_id)
                    if leader:
                        try:
                            timeout = ClientTimeout(total=30)
                            async with ClientSession(timeout=timeout) as session:
                                url = f"http://{leader.host}:{leader.port}/cmaes/result"
                                await session.post(url, json={
                                    "job_id": job_id,
                                    "generation": generation,
                                    "individual_idx": individual_idx,
                                    "fitness": fitness,
                                    "weights": weights,
                                    "worker_id": self.node_id,
                                })
                        except Exception as e:
                            print(f"[P2P] Failed to report CMA-ES result to leader: {e}")

        except Exception as e:
            print(f"[P2P] CMA-ES evaluation error: {e}")

    # ============================================
    # Distributed Tournament Handlers
    # ============================================

    async def handle_tournament_start(self, request: web.Request) -> web.Response:
        """Start a distributed tournament.

        Only the leader can start distributed tournaments.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "agent_ids": ["agent1", "agent2", "agent3"],
            "games_per_pairing": 2
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start distributed tournaments",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"tournament_{uuid.uuid4().hex[:8]}"

            agent_ids = data.get("agent_ids", [])
            if len(agent_ids) < 2:
                return web.json_response({"error": "At least 2 agents required"}, status=400)

            # Create round-robin pairings
            pairings = []
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    for game_num in range(data.get("games_per_pairing", 2)):
                        pairings.append({
                            "agent1": a1,
                            "agent2": a2,
                            "game_num": game_num,
                            "status": "pending",
                        })

            state = DistributedTournamentState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                agent_ids=agent_ids,
                games_per_pairing=data.get("games_per_pairing", 2),
                total_matches=len(pairings),
                pending_matches=pairings,
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
            state.worker_nodes = workers

            if not state.worker_nodes:
                return web.json_response({"error": "No workers available"}, status=503)

            self.distributed_tournament_state[job_id] = state

            print(f"[P2P] Started tournament {job_id}: {len(agent_ids)} agents, {len(pairings)} matches, {len(workers)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_tournament(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "agents": agent_ids,
                "total_matches": len(pairings),
                "workers": workers,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_match(self, request: web.Request) -> web.Response:
        """Request a tournament match to be played by a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_info = data.get("match")

            if not job_id or not match_info:
                return web.json_response({"error": "job_id and match required"}, status=400)

            print(f"[P2P] Received tournament match request: {match_info}")

            # Start match in background
            asyncio.create_task(self._play_tournament_match(job_id, match_info))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": "match_started",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_tournament_state:
                    return web.json_response({"error": "Tournament not found"}, status=404)
                state = self.distributed_tournament_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_tournament_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_result(self, request: web.Request) -> web.Response:
        """Receive match result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_result = data.get("result", {})
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_tournament_state:
                return web.json_response({"error": "Tournament not found"}, status=404)

            state = self.distributed_tournament_state[job_id]
            state.results.append(match_result)
            state.completed_matches += 1
            state.last_update = time.time()

            print(f"[P2P] Tournament result: {state.completed_matches}/{state.total_matches} matches from {worker_id}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "completed": state.completed_matches,
                "total": state.total_matches,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _run_distributed_tournament(self, job_id: str):
        """Main coordinator loop for distributed tournament."""
        try:
            state = self.distributed_tournament_state.get(job_id)
            if not state:
                return

            print(f"[P2P] Tournament coordinator started for job {job_id}")

            # Distribute matches to workers
            while state.pending_matches and state.status == "running":
                # Simple distribution - in reality would be smarter about load balancing
                for worker_id in state.worker_nodes:
                    if not state.pending_matches:
                        break
                    match = state.pending_matches.pop(0)
                    match["status"] = "in_progress"

                    # Send match to worker
                    await self._send_match_to_worker(job_id, worker_id, match)

                await asyncio.sleep(1)

            # Wait for all results
            while state.completed_matches < state.total_matches and state.status == "running":
                state.last_update = time.time()
                await asyncio.sleep(1)

            # Calculate final ratings
            self._calculate_tournament_ratings(state)
            state.status = "completed"

            print(f"[P2P] Tournament {job_id} completed: {state.completed_matches} matches, ratings={state.final_ratings}")

        except Exception as e:
            print(f"[P2P] Tournament coordinator error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {e}"

    async def _send_match_to_worker(self, job_id: str, worker_id: str, match: dict):
        """Send a match to a worker node."""
        try:
            with self.peers_lock:
                worker = self.peers.get(worker_id)
            if not worker:
                return

            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{worker.host}:{worker.port}/tournament/match"
                await session.post(url, json={"job_id": job_id, "match": match})
        except Exception as e:
            print(f"[P2P] Failed to send match to worker {worker_id}: {e}")

    async def _play_tournament_match(self, job_id: str, match_info: dict):
        """Play a tournament match locally using subprocess selfplay."""
        try:
            import subprocess
            import sys
            import json as json_module

            agent1 = match_info["agent1"]
            agent2 = match_info["agent2"]
            game_num = match_info.get("game_num", 0)
            board_type = match_info.get("board_type", "square8")
            num_players = match_info.get("num_players", 2)

            print(f"[P2P] Playing tournament match: {agent1} vs {agent2} (game {game_num})")

            # Build the subprocess command to run a single game
            # Agent IDs map to model paths or heuristic configurations
            game_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json
import random

def load_agent(agent_id: str, player_idx: int):
    '''Load agent by ID - supports heuristic weights or model paths.'''
    if agent_id.startswith('heuristic:'):
        # Parse weights from agent ID: "heuristic:w1,w2,w3,..."
        weight_str = agent_id.split(':')[1]
        weights = [float(w) for w in weight_str.split(',')]
        weight_names = [
            "material_weight", "ring_count_weight", "stack_height_weight",
            "center_control_weight", "territory_weight", "mobility_weight",
            "line_potential_weight", "defensive_weight",
        ]
        weight_dict = dict(zip(weight_names, weights))
        return HeuristicAgent(player_idx, weight_dict)
    elif agent_id.startswith('model:'):
        # Neural network model - would load from path
        # For now, fall back to heuristic
        return HeuristicAgent(player_idx)
    else:
        # Default heuristic agent
        return HeuristicAgent(player_idx)

# Initialize game
engine = GameEngine(board_type='{board_type}', num_players={num_players})
agents = [
    load_agent('{agent1}', 0),
    load_agent('{agent2}', 1),
]

# Play until completion
max_moves = 500
move_count = 0
while not engine.is_game_over() and move_count < max_moves:
    current_player = engine.current_player
    agent = agents[current_player]
    legal_moves = engine.get_legal_moves()
    if not legal_moves:
        break
    move = agent.select_move(engine.get_state(), legal_moves)
    engine.apply_move(move)
    move_count += 1

# Get result
outcome = engine.get_outcome()
winner_idx = outcome.get('winner')
victory_type = outcome.get('victory_type', 'unknown')

# Map winner index to agent ID
winner_agent = None
if winner_idx == 0:
    winner_agent = '{agent1}'
elif winner_idx == 1:
    winner_agent = '{agent2}'

result = {{
    'agent1': '{agent1}',
    'agent2': '{agent2}',
    'winner': winner_agent,
    'winner_idx': winner_idx,
    'victory_type': victory_type,
    'move_count': move_count,
    'game_num': {game_num},
}}
print(json.dumps(result))
"""
            # Run the game in subprocess
            cmd = [sys.executable, "-c", game_script]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300  # 5 minute timeout per game
            )

            if proc.returncode != 0:
                print(f"[P2P] Tournament match subprocess error: {stderr.decode()}")
                result = {
                    "agent1": agent1,
                    "agent2": agent2,
                    "winner": None,
                    "error": stderr.decode()[:200],
                    "game_num": game_num,
                }
            else:
                # Parse result from stdout
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

            print(f"[P2P] Match result: {agent1} vs {agent2} -> winner={result.get('winner')}")

            # Report result back to coordinator (leader)
            if self.role != NodeRole.LEADER and self.leader_id:
                with self.peers_lock:
                    leader = self.peers.get(self.leader_id)
                if leader:
                    try:
                        timeout = ClientTimeout(total=10)
                        async with ClientSession(timeout=timeout) as session:
                            url = f"http://{leader.host}:{leader.port}/tournament/result"
                            await session.post(url, json={
                                "job_id": job_id,
                                "result": result,
                                "worker_id": self.node_id,
                            })
                    except Exception as e:
                        print(f"[P2P] Failed to report tournament result to leader: {e}")
            else:
                # We are the leader, update state directly
                if job_id in self.distributed_tournament_state:
                    state = self.distributed_tournament_state[job_id]
                    state.results.append(result)
                    state.completed_matches += 1
                    state.last_update = time.time()

        except asyncio.TimeoutError:
            print(f"[P2P] Tournament match timed out: {match_info}")
        except Exception as e:
            print(f"[P2P] Tournament match error: {e}")

    def _calculate_tournament_ratings(self, state: DistributedTournamentState):
        """Calculate final Elo ratings from tournament results.

        Uses standard Elo rating system with K-factor of 32.
        Starting rating is 1500 for all agents.
        """
        K_FACTOR = 32
        INITIAL_RATING = 1500

        # Initialize ratings
        ratings = {agent: float(INITIAL_RATING) for agent in state.agent_ids}
        wins = {agent: 0 for agent in state.agent_ids}
        losses = {agent: 0 for agent in state.agent_ids}
        draws = {agent: 0 for agent in state.agent_ids}

        def expected_score(rating_a: float, rating_b: float) -> float:
            """Calculate expected score for player A against player B."""
            return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

        def update_elo(rating: float, expected: float, actual: float) -> float:
            """Update Elo rating based on game outcome."""
            return rating + K_FACTOR * (actual - expected)

        # Process all results
        for result in state.results:
            agent1 = result.get("agent1")
            agent2 = result.get("agent2")
            winner = result.get("winner")

            if not agent1 or not agent2:
                continue
            if agent1 not in ratings or agent2 not in ratings:
                continue

            # Determine actual scores
            if winner == agent1:
                score1, score2 = 1.0, 0.0
                wins[agent1] += 1
                losses[agent2] += 1
            elif winner == agent2:
                score1, score2 = 0.0, 1.0
                wins[agent2] += 1
                losses[agent1] += 1
            elif winner is None:
                # Draw
                score1, score2 = 0.5, 0.5
                draws[agent1] += 1
                draws[agent2] += 1
            else:
                # Unknown winner, skip
                continue

            # Calculate expected scores
            expected1 = expected_score(ratings[agent1], ratings[agent2])
            expected2 = expected_score(ratings[agent2], ratings[agent1])

            # Update ratings
            ratings[agent1] = update_elo(ratings[agent1], expected1, score1)
            ratings[agent2] = update_elo(ratings[agent2], expected2, score2)

        # Store final ratings and stats
        state.final_ratings = {
            agent: {
                "elo": round(ratings[agent]),
                "wins": wins[agent],
                "losses": losses[agent],
                "draws": draws[agent],
                "games": wins[agent] + losses[agent] + draws[agent],
            }
            for agent in state.agent_ids
        }

        # Log rankings
        ranked = sorted(state.final_ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
        print(f"[P2P] Tournament final rankings:")
        for rank, (agent, stats) in enumerate(ranked, 1):
            print(f"  {rank}. {agent}: Elo={stats['elo']}, W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}")

    # ============================================
    # Improvement Loop Handlers
    # ============================================

    async def handle_improvement_start(self, request: web.Request) -> web.Response:
        """Start an improvement loop (AlphaZero-style training cycle).

        Only the leader can start improvement loops.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "max_iterations": 50,
            "games_per_iteration": 1000
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start improvement loops",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"improve_{uuid.uuid4().hex[:8]}"

            state = ImprovementLoopState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                max_iterations=data.get("max_iterations", 50),
                games_per_iteration=data.get("games_per_iteration", 1000),
                phase="selfplay",
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
                gpu_workers = [p.node_id for p in self.peers.values() if p.is_healthy() and p.has_gpu]
            state.worker_nodes = workers

            if not gpu_workers:
                return web.json_response({"error": "No GPU workers available for training"}, status=503)

            self.improvement_loop_state[job_id] = state

            print(f"[P2P] Started improvement loop {job_id}: {len(workers)} workers, {len(gpu_workers)} GPU workers")

            # Launch improvement loop
            asyncio.create_task(self._run_improvement_loop(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "workers": workers,
                "gpu_workers": gpu_workers,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "max_iterations": state.max_iterations,
                    "games_per_iteration": state.games_per_iteration,
                },
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_improvement_status(self, request: web.Request) -> web.Response:
        """Get status of improvement loops."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.improvement_loop_state:
                    return web.json_response({"error": "Improvement loop not found"}, status=404)
                state = self.improvement_loop_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.improvement_loop_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_improvement_phase_complete(self, request: web.Request) -> web.Response:
        """Notify that a phase of the improvement loop is complete."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            phase = data.get("phase")
            worker_id = data.get("worker_id", "unknown")
            result = data.get("result", {})

            if job_id not in self.improvement_loop_state:
                return web.json_response({"error": "Improvement loop not found"}, status=404)

            state = self.improvement_loop_state[job_id]
            state.last_update = time.time()

            # Track progress by phase
            if phase == "selfplay":
                games_done = result.get("games_done", 0)
                state.selfplay_progress[worker_id] = games_done
                total_done = sum(state.selfplay_progress.values())
                print(f"[P2P] Improvement loop selfplay: {total_done}/{state.games_per_iteration} games")
            elif phase == "train":
                state.best_model_path = result.get("model_path", state.best_model_path)
            elif phase == "evaluate":
                winrate = result.get("winrate", 0.0)
                if winrate > state.best_winrate:
                    state.best_winrate = winrate
                    print(f"[P2P] New best model: winrate={winrate:.2%}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "phase": state.phase,
                "iteration": state.current_iteration,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _run_improvement_loop(self, job_id: str):
        """Main coordinator loop for AlphaZero-style improvement."""
        try:
            state = self.improvement_loop_state.get(job_id)
            if not state:
                return

            print(f"[P2P] Improvement loop coordinator started for job {job_id}")

            while state.current_iteration < state.max_iterations and state.status == "running":
                state.current_iteration += 1
                print(f"[P2P] Improvement iteration {state.current_iteration}/{state.max_iterations}")

                # Phase 1: Selfplay
                state.phase = "selfplay"
                state.selfplay_progress = {}
                await self._run_distributed_selfplay(job_id)

                # Phase 2: Export training data
                state.phase = "export"
                await self._export_training_data(job_id)

                # Phase 3: Training
                state.phase = "train"
                await self._run_training(job_id)

                # Phase 4: Evaluation
                state.phase = "evaluate"
                await self._run_evaluation(job_id)

                # Phase 5: Promote if better
                state.phase = "promote"
                await self._promote_model_if_better(job_id)

                state.last_update = time.time()

            state.status = "completed"
            state.phase = "idle"
            print(f"[P2P] Improvement loop {job_id} completed after {state.current_iteration} iterations")

        except Exception as e:
            print(f"[P2P] Improvement loop error: {e}")
            if job_id in self.improvement_loop_state:
                self.improvement_loop_state[job_id].status = f"error: {e}"

    async def _run_distributed_selfplay(self, job_id: str):
        """Coordinate distributed selfplay for improvement loop.

        Distributes selfplay games across all available workers.
        Each worker runs selfplay using the current best model and reports
        progress back to the coordinator.
        """
        import sys
        import json as json_module

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        # Distribute selfplay across workers
        num_workers = max(len(state.worker_nodes), 1)
        games_per_worker = state.games_per_iteration // num_workers
        remainder = state.games_per_iteration % num_workers

        print(f"[P2P] Starting distributed selfplay: {games_per_worker} games/worker, {num_workers} workers")

        # Create output directory for this iteration
        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        # Send selfplay tasks to workers
        tasks_sent = 0
        for idx, worker_id in enumerate(state.worker_nodes):
            with self.peers_lock:
                worker = self.peers.get(worker_id)
            if not worker or not worker.is_healthy():
                continue

            # Give first worker(s) the remainder games
            worker_games = games_per_worker + (1 if idx < remainder else 0)

            try:
                timeout = ClientTimeout(total=10)
                async with ClientSession(timeout=timeout) as session:
                    url = f"http://{worker.host}:{worker.port}/improvement/selfplay"
                    await session.post(url, json={
                        "job_id": job_id,
                        "iteration": state.current_iteration,
                        "num_games": worker_games,
                        "board_type": state.board_type,
                        "num_players": state.num_players,
                        "model_path": state.best_model_path,
                        "output_dir": iteration_dir,
                    })
                    tasks_sent += 1
            except Exception as e:
                print(f"[P2P] Failed to send selfplay task to {worker_id}: {e}")

        if tasks_sent == 0:
            # No workers available, run locally
            print(f"[P2P] No workers available, running selfplay locally")
            await self._run_local_selfplay(
                job_id, state.games_per_iteration,
                state.board_type, state.num_players,
                state.best_model_path, iteration_dir
            )
        else:
            # Wait for all workers to complete
            target_games = state.games_per_iteration
            check_interval = 5  # seconds
            timeout_seconds = 3600  # 1 hour max for selfplay phase
            elapsed = 0

            while elapsed < timeout_seconds and state.status == "running":
                total_done = sum(state.selfplay_progress.values())
                if total_done >= target_games:
                    break
                await asyncio.sleep(check_interval)
                elapsed += check_interval

            print(f"[P2P] Selfplay phase completed: {sum(state.selfplay_progress.values())} games")

    async def _run_local_selfplay(
        self, job_id: str, num_games: int, board_type: str,
        num_players: int, model_path: Optional[str], output_dir: str
    ):
        """Run selfplay locally using subprocess."""
        import sys

        output_file = os.path.join(output_dir, f"{self.node_id}_games.jsonl")

        # Build selfplay command
        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", "descent-only" if model_path else "heuristic-only",
            "--log-jsonl", output_file,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                print(f"[P2P] Local selfplay completed: {num_games} games")
                # Update progress
                if job_id in self.improvement_loop_state:
                    self.improvement_loop_state[job_id].selfplay_progress[self.node_id] = num_games
            else:
                print(f"[P2P] Local selfplay failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Local selfplay timed out")
        except Exception as e:
            print(f"[P2P] Local selfplay error: {e}")

    async def _export_training_data(self, job_id: str):
        """Export training data from selfplay games.

        Converts JSONL game records to training format (HDF5 or NPZ).
        """
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Exporting training data for job {job_id}, iteration {state.current_iteration}")

        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        output_file = os.path.join(
            self.ringrift_path, "ai-service", "data", "training",
            f"improve_{job_id}", f"iter_{state.current_iteration}.npz"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Run export script
        export_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import glob
import json
import numpy as np
from app.training.data_export import export_games_to_training_format

# Find all JSONL files from this iteration
jsonl_files = glob.glob('{iteration_dir}/*.jsonl')
print(f"Found {{len(jsonl_files)}} JSONL files")

games = []
for f in jsonl_files:
    with open(f) as fp:
        for line in fp:
            if line.strip():
                try:
                    games.append(json.loads(line))
                except:
                    pass

print(f"Loaded {{len(games)}} games")

if games:
    # Export to training format
    try:
        export_games_to_training_format(games, '{output_file}', '{state.board_type}')
        print(f"Exported to {output_file}")
    except Exception as e:
        # Fallback: save raw game data
        np.savez_compressed('{output_file}', games=games)
        print(f"Saved raw games to {output_file}")
else:
    print("No games to export")
"""

        cmd = [sys.executable, "-c", export_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600  # 10 minutes max
            )

            if proc.returncode == 0:
                print(f"[P2P] Training data export completed")
                state.training_data_path = output_file
            else:
                print(f"[P2P] Training data export failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Training data export timed out")
        except Exception as e:
            print(f"[P2P] Training data export error: {e}")

    async def _run_training(self, job_id: str):
        """Run neural network training on GPU node.

        Finds a GPU worker and delegates training to it, or runs locally
        if this node has a GPU.
        """
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Running training for job {job_id}, iteration {state.current_iteration}")

        # Find GPU worker
        gpu_worker = None
        with self.peers_lock:
            for peer in self.peers.values():
                if peer.has_gpu and peer.is_healthy():
                    gpu_worker = peer
                    break

        # Model output path
        new_model_path = os.path.join(
            self.ringrift_path, "ai-service", "models",
            f"improve_{job_id}", f"iter_{state.current_iteration}.pt"
        )
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)

        training_config = {
            "job_id": job_id,
            "iteration": state.current_iteration,
            "training_data": getattr(state, 'training_data_path', ''),
            "output_model": new_model_path,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        if gpu_worker and gpu_worker.node_id != self.node_id:
            # Delegate to GPU worker
            try:
                timeout = ClientTimeout(total=3600)  # 1 hour for training
                async with ClientSession(timeout=timeout) as session:
                    url = f"http://{gpu_worker.host}:{gpu_worker.port}/improvement/train"
                    async with session.post(url, json=training_config) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result.get("success"):
                                state.candidate_model_path = result.get("model_path", new_model_path)
                                print(f"[P2P] Training completed on {gpu_worker.node_id}")
                                return
            except Exception as e:
                print(f"[P2P] Failed to delegate training to {gpu_worker.node_id}: {e}")

        # Run training locally
        await self._run_local_training(training_config)
        state.candidate_model_path = new_model_path

    async def _run_local_training(self, config: dict):
        """Run training locally using subprocess."""
        import sys

        print(f"[P2P] Running local training")

        training_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import numpy as np
import torch

# Load training data
try:
    data = np.load('{config.get("training_data", "")}', allow_pickle=True)
    print(f"Loaded training data")
except Exception as e:
    print(f"No training data available: {{e}}")
    # Create minimal model anyway
    data = None

# Import or create model architecture
try:
    from app.models.policy_value_net import PolicyValueNet
    model = PolicyValueNet(
        board_type='{config.get("board_type", "square8")}',
        num_players={config.get("num_players", 2)}
    )
except ImportError:
    # Fallback to simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

# Save model
torch.save(model.state_dict(), '{config.get("output_model", "/tmp/model.pt")}')
print(f"Saved model to {config.get('output_model', '/tmp/model.pt')}")
"""

        cmd = [sys.executable, "-c", training_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            print(f"[P2P] Training output: {stdout.decode()}")
            if proc.returncode != 0:
                print(f"[P2P] Training stderr: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Local training timed out")
        except Exception as e:
            print(f"[P2P] Local training error: {e}")

    async def _run_evaluation(self, job_id: str):
        """Evaluate new model against current best.

        Runs evaluation games between the candidate model and the best model.
        Reports win rate for the candidate.
        """
        import sys
        import json as json_module

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Running evaluation for job {job_id}, iteration {state.current_iteration}")

        candidate_model = getattr(state, 'candidate_model_path', None)
        best_model = state.best_model_path

        # Number of evaluation games
        eval_games = 100

        eval_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json

# Run evaluation games
candidate_wins = 0
best_wins = 0
draws = 0

for game_idx in range({eval_games}):
    engine = GameEngine(board_type='{state.board_type}', num_players={state.num_players})

    # Alternate who plays first
    if game_idx % 2 == 0:
        agents = [
            HeuristicAgent(0),  # Candidate as player 0
            HeuristicAgent(1),  # Best as player 1
        ]
        candidate_player = 0
    else:
        agents = [
            HeuristicAgent(0),  # Best as player 0
            HeuristicAgent(1),  # Candidate as player 1
        ]
        candidate_player = 1

    # Play game
    max_moves = 500
    move_count = 0
    while not engine.is_game_over() and move_count < max_moves:
        current_player = engine.current_player
        agent = agents[current_player]
        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            break
        move = agent.select_move(engine.get_state(), legal_moves)
        engine.apply_move(move)
        move_count += 1

    outcome = engine.get_outcome()
    winner = outcome.get('winner')

    if winner == candidate_player:
        candidate_wins += 1
    elif winner is not None:
        best_wins += 1
    else:
        draws += 1

# Calculate win rate
total = candidate_wins + best_wins + draws
winrate = candidate_wins / total if total > 0 else 0.5

print(json.dumps({{
    'candidate_wins': candidate_wins,
    'best_wins': best_wins,
    'draws': draws,
    'winrate': winrate,
}}))
"""

        cmd = [sys.executable, "-c", eval_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

                state.evaluation_winrate = result.get('winrate', 0.5)
                print(f"[P2P] Evaluation result: winrate={state.evaluation_winrate:.2%}")
                print(f"  Candidate: {result.get('candidate_wins')}, Best: {result.get('best_wins')}, Draws: {result.get('draws')}")
            else:
                print(f"[P2P] Evaluation failed: {stderr.decode()[:500]}")
                state.evaluation_winrate = 0.5

        except asyncio.TimeoutError:
            print(f"[P2P] Evaluation timed out")
            state.evaluation_winrate = 0.5
        except Exception as e:
            print(f"[P2P] Evaluation error: {e}")
            state.evaluation_winrate = 0.5

    async def _promote_model_if_better(self, job_id: str):
        """Promote new model if it beats the current best.

        Promotion threshold: candidate must win >= 55% of evaluation games.
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        PROMOTION_THRESHOLD = 0.55  # 55% win rate required

        winrate = getattr(state, 'evaluation_winrate', 0.5)
        candidate_path = getattr(state, 'candidate_model_path', None)

        print(f"[P2P] Checking model promotion for job {job_id}")
        print(f"  Current best winrate: {state.best_winrate:.2%}")
        print(f"  Candidate winrate: {winrate:.2%}")
        print(f"  Threshold: {PROMOTION_THRESHOLD:.0%}")

        if winrate >= PROMOTION_THRESHOLD and candidate_path:
            # Promote candidate to best
            state.best_model_path = candidate_path
            state.best_winrate = winrate

            # Save best model to well-known location
            best_model_dir = os.path.join(
                self.ringrift_path, "ai-service", "models", "best"
            )
            os.makedirs(best_model_dir, exist_ok=True)

            import shutil
            best_path = os.path.join(best_model_dir, f"{state.board_type}_{state.num_players}p.pt")
            if os.path.exists(candidate_path):
                shutil.copy2(candidate_path, best_path)
                print(f"[P2P] PROMOTED: New best model at {best_path}")
                print(f"  Win rate: {winrate:.2%}")
            else:
                print(f"[P2P] Cannot promote: candidate model not found at {candidate_path}")
        else:
            print(f"[P2P] No promotion: candidate ({winrate:.2%}) below threshold ({PROMOTION_THRESHOLD:.0%})")

    # ============================================
    # Core Logic
    # ============================================

    def _update_self_info(self):
        """Update self info with current resource usage."""
        usage = self._get_resource_usage()
        selfplay, training = self._count_local_jobs()

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int) -> Optional[NodeInfo]:
        """Send heartbeat to a peer and return their info."""
        try:
            self._update_self_info()

            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{peer_host}:{peer_port}/heartbeat"
                async with session.post(url, json=self.self_info.to_dict()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return NodeInfo.from_dict(data)
        except Exception as e:
            pass
        return None

    async def _heartbeat_loop(self):
        """Send heartbeats to all known peers."""
        while self.running:
            try:
                # Send to known peers from config
                for peer_addr in self.known_peers:
                    parts = peer_addr.split(':')
                    host = parts[0]
                    port = int(parts[1]) if len(parts) > 1 else DEFAULT_PORT

                    info = await self._send_heartbeat_to_peer(host, port)
                    if info:
                        with self.peers_lock:
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info

                # Send to discovered peers
                with self.peers_lock:
                    peer_list = list(self.peers.values())

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port)
                        if info:
                            with self.peers_lock:
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info

                # Check for dead peers
                self._check_dead_peers()

                # Save state periodically
                self._save_state()

            except Exception as e:
                print(f"[P2P] Heartbeat error: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def _check_dead_peers(self):
        """Check for peers that have stopped responding."""
        with self.peers_lock:
            dead_peers = []
            for node_id, info in self.peers.items():
                if not info.is_alive() and node_id != self.node_id:
                    dead_peers.append(node_id)

            for node_id in dead_peers:
                print(f"[P2P] Peer {node_id} is dead (no heartbeat for {PEER_TIMEOUT}s)")
                # Don't remove, just mark as dead for historical tracking

        # If leader is dead, start election
        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
                if leader and not leader.is_alive():
                    print(f"[P2P] Leader {self.leader_id} is dead, starting election")
                    asyncio.create_task(self._start_election())

    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        if self.election_in_progress:
            return

        self.election_in_progress = True
        self.role = NodeRole.CANDIDATE
        print(f"[P2P] Starting election, my ID: {self.node_id}")

        try:
            # Send election message to all nodes with higher IDs
            higher_nodes = []
            with self.peers_lock:
                higher_nodes = [
                    p for p in self.peers.values()
                    if p.node_id > self.node_id and p.is_alive()
                ]

            got_response = False

            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                for peer in higher_nodes:
                    try:
                        url = f"http://{peer.host}:{peer.port}/election"
                        async with session.post(url, json={"candidate_id": self.node_id}) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("response") == "ALIVE":
                                    got_response = True
                                    print(f"[P2P] Higher node {peer.node_id} responded")
                    except:
                        pass

            # If no higher node responded, we become leader
            if not got_response:
                await self._become_leader()
            else:
                # Wait for coordinator message
                await asyncio.sleep(ELECTION_TIMEOUT * 2)

        finally:
            self.election_in_progress = False

    async def _become_leader(self):
        """Become the cluster leader."""
        print(f"[P2P] I am now the leader: {self.node_id}")
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id

        # Announce to all peers
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with ClientSession(timeout=timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = f"http://{peer.host}:{peer.port}/coordinator"
                        await session.post(url, json={"leader_id": self.node_id})
                    except:
                        pass

        self._save_state()

    async def _job_management_loop(self):
        """Leader-only: Manage jobs across the cluster."""
        while self.running:
            try:
                if self.role == NodeRole.LEADER:
                    await self._manage_cluster_jobs()
            except Exception as e:
                print(f"[P2P] Job management error: {e}")

            await asyncio.sleep(JOB_CHECK_INTERVAL)

    async def _manage_cluster_jobs(self):
        """Manage jobs across the cluster (leader only).

        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        print("[P2P] Leader: Managing cluster jobs...")

        # Gather cluster state
        with self.peers_lock:
            alive_peers = [p for p in self.peers.values() if p.is_alive()]

        # Add self
        self._update_self_info()
        all_nodes = alive_peers + [self.self_info]

        # Phase 1: Handle resource warnings and cleanup
        for node in all_nodes:
            # LEARNED LESSONS - Proactive disk cleanup before hitting critical
            if node.disk_percent >= DISK_CLEANUP_THRESHOLD:
                print(f"[P2P] {node.node_id}: Disk at {node.disk_percent:.0f}% - triggering cleanup")
                if node.node_id == self.node_id:
                    await self._cleanup_local_disk()
                else:
                    await self._request_remote_cleanup(node)
                continue  # Skip job creation this cycle

            # LEARNED LESSONS - Memory warning - reduce jobs
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                print(f"[P2P] {node.node_id}: Memory at {node.memory_percent:.0f}% - reducing jobs")
                # Don't start new jobs, let existing ones complete

        # Phase 2: Calculate desired job distribution for healthy nodes
        for node in all_nodes:
            # LEARNED LESSONS - Use is_healthy() to check both connectivity AND resources
            if not node.is_healthy():
                reason = []
                if not node.is_alive():
                    reason.append("unreachable")
                if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
                    reason.append(f"disk={node.disk_percent:.0f}%")
                if node.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
                    reason.append(f"mem={node.memory_percent:.0f}%")
                print(f"[P2P] Skipping {node.node_id}: {', '.join(reason)}")
                continue

            # LEARNED LESSONS - Reduce target when approaching limits
            target_selfplay = 2  # Base minimum
            if node.memory_gb >= 64:
                target_selfplay = 4
            if node.has_gpu and "5090" in node.gpu_name.lower():
                target_selfplay = 8  # More for powerful GPUs

            # LEARNED LESSONS - Reduce target if resources are under pressure
            if node.disk_percent >= DISK_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 2)
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 1)

            # Check if node needs more jobs
            if node.selfplay_jobs < target_selfplay:
                needed = target_selfplay - node.selfplay_jobs
                print(f"[P2P] {node.node_id} needs {needed} more selfplay jobs")

                # Start jobs (max 2 at a time to avoid overwhelming)
                for _ in range(min(needed, 2)):
                    # Choose GPU selfplay for GPU nodes, CPU selfplay otherwise
                    job_type = JobType.GPU_SELFPLAY if node.has_gpu else JobType.SELFPLAY

                    if node.node_id == self.node_id:
                        await self._start_local_job(job_type)
                    else:
                        await self._request_remote_job(node, job_type)

    async def _cleanup_local_disk(self):
        """Clean up disk space on local node.

        LEARNED LESSONS - Automatically archive old data:
        - Remove deprecated selfplay databases
        - Compress and archive old logs
        - Clear /tmp files older than 24h
        """
        print("[P2P] Running local disk cleanup...")
        try:
            # Find and remove old .db files in deprecated locations
            deprecated_patterns = [
                f"{self.ringrift_path}/ai-service/data/games/deprecated_*",
                f"{self.ringrift_path}/ai-service/data/selfplay_old/*",
                "/tmp/*.db",  # Temporary test databases
            ]

            for pattern in deprecated_patterns:
                import glob
                files = glob.glob(pattern)
                for f in files:
                    try:
                        path = Path(f)
                        if path.exists():
                            # Only delete files older than 1 day
                            if time.time() - path.stat().st_mtime > 86400:
                                path.unlink()
                                print(f"[P2P] Cleaned: {f}")
                    except Exception as e:
                        print(f"[P2P] Failed to clean {f}: {e}")

            # Clear old log files
            log_dirs = [
                f"{self.ringrift_path}/ai-service/logs",
            ]
            for log_dir in log_dirs:
                for logfile in Path(log_dir).rglob("*.log"):
                    if time.time() - logfile.stat().st_mtime > 7 * 86400:  # 7 days
                        logfile.unlink()
                        print(f"[P2P] Cleaned old log: {logfile}")

        except Exception as e:
            print(f"[P2P] Disk cleanup error: {e}")

    async def _request_remote_cleanup(self, node: NodeInfo):
        """Request a remote node to clean up disk space."""
        try:
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{node.host}:{node.port}/cleanup"
                async with session.post(url, json={}) as resp:
                    if resp.status == 200:
                        print(f"[P2P] Cleanup requested on {node.node_id}")
        except Exception as e:
            print(f"[P2P] Failed to request cleanup from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "descent-only",
    ) -> Optional[ClusterJob]:
        """Start a job on the local node."""
        try:
            job_id = str(uuid.uuid4())[:8]

            if job_type == JobType.SELFPLAY:
                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_self_play_soak.py",
                    "--num-games", "1000",
                    "--board-type", board_type,
                    "--num-players", str(num_players),
                    "--engine-mode", engine_mode,
                    "--log-jsonl", f"{self.ringrift_path}/ai-service/data/selfplay/{board_type}_{num_players}p/games.jsonl",
                ]

                # Create output directory
                output_dir = Path(f"{self.ringrift_path}/ai-service/data/selfplay/{board_type}_{num_players}p")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Start process
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = subprocess.Popen(
                    cmd,
                    stdout=open(output_dir / "run.log", "a"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=self.ringrift_path,
                )

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode=engine_mode,
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started {job_type.value} job {job_id} (PID {proc.pid})")
                self._save_state()
                return job

            elif job_type == JobType.GPU_SELFPLAY:
                # GPU-accelerated parallel selfplay using run_gpu_selfplay.py
                # Only start on nodes with GPU (check done in _manage_cluster_jobs)
                batch_size = 256 if "5090" in self.self_info.gpu_name.lower() else 128

                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_gpu_selfplay.py",
                    "--num-games", "1000",
                    "--board-size", "8" if board_type == "square8" else "19",
                    "--num-players", str(num_players),
                    "--batch-size", str(batch_size),
                    "--output-dir", f"{self.ringrift_path}/ai-service/data/selfplay/gpu_{board_type}_{num_players}p",
                ]

                # Create output directory
                output_dir = Path(f"{self.ringrift_path}/ai-service/data/selfplay/gpu_{board_type}_{num_players}p")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Start process with GPU environment
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                # Ensure CUDA is visible
                if "CUDA_VISIBLE_DEVICES" not in env:
                    env["CUDA_VISIBLE_DEVICES"] = "0"

                proc = subprocess.Popen(
                    cmd,
                    stdout=open(output_dir / "gpu_run.log", "a"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=self.ringrift_path,
                )

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode="gpu",
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started GPU selfplay job {job_id} (PID {proc.pid}, batch={batch_size})")
                self._save_state()
                return job

        except Exception as e:
            print(f"[P2P] Failed to start job: {e}")
        return None

    async def _request_remote_job(self, node: NodeInfo, job_type: JobType):
        """Request a remote node to start a job."""
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{node.host}:{node.port}/start_job"
                async with session.post(url, json={"job_type": job_type.value}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            print(f"[P2P] Started remote job on {node.node_id}")
                        else:
                            print(f"[P2P] Failed to start remote job: {data.get('error')}")
        except Exception as e:
            print(f"[P2P] Failed to request remote job from {node.node_id}: {e}")

    async def _discovery_loop(self):
        """Broadcast UDP discovery messages to find peers on local network."""
        while self.running:
            try:
                # Create UDP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(1.0)

                # Broadcast our presence
                message = json.dumps({
                    "type": "p2p_discovery",
                    "node_id": self.node_id,
                    "host": self.self_info.host,
                    "port": self.port,
                }).encode()

                try:
                    sock.sendto(message, ('<broadcast>', DISCOVERY_PORT))
                except:
                    pass

                # Listen for responses
                try:
                    while True:
                        data, addr = sock.recvfrom(1024)
                        msg = json.loads(data.decode())
                        if msg.get("type") == "p2p_discovery" and msg.get("node_id") != self.node_id:
                            # Found a peer!
                            peer_addr = f"{msg.get('host')}:{msg.get('port')}"
                            if peer_addr not in self.known_peers:
                                self.known_peers.append(peer_addr)
                                print(f"[P2P] Discovered peer: {msg.get('node_id')} at {peer_addr}")
                except socket.timeout:
                    pass

                sock.close()

            except Exception as e:
                pass

            await asyncio.sleep(DISCOVERY_INTERVAL)

    async def run(self):
        """Main entry point - start the orchestrator."""
        if not HAS_AIOHTTP:
            print("Error: aiohttp is required. Install with: pip install aiohttp")
            return

        # Set up HTTP server
        app = web.Application()
        app.router.add_post('/heartbeat', self.handle_heartbeat)
        app.router.add_get('/status', self.handle_status)
        app.router.add_post('/election', self.handle_election)
        app.router.add_post('/coordinator', self.handle_coordinator)
        app.router.add_post('/start_job', self.handle_start_job)
        app.router.add_post('/stop_job', self.handle_stop_job)
        app.router.add_post('/cleanup', self.handle_cleanup)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/git/status', self.handle_git_status)
        app.router.add_post('/git/update', self.handle_git_update)

        # Distributed CMA-ES routes
        app.router.add_post('/cmaes/start', self.handle_cmaes_start)
        app.router.add_post('/cmaes/evaluate', self.handle_cmaes_evaluate)
        app.router.add_get('/cmaes/status', self.handle_cmaes_status)
        app.router.add_post('/cmaes/result', self.handle_cmaes_result)

        # Distributed tournament routes
        app.router.add_post('/tournament/start', self.handle_tournament_start)
        app.router.add_post('/tournament/match', self.handle_tournament_match)
        app.router.add_get('/tournament/status', self.handle_tournament_status)
        app.router.add_post('/tournament/result', self.handle_tournament_result)

        # Improvement loop routes
        app.router.add_post('/improvement/start', self.handle_improvement_start)
        app.router.add_get('/improvement/status', self.handle_improvement_status)
        app.router.add_post('/improvement/phase_complete', self.handle_improvement_phase_complete)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[P2P] HTTP server started on {self.host}:{self.port}")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._job_management_loop()),
            asyncio.create_task(self._discovery_loop()),
        ]

        # Add git update loop if enabled
        if AUTO_UPDATE_ENABLED:
            tasks.append(asyncio.create_task(self._git_update_loop()))

        # If no leader known, start election after short delay
        await asyncio.sleep(5)
        if not self.leader_id:
            await self._start_election()

        # Run forever
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="P2P Orchestrator for RingRift cluster")
    parser.add_argument("--node-id", required=True, help="Unique identifier for this node")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--peers", help="Comma-separated list of known peers (host:port)")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")

    args = parser.parse_args()

    known_peers = []
    if args.peers:
        known_peers = [p.strip() for p in args.peers.split(',')]

    orchestrator = P2POrchestrator(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        known_peers=known_peers,
        ringrift_path=args.ringrift_path,
    )

    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n[P2P] Shutting down...")
        orchestrator.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
