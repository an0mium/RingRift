#!/usr/bin/env python3
"""Unified Work Orchestrator - Ensures all cluster nodes are fully utilized.

This script monitors node resources and assigns appropriate work types:
- GPU idle → Start training jobs, GPU selfplay, or gauntlets
- CPU idle → Start CMA-ES optimization, hybrid selfplay, or database merges
- Both idle → Start both GPU and CPU work

It runs on each node independently (decentralized) to avoid leader dependency.

Usage:
    python scripts/unified_work_orchestrator.py --node-id <id>

    # As daemon (recommended):
    python scripts/unified_work_orchestrator.py --node-id <id> --daemon

Work Types:
    1. NNUE Training (GPU) - Train neural network on selfplay data
    2. CMA-ES Optimization (CPU) - Optimize heuristic weights
    3. Hybrid Selfplay (CPU+light GPU) - Generate training data
    4. GPU Selfplay (GPU) - Pure neural network self-play
    5. ELO Tournaments (GPU) - Model evaluation
    6. Gauntlets (GPU) - Quick model benchmarking
    7. Data Aggregation (CPU/IO) - Merge and export databases
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thresholds
GPU_IDLE_THRESHOLD = 30.0       # Start GPU work if GPU < 30%
CPU_IDLE_THRESHOLD = 50.0       # Start CPU work if CPU < 50%
GPU_MEM_MAX = 90.0              # Don't start GPU work if memory > 90%
MEMORY_MAX = 85.0               # Don't start work if system memory > 85%
CHECK_INTERVAL = 60             # Check every 60 seconds
WORK_COOLDOWN = 120             # Wait 2 min between starting same work type

P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))


@dataclass
class NodeResources:
    """Current node resource usage."""
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    has_gpu: bool = False
    gpu_name: str = ""
    # Current jobs
    selfplay_jobs: int = 0
    training_jobs: int = 0


@dataclass
class WorkState:
    """Track work being done on this node."""
    last_training_start: float = 0.0
    last_cmaes_start: float = 0.0
    last_tournament_start: float = 0.0
    last_gauntlet_start: float = 0.0
    last_selfplay_start: float = 0.0
    last_data_merge_start: float = 0.0
    # Active processes
    active_pids: Dict[str, int] = field(default_factory=dict)


def get_node_resources() -> NodeResources:
    """Get current node resource usage."""
    resources = NodeResources()

    # Try to get from P2P orchestrator first
    try:
        with urllib.request.urlopen(f"http://localhost:{P2P_PORT}/health", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            resources.gpu_percent = float(data.get("gpu_percent", 0) or 0)
            resources.cpu_percent = float(data.get("cpu_percent", 0) or 0)
            resources.memory_percent = float(data.get("memory_percent", 0) or 0)
            resources.disk_percent = float(data.get("disk_percent", 0) or 0)
            resources.selfplay_jobs = int(data.get("selfplay_jobs", 0) or 0)
            resources.training_jobs = int(data.get("training_jobs", 0) or 0)
    except Exception:
        pass

    # Get GPU info directly
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                resources.gpu_percent = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                resources.gpu_memory_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                resources.gpu_name = parts[3]
                resources.has_gpu = True
    except Exception:
        pass

    # Get CPU/memory if not from P2P
    if resources.cpu_percent == 0:
        try:
            import psutil
            resources.cpu_percent = psutil.cpu_percent(interval=1)
            resources.memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass

    return resources


def count_running_processes(pattern: str) -> int:
    """Count processes matching a pattern."""
    try:
        result = subprocess.run(
            ["pgrep", "-fc", pattern],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        return 0


def is_training_running() -> bool:
    """Check if any training is running."""
    return count_running_processes("train_nnue|run_training") > 0


def is_cmaes_running() -> bool:
    """Check if CMA-ES optimization is running."""
    return count_running_processes("cmaes|cma_es|HeuristicAI.*json") > 0


def is_tournament_running() -> bool:
    """Check if ELO tournament is running."""
    return count_running_processes("run_model_elo_tournament") > 0


def is_gauntlet_running() -> bool:
    """Check if gauntlet is running."""
    return count_running_processes("baseline_gauntlet") > 0


def is_data_merge_running() -> bool:
    """Check if data merge/export is running."""
    return count_running_processes("merge_game_dbs|export_training_data") > 0


def get_ai_service_root() -> Path:
    """Get the AI service root directory."""
    candidates = [
        Path(__file__).parent.parent,
        Path.home() / "ringrift" / "ai-service",
        Path("/workspace/ringrift/ai-service"),
    ]
    for path in candidates:
        if path.exists() and (path / "scripts").exists():
            return path
    return Path(__file__).parent.parent


def find_training_data() -> List[Tuple[str, int, int]]:
    """Find available training data by board type.

    Returns: List of (board_config, num_players, game_count) tuples
    """
    ai_root = get_ai_service_root()
    data_dir = ai_root / "data" / "selfplay"

    configs = []

    if not data_dir.exists():
        return configs

    # Check p2p_hybrid directory structure
    p2p_dir = data_dir / "p2p_hybrid"
    if p2p_dir.exists():
        for config_dir in p2p_dir.iterdir():
            if config_dir.is_dir():
                # Parse board type and players from dir name (e.g., "square8_2p")
                name = config_dir.name
                if "_" in name and name.endswith("p"):
                    parts = name.rsplit("_", 1)
                    board = parts[0]
                    try:
                        players = int(parts[1].rstrip("p"))
                        # Count games in all subdirs
                        game_count = 0
                        for db_file in config_dir.rglob("*.db"):
                            # Estimate games from file size (rough: 1KB per game)
                            game_count += int(db_file.stat().st_size / 1024)
                        if game_count > 100:
                            configs.append((board, players, game_count))
                    except ValueError:
                        continue

    # Sort by game count descending
    configs.sort(key=lambda x: x[2], reverse=True)
    return configs


def start_training(board_type: str, num_players: int) -> Optional[int]:
    """Start a training job and return PID."""
    ai_root = get_ai_service_root()

    # Try via P2P API first
    try:
        data = json.dumps({
            "board_type": board_type,
            "num_players": num_players,
            "model_type": "nnue"
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{P2P_PORT}/training/start",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("success"):
                logger.info(f"Started training via P2P: {board_type}_{num_players}p -> {result.get('worker')}")
                return -1  # API started it, no local PID
    except Exception as e:
        logger.debug(f"P2P training start failed: {e}")

    # Fallback to local training
    train_script = ai_root / "scripts" / "train_nnue.py"
    if not train_script.exists():
        logger.warning(f"Training script not found: {train_script}")
        return None

    log_file = ai_root / "logs" / f"training_{board_type}_{num_players}p.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(train_script),
             "--board", board_type,
             "--players", str(num_players)],
            cwd=str(ai_root),
            stdout=log,
            stderr=log,
            start_new_session=True
        )

    logger.info(f"Started local training: {board_type}_{num_players}p (PID: {proc.pid})")
    return proc.pid


def start_cmaes(board_type: str, num_players: int) -> Optional[int]:
    """Start CMA-ES optimization."""
    ai_root = get_ai_service_root()

    # Try via P2P API first (if endpoint exists)
    try:
        data = json.dumps({
            "board_type": board_type,
            "num_players": num_players,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{P2P_PORT}/cmaes/start",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("success"):
                logger.info(f"Started CMA-ES via P2P: {board_type}_{num_players}p")
                return -1
    except Exception:
        pass

    # Fallback to local
    cmaes_script = ai_root / "scripts" / "cmaes_distributed.py"
    if not cmaes_script.exists():
        cmaes_script = ai_root / "scripts" / "run_cmaes.py"

    if not cmaes_script.exists():
        return None

    log_file = ai_root / "logs" / f"cmaes_{board_type}_{num_players}p.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(cmaes_script),
             "--board", board_type,
             "--players", str(num_players)],
            cwd=str(ai_root),
            stdout=log,
            stderr=log,
            start_new_session=True
        )

    logger.info(f"Started local CMA-ES: {board_type}_{num_players}p (PID: {proc.pid})")
    return proc.pid


def start_tournament(board_type: str = "square8", num_players: int = 2) -> Optional[int]:
    """Start an ELO tournament."""
    ai_root = get_ai_service_root()

    script = ai_root / "scripts" / "run_model_elo_tournament.py"
    if not script.exists():
        return None

    log_file = ai_root / "logs" / f"tournament_{board_type}_{num_players}p.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script),
             "--board", board_type,
             "--players", str(num_players),
             "--games", "20",
             "--quick",
             "--include-baselines"],
            cwd=str(ai_root),
            stdout=log,
            stderr=log,
            start_new_session=True
        )

    logger.info(f"Started tournament: {board_type}_{num_players}p (PID: {proc.pid})")
    return proc.pid


def start_gauntlet(board_type: str = "square8", num_players: int = 2) -> Optional[int]:
    """Start a baseline gauntlet."""
    ai_root = get_ai_service_root()

    script = ai_root / "scripts" / "baseline_gauntlet.py"
    if not script.exists():
        return None

    log_file = ai_root / "logs" / f"gauntlet_{board_type}_{num_players}p.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script),
             "--board", board_type,
             "--players", str(num_players)],
            cwd=str(ai_root),
            stdout=log,
            stderr=log,
            start_new_session=True
        )

    logger.info(f"Started gauntlet: {board_type}_{num_players}p (PID: {proc.pid})")
    return proc.pid


def start_hybrid_selfplay(board_type: str, num_players: int) -> Optional[int]:
    """Start hybrid selfplay via P2P."""
    try:
        data = json.dumps({
            "board_type": board_type,
            "num_players": num_players,
            "engine_mode": "mixed"
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{P2P_PORT}/jobs/start",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("success"):
                logger.info(f"Started selfplay via P2P: {board_type}_{num_players}p")
                return -1
    except Exception as e:
        logger.debug(f"P2P selfplay start failed: {e}")
    return None


class UnifiedWorkOrchestrator:
    """Orchestrates work across the node to maximize utilization."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.ai_root = get_ai_service_root()
        self.state = WorkState()
        self.running = True

    def _cooldown_ok(self, work_type: str) -> bool:
        """Check if we're past cooldown for a work type."""
        now = time.time()
        last_start = getattr(self.state, f"last_{work_type}_start", 0)
        return (now - last_start) >= WORK_COOLDOWN

    def _should_start_gpu_work(self, resources: NodeResources) -> bool:
        """Check if we should start GPU work."""
        if not resources.has_gpu:
            return False
        if resources.gpu_percent >= GPU_IDLE_THRESHOLD:
            return False
        if resources.gpu_memory_percent >= GPU_MEM_MAX:
            return False
        if resources.memory_percent >= MEMORY_MAX:
            return False
        return True

    def _should_start_cpu_work(self, resources: NodeResources) -> bool:
        """Check if we should start CPU work."""
        if resources.cpu_percent >= CPU_IDLE_THRESHOLD:
            return False
        if resources.memory_percent >= MEMORY_MAX:
            return False
        return True

    def _pick_training_config(self) -> Optional[Tuple[str, int]]:
        """Pick a board config for training based on available data."""
        configs = find_training_data()
        if not configs:
            return None

        # Pick config with most data that hasn't been trained recently
        for board, players, games in configs[:5]:
            if games > 1000:
                return (board, players)
        return None

    def run_iteration(self):
        """Run one iteration of work assignment."""
        resources = get_node_resources()

        logger.debug(f"Resources: GPU={resources.gpu_percent:.1f}%, CPU={resources.cpu_percent:.1f}%, "
                     f"Mem={resources.memory_percent:.1f}%, GPUMem={resources.gpu_memory_percent:.1f}%")

        work_started = []

        # GPU work
        if self._should_start_gpu_work(resources):
            # Priority 1: Training (if not already running)
            if not is_training_running() and self._cooldown_ok("training"):
                config = self._pick_training_config()
                if config:
                    pid = start_training(config[0], config[1])
                    if pid:
                        self.state.last_training_start = time.time()
                        work_started.append(f"training:{config[0]}_{config[1]}p")

            # Priority 2: Tournament (if GPU still available)
            resources = get_node_resources()  # Refresh
            if (self._should_start_gpu_work(resources) and
                not is_tournament_running() and
                self._cooldown_ok("tournament")):
                pid = start_tournament()
                if pid:
                    self.state.last_tournament_start = time.time()
                    work_started.append("tournament:square8_2p")

            # Priority 3: Gauntlet
            resources = get_node_resources()
            if (self._should_start_gpu_work(resources) and
                not is_gauntlet_running() and
                self._cooldown_ok("gauntlet")):
                pid = start_gauntlet()
                if pid:
                    self.state.last_gauntlet_start = time.time()
                    work_started.append("gauntlet:square8_2p")

        # CPU work
        if self._should_start_cpu_work(resources):
            # Priority 1: CMA-ES
            if not is_cmaes_running() and self._cooldown_ok("cmaes"):
                config = self._pick_training_config()
                if config:
                    pid = start_cmaes(config[0], config[1])
                    if pid:
                        self.state.last_cmaes_start = time.time()
                        work_started.append(f"cmaes:{config[0]}_{config[1]}p")

            # Priority 2: More selfplay (if P2P isn't spawning enough)
            resources = get_node_resources()
            if (self._should_start_cpu_work(resources) and
                resources.selfplay_jobs < 5 and
                self._cooldown_ok("selfplay")):
                config = self._pick_training_config()
                if config:
                    pid = start_hybrid_selfplay(config[0], config[1])
                    if pid:
                        self.state.last_selfplay_start = time.time()
                        work_started.append(f"selfplay:{config[0]}_{config[1]}p")

        if work_started:
            logger.info(f"Started work: {', '.join(work_started)}")

        return len(work_started)

    def run(self):
        """Main loop."""
        logger.info(f"Starting Unified Work Orchestrator for {self.node_id}")
        logger.info(f"AI service root: {self.ai_root}")

        while self.running:
            try:
                self.run_iteration()
            except Exception as e:
                logger.error(f"Iteration error: {e}")

            time.sleep(CHECK_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Unified Work Orchestrator")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    global CHECK_INTERVAL
    CHECK_INTERVAL = args.interval

    orchestrator = UnifiedWorkOrchestrator(args.node_id)

    if args.once:
        started = orchestrator.run_iteration()
        logger.info(f"Started {started} work item(s)")
        sys.exit(0)

    orchestrator.run()


if __name__ == "__main__":
    main()
