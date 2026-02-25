"""Job Spawner - Extracted from P2POrchestrator._start_local_job().

January 2026: Extracts common job spawning logic to reduce duplication.

This module provides:
- JobSpawnConfig: Configuration for spawning a job
- JobSpawnResult: Result of a job spawn attempt
- spawn_subprocess_job(): Common subprocess spawning logic

Usage:
    from scripts.p2p.job_spawner import (
        JobSpawnConfig,
        JobSpawnResult,
        spawn_subprocess_job,
        build_selfplay_command,
    )

    config = JobSpawnConfig(
        job_type=JobType.SELFPLAY,
        board_type="hex8",
        num_players=2,
        ...
    )
    result = spawn_subprocess_job(config)
    if result.success:
        print(f"Started job {result.job_id} with PID {result.pid}")
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import ClusterJob

logger = logging.getLogger(__name__)


# Engine mode constants
GUMBEL_ENGINE_MODES = {"gumbel", "gumbel-mcts", "gumbel-mcts-only"}

SELFPLAY_ENGINE_MODES = {
    "descent-only",
    "mixed",
    "random-only",
    "heuristic-only",
    "minimax-only",
    "mcts-only",
    "nn-only",
    "best-vs-pool",
    "gumbel",
    "gumbel-mcts",
    "gumbel-mcts-only",
    "nnue-guided",
    "nnue",
    "maxn",
    "brs",
    "paranoid",
    "policy-only",
    "nn-vs-mcts",
    "nn-vs-minimax",
    "nn-vs-descent",
    "tournament-varied",
    "heuristic-vs-nn",
    "heuristic-vs-mcts",
    "random-vs-mcts",
}

CPU_ENGINE_MODES = {
    "descent-only",
    "minimax-only",
    "mcts-only",
    "heuristic-only",
    "random-only",
    "mixed",
    "nn-only",
    "best-vs-pool",
    "nn-vs-mcts",
    "nn-vs-minimax",
    "nn-vs-descent",
    "tournament-varied",
    "heuristic-vs-nn",
    "heuristic-vs-mcts",
    "random-vs-mcts",
}


@dataclass
class JobSpawnConfig:
    """Configuration for spawning a job."""

    job_id: str
    job_type: str  # JobType.value
    board_type: str
    num_players: int
    engine_mode: str
    ringrift_path: str
    ai_service_path: str
    node_id: str

    # Optional parameters
    num_games: int = 100
    output_dir: Path | None = None
    cuda_visible_devices: str | None = None
    simulation_budget: int | None = None
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)

    # Safeguard callbacks
    can_spawn_fn: Callable[[str], tuple[bool, str]] | None = None
    record_spawn_fn: Callable[[], None] | None = None

    # Log file path (relative to output_dir)
    log_filename: str = "run.log"


@dataclass
class JobSpawnResult:
    """Result of a job spawn attempt."""

    success: bool
    job_id: str = ""
    pid: int = 0
    error: str | None = None
    command: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    engine_mode_normalized: str = ""


def normalize_engine_mode(engine_mode: str, job_type: str) -> str:
    """Normalize engine mode based on job type.

    Args:
        engine_mode: Raw engine mode string
        job_type: Type of job (selfplay, cpu_selfplay, etc.)

    Returns:
        Normalized engine mode string
    """
    if job_type in ("gumbel_selfplay",):
        return "gumbel-mcts"

    if engine_mode in GUMBEL_ENGINE_MODES:
        return "gumbel-mcts-only"

    if job_type == "cpu_selfplay":
        return engine_mode if engine_mode in CPU_ENGINE_MODES else "nn-only"

    if engine_mode in SELFPLAY_ENGINE_MODES:
        return engine_mode

    return "nn-only"


def get_num_games_for_board(
    board_type: str,
    job_type: str,
    simulation_budget: int | None = None,
) -> int:
    """Get default number of games based on board type and job type.

    Args:
        board_type: Board type (hex8, square8, square19, hexagonal)
        job_type: Type of job
        simulation_budget: Optional simulation budget (affects gumbel jobs)

    Returns:
        Number of games to run
    """
    # Base defaults by job type
    base_games = {
        "selfplay": 1000,
        "cpu_selfplay": 2000,
        "gpu_selfplay": 100,
        "gumbel_selfplay": 100,
        "hybrid_selfplay": 500,
    }.get(job_type, 100)

    # Adjust for large boards
    if board_type == "square19":
        if job_type == "selfplay":
            return 200
        elif job_type == "cpu_selfplay":
            return 400
        elif job_type == "gpu_selfplay":
            return 50
        elif job_type == "gumbel_selfplay":
            return 10 if (simulation_budget and simulation_budget >= 800) else 50
        return base_games // 5

    if board_type == "hexagonal":
        if job_type == "selfplay":
            return 100
        elif job_type == "cpu_selfplay":
            return 200
        elif job_type in ("gumbel_selfplay", "hex8"):
            return 20 if (simulation_budget and simulation_budget >= 800) else 100
        return base_games // 2

    # Gumbel jobs with high budget use fewer games
    if job_type == "gumbel_selfplay" and simulation_budget and simulation_budget >= 800:
        return 50

    return base_games


def build_base_env(ai_service_path: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build base environment for subprocess.

    Args:
        ai_service_path: Path to ai-service directory
        extra_env: Additional environment variables

    Returns:
        Environment dict for subprocess
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = ai_service_path
    env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
    env["RINGRIFT_JOB_ORIGIN"] = "p2p_orchestrator"

    if extra_env:
        env.update(extra_env)

    return env


def get_python_executable(ai_service_path: str) -> str:
    """Get Python executable path.

    Prefers venv Python if available, falls back to system python3.

    Args:
        ai_service_path: Path to ai-service directory

    Returns:
        Path to Python executable
    """
    venv_python = Path(ai_service_path, "venv", "bin", "python")
    return str(venv_python) if venv_python.exists() else "python3"


def build_selfplay_command(
    config: JobSpawnConfig,
    script_name: str = "run_self_play_soak.py",
) -> list[str]:
    """Build command for selfplay job.

    Args:
        config: Job spawn configuration
        script_name: Name of the selfplay script

    Returns:
        Command list for subprocess
    """
    python_exec = get_python_executable(config.ai_service_path)
    script_path = str(Path(config.ai_service_path, "scripts", script_name))

    cmd = [
        python_exec,
        script_path,
        "--num-games", str(config.num_games),
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--engine-mode", config.engine_mode,
        "--max-moves", "10000",
        "--verbose", "0",
    ]

    if config.output_dir:
        cmd.extend([
            "--log-jsonl", str(config.output_dir / "games.jsonl"),
            "--summary-json", str(config.output_dir / "summary.json"),
            "--record-db", str(config.output_dir / "games.db"),
            "--lean-db",
        ])

    cmd.extend(config.extra_args)
    return cmd


def build_gumbel_command(
    config: JobSpawnConfig,
    use_gpu_tree: bool = True,
) -> list[str]:
    """Build command for Gumbel selfplay job.

    Args:
        config: Job spawn configuration
        use_gpu_tree: Whether to use GPU tree acceleration

    Returns:
        Command list for subprocess
    """
    python_exec = get_python_executable(config.ai_service_path)
    script_path = str(Path(config.ai_service_path, "scripts", "generate_gumbel_selfplay.py"))

    # Normalize board type for gumbel script
    board_arg = {
        "hex": "hexagonal",
        "hex8": "hex8",
    }.get(config.board_type, config.board_type)

    cmd = [
        python_exec,
        script_path,
        "--board", board_arg,
        "--num-players", str(config.num_players),
        "--num-games", str(config.num_games),
        "--simulation-budget", str(config.simulation_budget or 150),
        "--seed", str(int(time.time() * 1000) % 2**31),
        "--allow-fresh-weights",
    ]

    if config.output_dir:
        cmd.extend([
            "--output-dir", str(config.output_dir),
            "--db", str(config.output_dir / "games.db"),
        ])

    if use_gpu_tree:
        cmd.append("--use-gpu-tree")

    cmd.extend(config.extra_args)
    return cmd


def build_gpu_selfplay_command(config: JobSpawnConfig) -> list[str]:
    """Build command for GPU selfplay job.

    Args:
        config: Job spawn configuration

    Returns:
        Command list for subprocess
    """
    python_exec = get_python_executable(config.ai_service_path)
    script_path = str(Path(config.ai_service_path, "scripts", "run_gpu_selfplay.py"))

    # Normalize board type
    board_arg = {
        "hex8": "hex8",
        "hex": "hex8",
        "square8": "square8",
        "square19": "square19",
        "hexagonal": "hexagonal",
    }.get(config.board_type, "square8")

    cmd = [
        python_exec,
        script_path,
        "--board", board_arg,
        "--num-players", str(config.num_players),
        "--num-games", str(config.num_games),
    ]

    if config.output_dir:
        cmd.extend(["--output-dir", str(config.output_dir)])

    cmd.extend(config.extra_args)
    return cmd


def spawn_subprocess(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: Path,
) -> tuple[subprocess.Popen | None, str | None]:
    """Spawn a subprocess with logging.

    Args:
        cmd: Command to execute
        env: Environment variables
        cwd: Working directory
        log_path: Path to log file

    Returns:
        Tuple of (process, error_message)
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=cwd,
                start_new_session=True,  # Feb 2026: Process isolation - selfplay survives P2P restarts
            )
            return proc, None
        finally:
            log_handle.close()
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)


def spawn_subprocess_job(config: JobSpawnConfig) -> JobSpawnResult:
    """Spawn a job subprocess with common setup logic.

    This function handles the common patterns across all job types:
    - Safeguard checks
    - Environment setup
    - Directory creation
    - Subprocess spawning
    - Spawn recording

    Args:
        config: Job spawn configuration

    Returns:
        JobSpawnResult with success status and details
    """
    # Check safeguards
    if config.can_spawn_fn:
        can_spawn, reason = config.can_spawn_fn(
            f"{config.job_type}-{config.board_type}-{config.num_players}p"
        )
        if not can_spawn:
            logger.info(f"BLOCKED {config.job_type} spawn: {reason}")
            return JobSpawnResult(
                success=False,
                job_id=config.job_id,
                error=f"Safeguard blocked: {reason}",
            )

    # Create output directory
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Build environment
    env = build_base_env(config.ai_service_path, config.extra_env)

    # Handle CUDA_VISIBLE_DEVICES
    if config.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices).strip()

    # Build command based on job type
    job_type_lower = config.job_type.lower()
    if job_type_lower == "gumbel_selfplay":
        cmd = build_gumbel_command(config)
    elif job_type_lower == "gpu_selfplay":
        cmd = build_gpu_selfplay_command(config)
    elif job_type_lower in ("selfplay", "cpu_selfplay", "hybrid_selfplay"):
        cmd = build_selfplay_command(config)
    else:
        return JobSpawnResult(
            success=False,
            job_id=config.job_id,
            error=f"Unknown job type: {config.job_type}",
        )

    # Determine log path
    log_path = (
        config.output_dir / config.log_filename
        if config.output_dir
        else Path("/tmp") / f"{config.job_id}.log"
    )

    # Spawn subprocess
    proc, error = spawn_subprocess(cmd, env, config.ringrift_path, log_path)

    if error or proc is None:
        return JobSpawnResult(
            success=False,
            job_id=config.job_id,
            error=error or "Failed to spawn subprocess",
            command=cmd,
        )

    # Record spawn
    if config.record_spawn_fn:
        config.record_spawn_fn()

    logger.info(f"Started {config.job_type} job {config.job_id} (PID {proc.pid})")

    return JobSpawnResult(
        success=True,
        job_id=config.job_id,
        pid=proc.pid,
        command=cmd,
        output_dir=config.output_dir,
        engine_mode_normalized=config.engine_mode,
    )


def create_output_dir(
    ringrift_path: str,
    job_type: str,
    board_type: str,
    num_players: int,
    job_id: str,
) -> Path:
    """Create output directory for a job.

    Args:
        ringrift_path: Base ringrift path
        job_type: Type of job
        board_type: Board type
        num_players: Number of players
        job_id: Job ID

    Returns:
        Path to output directory
    """
    subdir_map = {
        "selfplay": "p2p",
        "cpu_selfplay": "p2p",
        "gpu_selfplay": "",
        "gumbel_selfplay": "gumbel",
        "hybrid_selfplay": "p2p",
    }

    base = Path(ringrift_path, "ai-service", "data")

    if job_type == "gpu_selfplay":
        output_dir = base / "games"
    else:
        subdir = subdir_map.get(job_type, "p2p")
        suffix = "_cpu" if job_type == "cpu_selfplay" else ""
        output_dir = base / "selfplay" / subdir / f"{board_type}_{num_players}p{suffix}" / job_id

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
