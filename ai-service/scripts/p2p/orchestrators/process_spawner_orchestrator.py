"""Process Spawner Orchestrator - Handles job process lifecycle.

January 2026: Created as part of Phase 3 P2POrchestrator decomposition.

Responsibilities:
- Local job process spawning (selfplay, GPU selfplay, etc.)
- Cluster-wide job management and distribution
- Job type selection and command building
- Process lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

# Import job type constants if available
try:
    from scripts.p2p.job_types import JobType, ClusterJob
    HAS_JOB_TYPES = True
except ImportError:
    HAS_JOB_TYPES = False
    JobType = None
    ClusterJob = None

# Import selfplay config helpers if available
try:
    from scripts.p2p.config.selfplay_job_configs import (
        SELFPLAY_ENGINE_MODES,
        GUMBEL_ENGINE_MODES,
    )
    HAS_SELFPLAY_CONFIG = True
except ImportError:
    HAS_SELFPLAY_CONFIG = False
    SELFPLAY_ENGINE_MODES = {"nn-only", "gumbel-mcts-only", "mixed"}
    GUMBEL_ENGINE_MODES = {"gumbel-mcts", "gumbel", "gumbel-mcts-only"}

# Import safeguards if available
try:
    from app.coordination.safeguards import check_before_spawn
    HAS_SAFEGUARDS = True
    _safeguards = True  # Placeholder for safeguard module
except ImportError:
    HAS_SAFEGUARDS = False
    _safeguards = None
    check_before_spawn = None

# Import Gumbel budget utilities if available
try:
    from app.ai.gumbel_common import get_adaptive_budget_for_elo
    HAS_GUMBEL_BUDGET = True
except ImportError:
    HAS_GUMBEL_BUDGET = False

    def get_adaptive_budget_for_elo(elo: float) -> int:
        """Fallback budget calculation."""
        if elo < 1200:
            return 64
        elif elo < 1400:
            return 150
        elif elo < 1600:
            return 400
        else:
            return 800


class ProcessSpawnerOrchestrator(BaseOrchestrator):
    """Orchestrator for job process spawning and lifecycle management.

    This orchestrator handles all aspects of starting and managing job processes:
    - Building commands for different job types (selfplay, GPU selfplay, etc.)
    - Spawning local processes with proper environment
    - Tracking process lifecycle and monitoring
    - Cluster-wide job distribution (leader only)

    The actual subprocess management is delegated to JobOrchestrator, but this
    orchestrator handles job type selection, command building, and lifecycle.

    Usage:
        # In P2POrchestrator.__init__:
        self.process_spawner = ProcessSpawnerOrchestrator(self)

        # Start a local job:
        job = await self.process_spawner.start_local_job(
            JobType.GPU_SELFPLAY, board_type="hex8", num_players=2
        )
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the process spawner orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Statistics
        self._local_jobs_started: int = 0
        self._local_jobs_completed: int = 0
        self._local_jobs_failed: int = 0
        self._cluster_jobs_dispatched: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "process_spawner"

    def health_check(self) -> HealthCheckResult:
        """Check the health of process spawner orchestrator.

        Returns:
            HealthCheckResult with job spawning status.
        """
        try:
            issues = []

            # Check job orchestrator availability
            jobs_orch = getattr(self._p2p, "jobs", None)
            if jobs_orch is None:
                issues.append("JobOrchestrator not available")

            # Check failure rate
            total = self._local_jobs_completed + self._local_jobs_failed
            if total > 10:
                failure_rate = self._local_jobs_failed / total
                if failure_rate > 0.3:
                    issues.append(f"High local job failure rate: {failure_rate:.0%}")

            # Check local jobs dict
            local_jobs = getattr(self._p2p, "local_jobs", None)
            if local_jobs is None:
                issues.append("local_jobs not available")

            healthy = len(issues) == 0
            message = "Process spawner healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "local_jobs_started": self._local_jobs_started,
                    "local_jobs_completed": self._local_jobs_completed,
                    "local_jobs_failed": self._local_jobs_failed,
                    "cluster_jobs_dispatched": self._cluster_jobs_dispatched,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_ai_service_path(self) -> str:
        """Get the AI service path."""
        if hasattr(self._p2p, "_get_ai_service_path"):
            return self._p2p._get_ai_service_path()
        ringrift_path = getattr(self._p2p, "ringrift_path", None)
        if ringrift_path:
            return str(Path(ringrift_path) / "ai-service")
        return ""

    def _get_script_path(self, script_name: str) -> str:
        """Get the full path to a script."""
        if hasattr(self._p2p, "_get_script_path"):
            return self._p2p._get_script_path(script_name)
        ai_service = self._get_ai_service_path()
        return str(Path(ai_service) / "scripts" / script_name)

    def _load_distributed_hosts(self) -> dict[str, Any]:
        """Load the distributed hosts configuration."""
        if hasattr(self._p2p, "_load_distributed_hosts"):
            return self._p2p._load_distributed_hosts()
        return {"hosts": {}}

    def _spawn_and_track_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cmd: list[str],
        output_dir: Path,
        log_filename: str = "run.log",
        cuda_visible_devices: str | None = None,
        safeguard_reason: str | None = None,
    ) -> tuple[Any, Any] | None:
        """Spawn and track a job using JobOrchestrator.

        Delegates to JobOrchestrator.spawn_and_track_job() if available,
        otherwise falls back to P2POrchestrator._spawn_and_track_job().
        """
        # Try JobOrchestrator first
        jobs_orch = getattr(self._p2p, "jobs", None)
        if jobs_orch is not None and hasattr(jobs_orch, "spawn_and_track_job"):
            return jobs_orch.spawn_and_track_job(
                job_id=job_id,
                job_type=job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                cmd=cmd,
                output_dir=output_dir,
                log_filename=log_filename,
                cuda_visible_devices=cuda_visible_devices,
                safeguard_reason=safeguard_reason,
            )

        # Fallback to P2POrchestrator method
        if hasattr(self._p2p, "_spawn_and_track_job"):
            return self._p2p._spawn_and_track_job(
                job_id=job_id,
                job_type=job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                cmd=cmd,
                output_dir=output_dir,
                log_filename=log_filename,
                cuda_visible_devices=cuda_visible_devices,
                safeguard_reason=safeguard_reason,
            )

        self._log_error("No job spawning method available")
        return None

    def _save_state(self) -> None:
        """Save P2P state."""
        if hasattr(self._p2p, "_save_state"):
            self._p2p._save_state()

    def _update_gpu_job_count(self, delta: int) -> None:
        """Update GPU job count."""
        if hasattr(self._p2p, "_update_gpu_job_count"):
            self._p2p._update_gpu_job_count(delta)

    # =========================================================================
    # Local Job Spawning
    # =========================================================================

    async def start_local_job(
        self,
        job_type: Any,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "gumbel-mcts",
        job_id: str | None = None,
        cuda_visible_devices: str | None = None,
        export_params: dict[str, Any] | None = None,
        simulation_budget: int | None = None,
    ) -> Any | None:
        """Start a job on the local node.

        Jan 29, 2026: Implementation moved from P2POrchestrator._start_local_job().

        SAFEGUARD: Checks coordination safeguards before spawning.

        Args:
            job_type: Type of job to start (JobType enum or string)
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            engine_mode: Engine mode for selfplay
            job_id: Optional job ID (auto-generated if not provided)
            cuda_visible_devices: CUDA device selection
            export_params: Parameters for DATA_EXPORT jobs
            simulation_budget: Gumbel MCTS budget (None = use tier default)

        Returns:
            ClusterJob if successful, None if blocked or failed
        """
        try:
            # SAFEGUARD: Check safeguards before spawning
            if HAS_SAFEGUARDS and _safeguards and check_before_spawn is not None:
                task_type_str = job_type.value if hasattr(job_type, "value") else str(job_type)
                allowed, reason = check_before_spawn(task_type_str, self.node_id)
                if not allowed:
                    self._log_info(f"SAFEGUARD blocked {task_type_str} on {self.node_id}: {reason}")
                    # Track blocked spawn via JobOrchestrationManager
                    job_orchestration = getattr(self._p2p, "job_orchestration", None)
                    if job_orchestration is not None:
                        job_orchestration.record_spawn_blocked(f"safeguard:{reason}")
                    return None

                # Apply backpressure delay
                try:
                    delay = _safeguards.get_delay() if hasattr(_safeguards, "get_delay") else 0
                except Exception:
                    delay = 0
                if delay > 0:
                    self._log_info(f"SAFEGUARD applying {delay:.1f}s backpressure delay")
                    await asyncio.sleep(delay)

            # Generate or validate job_id
            if job_id:
                job_id = str(job_id)
                jobs_lock = getattr(self._p2p, "jobs_lock", None)
                local_jobs = getattr(self._p2p, "local_jobs", {})
                if jobs_lock is not None:
                    with jobs_lock:
                        existing = local_jobs.get(job_id)
                else:
                    existing = local_jobs.get(job_id)
                if existing and getattr(existing, "status", None) == "running":
                    return existing
            else:
                job_id = str(uuid.uuid4())[:8]

            # Get JobType enum if needed
            if HAS_JOB_TYPES and JobType is not None:
                if isinstance(job_type, str):
                    job_type = JobType(job_type)

            # Route to appropriate handler based on job type
            job_type_val = job_type.value if hasattr(job_type, "value") else str(job_type)

            if job_type_val == "selfplay":
                return await self._start_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode
                )
            elif job_type_val == "cpu_selfplay":
                return await self._start_cpu_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode
                )
            elif job_type_val == "gpu_selfplay":
                return await self._start_gpu_selfplay_job(
                    job_id, job_type, board_type, num_players, cuda_visible_devices
                )
            elif job_type_val == "hybrid_selfplay":
                return await self._start_hybrid_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode, cuda_visible_devices
                )
            elif job_type_val == "gumbel_selfplay":
                return await self._start_gumbel_selfplay_job(
                    job_id, job_type, board_type, num_players, simulation_budget, cuda_visible_devices
                )
            elif job_type_val == "data_export":
                return await self._start_data_export_job(
                    job_id, job_type, board_type, num_players, export_params
                )
            else:
                self._log_warning(f"Unknown job type: {job_type_val}")
                return None

        except Exception as e:
            self._log_error(f"Failed to start job: {e}")
            self._local_jobs_failed += 1
            return None

    async def _start_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
    ) -> Any | None:
        """Start a standard selfplay job."""
        # Normalize engine mode
        if engine_mode in GUMBEL_ENGINE_MODES:
            engine_mode_norm = "gumbel-mcts-only"
        elif engine_mode in SELFPLAY_ENGINE_MODES:
            engine_mode_norm = engine_mode
        else:
            engine_mode_norm = "nn-only"

        # Memory-safety defaults for large boards
        num_games = 1000
        extra_args: list[str] = []
        if board_type in ("square19", "hexagonal"):
            num_games = 200 if board_type == "square19" else 100
            extra_args.extend(["--memory-constrained"])

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "p2p",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use venv python if available
        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--verbose", "0",
            *extra_args,
        ]

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            safeguard_reason=f"selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Add process monitoring
        asyncio.create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "selfplay"
        ))

        return job

    async def _start_cpu_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
    ) -> Any | None:
        """Start a CPU-only selfplay job."""
        # CPU-friendly engine modes
        cpu_engine_modes = {
            "descent-only", "minimax-only", "mcts-only", "heuristic-only",
            "random-only", "mixed", "nn-only", "best-vs-pool",
            "nn-vs-mcts", "nn-vs-minimax", "nn-vs-descent", "tournament-varied",
            "heuristic-vs-nn", "heuristic-vs-mcts", "random-vs-mcts",
        }
        engine_mode_norm = engine_mode if engine_mode in cpu_engine_modes else "nn-only"

        # CPU-only jobs can handle more games per batch
        num_games = 2000
        extra_args: list[str] = []
        if board_type in ("square19", "hexagonal"):
            num_games = 400 if board_type == "square19" else 200
            extra_args.extend(["--memory-constrained"])

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "p2p",
            f"{board_type}_{num_players}p_cpu",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--verbose", "0",
            *extra_args,
        ]

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            cuda_visible_devices="",  # Disable GPU
            safeguard_reason=f"cpu-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        asyncio.create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "cpu_selfplay"
        ))

        return job

    async def _start_gpu_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a GPU selfplay job."""
        # Normalize board type
        board_arg = {
            "hex8": "hex8",
            "hex": "hex8",
            "square8": "square8",
            "square19": "square19",
            "hexagonal": "hexagonal",
        }.get(board_type, "square8")

        # Number of games per batch
        num_games = 100
        if board_arg == "square19":
            num_games = 50
        elif board_arg == "hexagonal":
            num_games = 100

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "games",
        )

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_gpu_selfplay.py"),
            "--board", board_arg,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--output-dir", str(output_dir),
        ]

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("gpu_selfplay")

        gpu_engine_mode = "gumbel-mcts"
        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=gpu_engine_mode,
            cmd=cmd,
            output_dir=output_dir,
            log_filename="gpu_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"gpu-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track GPU job count
        self._update_gpu_job_count(+1)

        # Track diversity metrics
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": gpu_engine_mode,
            })

        # Monitor GPU selfplay
        job_coord_manager = getattr(self._p2p, "job_coordination_manager", None)
        if job_coord_manager is not None:
            asyncio.create_task(job_coord_manager.monitor_gpu_selfplay_and_validate(
                job_id, proc, output_dir, board_type, num_players
            ))

        return job

    async def _start_hybrid_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a hybrid CPU/GPU selfplay job."""
        # Normalize engine mode for hybrid
        hybrid_engine_modes = {
            "random-only", "heuristic-only", "mixed", "nnue-guided", "mcts",
            "gumbel-mcts-only", "maxn-only", "brs-only", "policy-only", "diverse"
        }
        nn_modes = {"nn-only", "best-vs-pool", "nn-vs-mcts", "nn-vs-minimax", "nn-vs-descent", "tournament-varied"}
        engine_mode_map = {
            "gumbel-mcts": "gumbel-mcts-only",
            "maxn": "maxn-only",
            "brs": "brs-only",
            "minimax": "minimax-only",
        }

        if engine_mode in hybrid_engine_modes:
            engine_mode_norm = engine_mode
        elif engine_mode in engine_mode_map:
            engine_mode_norm = engine_mode_map[engine_mode]
        elif engine_mode in nn_modes:
            engine_mode_norm = "nnue-guided"
        elif engine_mode in ("mcts-only", "descent-only"):
            engine_mode_norm = "mcts"
        elif engine_mode == "minimax-only":
            engine_mode_norm = "minimax-only"
        else:
            engine_mode_norm = "diverse"

        # Game counts based on board type
        num_games = 1000
        if board_type == "square19":
            num_games = 500
        elif board_type in ("hex", "hexagonal"):
            num_games = 300

        output_dir = Path(
            self._get_ai_service_path(),
            "data",
            "selfplay",
            "p2p_hybrid",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--verbose", "0",
        ]

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("hybrid_selfplay")

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            log_filename="hybrid_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"hybrid-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track diversity
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": engine_mode_norm,
            })

        asyncio.create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "hybrid_selfplay"
        ))

        return job

    async def _start_gumbel_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        simulation_budget: int | None,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a Gumbel MCTS selfplay job."""
        # Determine effective budget
        if simulation_budget is not None:
            effective_budget = simulation_budget
        else:
            # Look up config Elo and use adaptive budget
            config_key = f"{board_type}_{num_players}p"
            try:
                selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
                if selfplay_scheduler is not None and hasattr(selfplay_scheduler, "get_config_elo"):
                    config_elo = selfplay_scheduler.get_config_elo(config_key)
                else:
                    config_elo = 1200.0
                effective_budget = get_adaptive_budget_for_elo(config_elo)
                self._log_debug(f"[Gumbel] {config_key}: Elo={config_elo:.0f} -> budget={effective_budget}")
            except Exception:
                effective_budget = 150  # Fallback to bootstrap tier

        # Games based on board type and budget
        num_games = 50 if effective_budget >= 800 else 100
        if board_type == "square19":
            num_games = 10 if effective_budget >= 800 else 50
        elif board_type in ("hex", "hexagonal", "hex8"):
            num_games = 20 if effective_budget >= 800 else 100

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "gumbel",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize board type for gumbel script
        board_arg = {"hex": "hexagonal", "hex8": "hex8"}.get(board_type, board_type)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("generate_gumbel_selfplay.py"),
            "--board", board_arg,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--simulation-budget", str(effective_budget),
            "--output-dir", str(output_dir),
            "--db", str(output_dir / "games.db"),
            "--seed", str(int(time.time() * 1000) % 2**31),
            "--allow-fresh-weights",
        ]

        # Check if GPU tree is disabled for this node
        node_config = self._load_distributed_hosts().get("hosts", {}).get(self.node_id, {})
        if not node_config.get("disable_gpu_tree", False):
            cmd.append("--use-gpu-tree")

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("gumbel_selfplay")

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode="gumbel-mcts",
            cmd=cmd,
            output_dir=output_dir,
            log_filename="gumbel_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"gumbel-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track diversity
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": "gumbel-mcts",
            })

        asyncio.create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "gumbel_selfplay"
        ))

        return job

    async def _start_data_export_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        export_params: dict[str, Any] | None,
    ) -> Any | None:
        """Start a data export job."""
        if not export_params:
            self._log_info("DATA_EXPORT job requires export_params")
            return None

        input_path = export_params.get("input_path")
        output_path = export_params.get("output_path")
        encoder_version = export_params.get("encoder_version", "v3")
        max_games = export_params.get("max_games", 5000)
        is_jsonl = export_params.get("is_jsonl", False)

        if not input_path or not output_path:
            self._log_info("DATA_EXPORT requires input_path and output_path")
            return None

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        if is_jsonl:
            export_script = self._get_script_path("jsonl_to_npz.py")
            cmd = [
                python_exec,
                export_script,
                "--input", str(input_path),
                "--output", str(output_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--gpu-selfplay",
                "--max-games", str(max_games),
            ]
            if encoder_version and encoder_version != "default":
                cmd.extend(["--encoder-version", encoder_version])
        else:
            export_script = self._get_script_path("export_replay_dataset.py")
            cmd = [
                python_exec,
                export_script,
                "--db", str(input_path),
                "--output", str(output_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--max-games", str(max_games),
                "--require-completed",
                "--min-moves", "10",
            ]
            if encoder_version and encoder_version != "default":
                cmd.extend(["--encoder-version", encoder_version])

        # Start export process
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_ai_service_path()
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        log_path = output_dir / f"export_{job_id}.log"
        log_handle = open(log_path, "w")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=self._get_ai_service_path(),
            )
        finally:
            log_handle.close()

        # Create ClusterJob
        if HAS_JOB_TYPES and ClusterJob is not None:
            job = ClusterJob(
                job_id=job_id,
                job_type=job_type,
                node_id=self.node_id,
                board_type=board_type,
                num_players=num_players,
                engine_mode="export",
                pid=proc.pid,
                started_at=time.time(),
                status="running",
            )
        else:
            job = {
                "job_id": job_id,
                "job_type": "data_export",
                "node_id": self.node_id,
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": "export",
                "pid": proc.pid,
                "started_at": time.time(),
                "status": "running",
            }

        # Track in local_jobs
        jobs_lock = getattr(self._p2p, "jobs_lock", None)
        local_jobs = getattr(self._p2p, "local_jobs", {})
        if jobs_lock is not None:
            with jobs_lock:
                local_jobs[job_id] = job
        else:
            local_jobs[job_id] = job

        self._log_info(f"Started DATA_EXPORT job {job_id} (PID {proc.pid}): {input_path} -> {output_path}")
        self._save_state()
        self._local_jobs_started += 1

        # Track via JobOrchestrationManager
        job_orchestration = getattr(self._p2p, "job_orchestration", None)
        if job_orchestration is not None:
            job_type_val = job_type.value if hasattr(job_type, "value") else str(job_type)
            job_orchestration.record_job_started(job_type_val)

        return job

    # =========================================================================
    # GPU Selection
    # =========================================================================

    def _auto_select_gpu(self, job_type: str) -> str:
        """Auto-select a GPU device for a job.

        Args:
            job_type: Type of job for counting running jobs

        Returns:
            CUDA_VISIBLE_DEVICES string
        """
        gpu_count = 0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                gpu_count = len([line for line in out.stdout.splitlines() if line.strip()])
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            gpu_count = 0

        if gpu_count > 0:
            # Count running jobs of this type
            jobs_lock = getattr(self._p2p, "jobs_lock", None)
            local_jobs = getattr(self._p2p, "local_jobs", {})
            if jobs_lock is not None:
                with jobs_lock:
                    running_jobs = sum(
                        1 for j in local_jobs.values()
                        if getattr(j, "status", None) == "running"
                        and job_type.lower() in str(getattr(j, "job_type", "")).lower()
                    )
            else:
                running_jobs = 0
            return str(running_jobs % gpu_count)
        else:
            return "0"

    # =========================================================================
    # Process Monitoring
    # =========================================================================

    async def _monitor_selfplay_process(
        self,
        job_id: str,
        proc: Any,
        output_dir: Path,
        board_type: str,
        num_players: int,
        job_type_str: str,
    ) -> None:
        """Monitor a selfplay process until completion.

        Delegates to P2POrchestrator._monitor_selfplay_process() if available.
        """
        try:
            if hasattr(self._p2p, "_monitor_selfplay_process"):
                await self._p2p._monitor_selfplay_process(
                    job_id, proc, output_dir, board_type, num_players, job_type_str
                )
                self._local_jobs_completed += 1
            else:
                # Simple fallback: wait for process to complete
                await asyncio.get_event_loop().run_in_executor(None, proc.wait)
                if proc.returncode == 0:
                    self._local_jobs_completed += 1
                else:
                    self._local_jobs_failed += 1
        except Exception as e:
            self._log_error(f"Error monitoring process {job_id}: {e}")
            self._local_jobs_failed += 1
