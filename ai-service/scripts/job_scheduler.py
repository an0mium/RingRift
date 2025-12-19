#!/usr/bin/env python3
"""
RingRift Job Scheduler - Distributed job scheduling across GPU cluster

Features:
- Priority-based job queue
- Node-aware job assignment (match GPU requirements)
- Automatic failover and retry
- Progress tracking and logging
- Integration with training pipeline

Usage:
    # Start the scheduler daemon
    python job_scheduler.py daemon

    # Submit jobs
    python job_scheduler.py submit selfplay --games 1000 --nodes primary_training
    python job_scheduler.py submit training --checkpoint latest
    python job_scheduler.py submit gauntlet --games 100

    # Monitor
    python job_scheduler.py status
    python job_scheduler.py logs --job-id abc123
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from queue import PriorityQueue
import signal

# Import cluster config
sys.path.insert(0, str(Path(__file__).parent))
from gpu_cluster_manager import ClusterConfig, check_node, ssh_command

# ============================================================================
# Configuration
# ============================================================================

SCHEDULER_DIR = Path(__file__).parent.parent / "data" / "scheduler"
JOBS_FILE = SCHEDULER_DIR / "jobs.json"
LOGS_DIR = SCHEDULER_DIR / "logs"

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class JobSpec:
    """Job specification."""
    id: str
    job_type: str  # selfplay, training, gauntlet, benchmark
    priority: int = 5  # 1=highest, 10=lowest
    params: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Execution state
    status: str = "pending"  # pending, queued, running, completed, failed, cancelled
    assigned_node: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    pid: Optional[int] = None

    def __lt__(self, other):
        """For priority queue comparison."""
        return self.priority < other.priority

# ============================================================================
# Job Templates
# ============================================================================

JOB_TEMPLATES = {
    "selfplay": {
        "command": "python scripts/generate_gpu_training_data.py --games {games} --output training_data/selfplay_{job_id}.jsonl",
        "requirements": {"min_vram_gb": 24, "roles": ["selfplay"]},
        "default_params": {"games": 1000},
        "priority": 3,
        "timeout_minutes": 60,
    },
    "training": {
        "command": "python scripts/train_nnue.py --checkpoint {checkpoint} --batch-size {batch_size} --epochs {epochs}",
        "requirements": {"min_vram_gb": 48, "roles": ["training"]},
        "default_params": {"checkpoint": "latest", "batch_size": 4096, "epochs": 10},
        "priority": 2,
        "timeout_minutes": 120,
    },
    "gauntlet": {
        "command": "python scripts/run_gauntlet.py --games {games} --board {board} --output results/gauntlet_{job_id}.json",
        "requirements": {"min_vram_gb": 16, "roles": ["gauntlet"]},
        "default_params": {"games": 100, "board": "square8"},
        "priority": 4,
        "timeout_minutes": 30,
    },
    "benchmark": {
        "command": "python scripts/benchmark_gpu_selfplay_cluster.py --games {games}",
        "requirements": {"min_vram_gb": 16, "roles": ["selfplay", "benchmark"]},
        "default_params": {"games": 100},
        "priority": 5,
        "timeout_minutes": 15,
    },
}

# ============================================================================
# Job Store
# ============================================================================

class JobStore:
    """Persistent job storage."""

    def __init__(self, path: Path = JOBS_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobSpec] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for jid, jdata in data.items():
                    self.jobs[jid] = JobSpec(**jdata)
            except Exception as e:
                print(f"Warning: Could not load jobs: {e}")

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({jid: asdict(job) for jid, job in self.jobs.items()}, f, indent=2)

    def add(self, job: JobSpec):
        with self._lock:
            self.jobs[job.id] = job
            self._save()

    def update(self, job: JobSpec):
        with self._lock:
            self.jobs[job.id] = job
            self._save()

    def get(self, job_id: str) -> Optional[JobSpec]:
        return self.jobs.get(job_id)

    def get_pending(self) -> List[JobSpec]:
        return [j for j in self.jobs.values() if j.status == "pending"]

    def get_running(self) -> List[JobSpec]:
        return [j for j in self.jobs.values() if j.status == "running"]

    def get_by_node(self, node: str) -> List[JobSpec]:
        return [j for j in self.jobs.values() if j.assigned_node == node and j.status == "running"]

# ============================================================================
# Scheduler
# ============================================================================

class JobScheduler:
    """Distributed job scheduler."""

    def __init__(self):
        self.config = ClusterConfig()
        self.store = JobStore()
        self.running = False
        self._lock = threading.Lock()

        # Ensure log directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def submit(self, job_type: str, params: Dict[str, Any] = None,
               priority: int = None, node_group: str = None) -> JobSpec:
        """Submit a new job."""
        template = JOB_TEMPLATES.get(job_type)
        if not template:
            raise ValueError(f"Unknown job type: {job_type}")

        # Merge params with defaults
        job_params = dict(template["default_params"])
        if params:
            job_params.update(params)

        # Create job
        job = JobSpec(
            id=str(uuid.uuid4())[:8],
            job_type=job_type,
            priority=priority or template["priority"],
            params=job_params,
            requirements=dict(template["requirements"]),
        )

        if node_group:
            job.requirements["node_group"] = node_group

        self.store.add(job)
        print(f"Submitted job {job.id} ({job_type})")
        return job

    def find_suitable_node(self, job: JobSpec) -> Optional[str]:
        """Find a node that can run this job."""
        requirements = job.requirements
        min_vram = requirements.get("min_vram_gb", 0)
        required_roles = requirements.get("roles", [])
        node_group = requirements.get("node_group")

        # Get candidate nodes
        if node_group:
            candidates = self.config.get_nodes_by_group(node_group)
        else:
            candidates = self.config.get_active_nodes()

        # Check each candidate
        for node_name in candidates:
            node_config = self.config.nodes.get(node_name)
            if not node_config:
                continue

            # Check VRAM
            if node_config.vram_gb < min_vram:
                continue

            # Check roles
            if required_roles:
                if not any(role in node_config.roles for role in required_roles):
                    continue

            # Check if node is online and not overloaded
            status = check_node(node_name, node_config)
            if not status.online:
                continue

            # Check GPU utilization (prefer less loaded nodes)
            if status.gpu_util and max(status.gpu_util) > 95:
                continue

            # Check how many jobs already running on this node
            running_jobs = self.store.get_by_node(node_name)
            if len(running_jobs) >= 2:  # Max 2 jobs per node
                continue

            return node_name

        return None

    def run_job(self, job: JobSpec, node: str):
        """Execute a job on a node."""
        template = JOB_TEMPLATES.get(job.job_type)
        if not template:
            job.status = "failed"
            job.error = "Unknown job type"
            self.store.update(job)
            return

        # Build command
        cmd_template = template["command"]
        cmd_params = dict(job.params)
        cmd_params["job_id"] = job.id

        try:
            command = cmd_template.format(**cmd_params)
        except KeyError as e:
            job.status = "failed"
            job.error = f"Missing parameter: {e}"
            self.store.update(job)
            return

        # Update job state
        job.status = "running"
        job.assigned_node = node
        job.started_at = datetime.now().isoformat()
        job.attempts += 1
        self.store.update(job)

        node_config = self.config.nodes[node]
        host = node_config.host

        # Execute remotely
        log_file = LOGS_DIR / f"{job.id}.log"
        full_command = f"cd ~/RingRift/ai-service && {command}"

        print(f"[{job.id}] Starting on {node}: {command[:60]}...")

        try:
            # Run with timeout
            timeout_sec = template.get("timeout_minutes", 60) * 60

            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, full_command],
                capture_output=True, text=True, timeout=timeout_sec
            )

            # Save output
            with open(log_file, "w") as f:
                f.write(f"=== Job {job.id} on {node} ===\n")
                f.write(f"Command: {command}\n")
                f.write(f"Started: {job.started_at}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode == 0:
                job.status = "completed"
                job.result = {"output_lines": len(result.stdout.split("\n"))}
                print(f"[{job.id}] Completed successfully")
            else:
                job.status = "failed"
                job.error = result.stderr[:500]
                print(f"[{job.id}] Failed: {job.error[:100]}")

        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "Timeout"
            print(f"[{job.id}] Timeout after {timeout_sec}s")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            print(f"[{job.id}] Error: {e}")

        job.completed_at = datetime.now().isoformat()
        self.store.update(job)

    def process_queue(self):
        """Process pending jobs."""
        pending = sorted(self.store.get_pending(), key=lambda j: (j.priority, j.created_at))

        for job in pending[:5]:  # Process up to 5 jobs per cycle
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.error = "Max attempts exceeded"
                self.store.update(job)
                continue

            node = self.find_suitable_node(job)
            if node:
                # Run in background thread
                thread = threading.Thread(target=self.run_job, args=(job, node))
                thread.daemon = True
                thread.start()

    def check_running_jobs(self):
        """Check status of running jobs."""
        running = self.store.get_running()

        for job in running:
            if not job.assigned_node:
                continue

            node_config = self.config.nodes.get(job.assigned_node)
            if not node_config:
                continue

            # Check if node is still online
            status = check_node(job.assigned_node, node_config)
            if not status.online:
                print(f"[{job.id}] Node {job.assigned_node} offline, marking for retry")
                job.status = "pending"
                job.assigned_node = None
                self.store.update(job)

    def daemon_loop(self, interval: int = 30):
        """Main scheduler loop."""
        self.running = True
        iteration = 0

        print(f"\n{'='*50}")
        print("  RingRift Job Scheduler Daemon")
        print(f"  Check interval: {interval}s")
        print(f"{'='*50}\n")

        def signal_handler(sig, frame):
            print("\nShutting down scheduler...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            pending = len(self.store.get_pending())
            running = len(self.store.get_running())

            print(f"[{timestamp}] #{iteration} | Pending: {pending} | Running: {running}")

            # Check running jobs
            self.check_running_jobs()

            # Process queue
            self.process_queue()

            time.sleep(interval)

        print("Scheduler stopped.")

    def status(self):
        """Print scheduler status."""
        print(f"\n{'='*60}")
        print("  RingRift Job Scheduler Status")
        print(f"{'='*60}\n")

        pending = self.store.get_pending()
        running = self.store.get_running()

        print(f"  Pending: {len(pending)}")
        print(f"  Running: {len(running)}\n")

        if running:
            print("  Running Jobs:")
            for job in running:
                elapsed = ""
                if job.started_at:
                    start = datetime.fromisoformat(job.started_at)
                    elapsed = str(datetime.now() - start).split(".")[0]
                print(f"    [{job.id}] {job.job_type} on {job.assigned_node} ({elapsed})")

        if pending:
            print("\n  Pending Jobs:")
            for job in pending[:10]:
                print(f"    [{job.id}] {job.job_type} (priority: {job.priority})")

        print()

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RingRift Job Scheduler")
    subparsers = parser.add_subparsers(dest="command")

    # Daemon
    sp = subparsers.add_parser("daemon", help="Start scheduler daemon")
    sp.add_argument("--interval", "-i", type=int, default=30, help="Check interval")

    # Submit
    sp = subparsers.add_parser("submit", help="Submit a job")
    sp.add_argument("job_type", choices=["selfplay", "training", "gauntlet", "benchmark"])
    sp.add_argument("--games", type=int, help="Number of games")
    sp.add_argument("--priority", "-p", type=int, help="Priority (1=high, 10=low)")
    sp.add_argument("--nodes", "-n", help="Node group")
    sp.add_argument("--checkpoint", help="Checkpoint for training")

    # Status
    subparsers.add_parser("status", help="Show scheduler status")

    # Logs
    sp = subparsers.add_parser("logs", help="View job logs")
    sp.add_argument("--job-id", "-j", required=True, help="Job ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    scheduler = JobScheduler()

    if args.command == "daemon":
        scheduler.daemon_loop(args.interval)

    elif args.command == "submit":
        params = {}
        if args.games:
            params["games"] = args.games
        if args.checkpoint:
            params["checkpoint"] = args.checkpoint

        scheduler.submit(
            args.job_type,
            params=params,
            priority=args.priority,
            node_group=args.nodes
        )

    elif args.command == "status":
        scheduler.status()

    elif args.command == "logs":
        log_file = LOGS_DIR / f"{args.job_id}.log"
        if log_file.exists():
            print(log_file.read_text())
        else:
            print(f"No logs found for job {args.job_id}")

if __name__ == "__main__":
    main()
