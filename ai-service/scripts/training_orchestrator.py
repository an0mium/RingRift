#!/usr/bin/env python3
"""
RingRift Training Orchestrator - Unified training pipeline

Coordinates:
1. GPU self-play game generation across cluster
2. Training data collection and preprocessing
3. NNUE model training with checkpointing
4. Validation gauntlets after training
5. Model versioning and deployment

Usage:
    # Run full training cycle
    python training_orchestrator.py run --cycles 10

    # Generate self-play data only
    python training_orchestrator.py selfplay --games 10000

    # Train from existing data
    python training_orchestrator.py train --data-dir training_data/

    # Run validation gauntlet
    python training_orchestrator.py validate --checkpoint models/nnue_v5.pt
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu_cluster_manager import ClusterConfig, check_node, ssh_command, check_all_nodes

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs" / "orchestrator"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Training configuration
DEFAULT_CONFIG = {
    "selfplay": {
        "games_per_node": 500,
        "batch_size_multiplier": 64,
        "ai_types": ["gumbel_mcts", "maxn"],
        "board_types": ["square8", "hex8"],
        "num_players": [2, 3, 4],
        "timeout_minutes": 60,
    },
    "training": {
        "batch_size": 4096,
        "learning_rate": 0.001,
        "epochs_per_cycle": 5,
        "validation_split": 0.1,
        "checkpoint_interval": 1000,
        "early_stopping_patience": 3,
    },
    "validation": {
        "games_per_matchup": 50,
        "opponents": ["heuristic", "previous_best"],
        "win_threshold": 0.55,  # Must win 55% to be promoted
    },
    "cluster": {
        "selfplay_nodes": ["lambda-gh200-d", "lambda-gh200-e", "lambda-gh200-f",
                          "lambda-gh200-g", "lambda-gh200-i"],
        "training_node": "lambda-gh200-d",
        "gauntlet_nodes": ["lambda-gh200-h", "lambda-gh200-k", "lambda-h100"],
    },
}

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TrainingCycle:
    """Represents one complete training cycle."""
    id: str
    started_at: str
    status: str = "starting"  # starting, selfplay, training, validating, completed, failed

    selfplay_games: int = 0
    selfplay_nodes: List[str] = field(default_factory=list)
    selfplay_data_path: Optional[str] = None

    training_epochs: int = 0
    training_loss: float = 0.0
    checkpoint_path: Optional[str] = None

    validation_winrate: float = 0.0
    promoted: bool = False

    completed_at: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# Training Orchestrator
# ============================================================================

class TrainingOrchestrator:
    """Coordinates the full training pipeline."""

    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.cluster = ClusterConfig()

        # Ensure directories exist
        for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self.current_cycle: Optional[TrainingCycle] = None

    def log(self, message: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)

        if self.current_cycle:
            log_file = LOGS_DIR / f"cycle_{self.current_cycle.id}.log"
            with open(log_file, "a") as f:
                f.write(log_line + "\n")

    def get_available_nodes(self, role: str = "selfplay") -> List[str]:
        """Get list of available nodes for a role."""
        node_list = self.config["cluster"].get(f"{role}_nodes", [])

        # Check which are online
        available = []
        for node_name in node_list:
            if node_name in self.cluster.nodes:
                status = check_node(node_name, self.cluster.nodes[node_name])
                if status.online:
                    available.append(node_name)

        return available

    def run_selfplay_on_node(self, node: str, games: int, output_dir: Path) -> Dict:
        """Run self-play on a single node."""
        node_config = self.cluster.nodes.get(node)
        if not node_config:
            return {"node": node, "success": False, "error": "Node not found"}

        host = node_config.host
        output_file = f"selfplay_{node}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        remote_output = f"~/RingRift/ai-service/training_data/{output_file}"

        # Command to run on remote node
        command = (
            f"cd ~/RingRift/ai-service && "
            f"python scripts/generate_gpu_training_data.py "
            f"--num-games {games} --output {remote_output} 2>&1"
        )

        self.log(f"  Starting self-play on {node}: {games} games")

        try:
            timeout = self.config["selfplay"]["timeout_minutes"] * 60
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
                capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                return {"node": node, "success": False, "error": result.stderr[:200]}

            # Copy data back
            local_output = output_dir / output_file
            scp_result = subprocess.run(
                ["scp", f"{host}:{remote_output}", str(local_output)],
                capture_output=True, timeout=120
            )

            if scp_result.returncode == 0 and local_output.exists():
                # Count games
                with open(local_output) as f:
                    game_count = sum(1 for _ in f)

                self.log(f"  {node}: {game_count} games collected")
                return {"node": node, "success": True, "games": game_count, "file": str(local_output)}
            else:
                return {"node": node, "success": False, "error": "Failed to copy data"}

        except subprocess.TimeoutExpired:
            return {"node": node, "success": False, "error": "Timeout"}
        except Exception as e:
            return {"node": node, "success": False, "error": str(e)}

    def run_selfplay_phase(self, total_games: int) -> tuple:
        """Run self-play across cluster."""
        self.log("=== Self-Play Phase ===")

        nodes = self.get_available_nodes("selfplay")
        if not nodes:
            raise RuntimeError("No self-play nodes available")

        self.log(f"Available nodes: {', '.join(nodes)}")

        # Distribute games across nodes
        games_per_node = total_games // len(nodes)

        # Create output directory
        output_dir = DATA_DIR / f"selfplay_{self.current_cycle.id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            futures = {
                executor.submit(self.run_selfplay_on_node, node, games_per_node, output_dir): node
                for node in nodes
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Summarize
        total_collected = sum(r.get("games", 0) for r in results if r.get("success"))
        successful_nodes = [r["node"] for r in results if r.get("success")]

        self.log(f"Self-play complete: {total_collected} games from {len(successful_nodes)} nodes")

        # Merge data files
        merged_file = DATA_DIR / f"training_data_{self.current_cycle.id}.jsonl"
        with open(merged_file, "w") as outf:
            for result in results:
                if result.get("success") and result.get("file"):
                    with open(result["file"]) as inf:
                        for line in inf:
                            outf.write(line)

        return total_collected, str(merged_file)

    def run_training_phase(self, data_path: str, epochs: int) -> tuple:
        """Run NNUE training on data."""
        self.log("=== Training Phase ===")

        training_node = self.config["cluster"]["training_node"]
        if training_node not in self.cluster.nodes:
            raise RuntimeError(f"Training node {training_node} not configured")

        node_config = self.cluster.nodes[training_node]
        status = check_node(training_node, node_config)
        if not status.online:
            raise RuntimeError(f"Training node {training_node} offline")

        host = node_config.host

        # Copy training data to node
        self.log(f"Copying training data to {training_node}...")
        subprocess.run(
            ["scp", data_path, f"{host}:~/RingRift/ai-service/training_data/"],
            capture_output=True, timeout=300
        )

        # Run training
        checkpoint_name = f"nnue_{self.current_cycle.id}.pt"
        remote_checkpoint = f"~/RingRift/ai-service/models/{checkpoint_name}"
        data_filename = Path(data_path).name

        command = (
            f"cd ~/RingRift/ai-service && "
            f"python -m app.training.train_nnue "
            f"--data training_data/{data_filename} "
            f"--epochs {epochs} "
            f"--batch-size {self.config['training']['batch_size']} "
            f"--lr {self.config['training']['learning_rate']} "
            f"--output {remote_checkpoint} 2>&1"
        )

        self.log(f"Starting training on {training_node}: {epochs} epochs")

        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
                capture_output=True, text=True, timeout=7200  # 2 hour timeout
            )

            # Parse output for final loss
            final_loss = 0.0
            for line in result.stdout.split("\n"):
                if "loss" in line.lower():
                    try:
                        # Extract loss value
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if "loss" in p.lower() and i + 1 < len(parts):
                                final_loss = float(parts[i + 1].strip(":,"))
                                break
                    except (ValueError, IndexError):
                        pass

            if result.returncode != 0:
                self.log(f"Training failed: {result.stderr[:200]}")
                return 0.0, None

            # Copy checkpoint back
            local_checkpoint = MODELS_DIR / checkpoint_name
            subprocess.run(
                ["scp", f"{host}:{remote_checkpoint}", str(local_checkpoint)],
                capture_output=True, timeout=120
            )

            if local_checkpoint.exists():
                self.log(f"Training complete. Loss: {final_loss:.4f}")
                return final_loss, str(local_checkpoint)
            else:
                return final_loss, None

        except subprocess.TimeoutExpired:
            self.log("Training timeout")
            return 0.0, None
        except Exception as e:
            self.log(f"Training error: {e}")
            return 0.0, None

    def run_validation_phase(self, checkpoint_path: str) -> float:
        """Validate model with gauntlet."""
        self.log("=== Validation Phase ===")

        gauntlet_nodes = self.get_available_nodes("gauntlet")
        if not gauntlet_nodes:
            self.log("No gauntlet nodes available, skipping validation")
            return 0.5  # Neutral

        node = gauntlet_nodes[0]
        node_config = self.cluster.nodes[node]
        host = node_config.host

        # Copy checkpoint to gauntlet node
        checkpoint_name = Path(checkpoint_path).name
        subprocess.run(
            ["scp", checkpoint_path, f"{host}:~/RingRift/ai-service/models/"],
            capture_output=True, timeout=60
        )

        games = self.config["validation"]["games_per_matchup"]
        command = (
            f"cd ~/RingRift/ai-service && "
            f"python scripts/run_gauntlet.py "
            f"--checkpoint models/{checkpoint_name} "
            f"--games {games} --json 2>&1"
        )

        self.log(f"Running validation gauntlet on {node}...")

        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
                capture_output=True, text=True, timeout=1800
            )

            # Parse win rate from output
            win_rate = 0.5
            for line in result.stdout.split("\n"):
                if "win" in line.lower() and "rate" in line.lower():
                    try:
                        parts = line.split()
                        for p in parts:
                            if "%" in p:
                                win_rate = float(p.strip("%")) / 100
                                break
                            try:
                                val = float(p)
                                if 0 <= val <= 1:
                                    win_rate = val
                                    break
                            except ValueError:
                                pass
                    except (ValueError, IndexError):
                        pass

            self.log(f"Validation complete. Win rate: {win_rate:.1%}")
            return win_rate

        except Exception as e:
            self.log(f"Validation error: {e}")
            return 0.5

    def run_cycle(self, games: int, epochs: int) -> TrainingCycle:
        """Run a complete training cycle."""
        cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_cycle = TrainingCycle(
            id=cycle_id,
            started_at=datetime.now().isoformat(),
        )

        self.log(f"\n{'='*60}")
        self.log(f"  Training Cycle {cycle_id}")
        self.log(f"  Games: {games} | Epochs: {epochs}")
        self.log(f"{'='*60}\n")

        try:
            # Phase 1: Self-play
            self.current_cycle.status = "selfplay"
            game_count, data_path = self.run_selfplay_phase(games)
            self.current_cycle.selfplay_games = game_count
            self.current_cycle.selfplay_data_path = data_path

            if game_count < games * 0.5:
                raise RuntimeError(f"Insufficient games collected: {game_count}/{games}")

            # Phase 2: Training
            self.current_cycle.status = "training"
            loss, checkpoint = self.run_training_phase(data_path, epochs)
            self.current_cycle.training_loss = loss
            self.current_cycle.training_epochs = epochs
            self.current_cycle.checkpoint_path = checkpoint

            if not checkpoint:
                raise RuntimeError("Training failed to produce checkpoint")

            # Phase 3: Validation
            self.current_cycle.status = "validating"
            win_rate = self.run_validation_phase(checkpoint)
            self.current_cycle.validation_winrate = win_rate

            # Check if model should be promoted
            threshold = self.config["validation"]["win_threshold"]
            if win_rate >= threshold:
                self.current_cycle.promoted = True
                # Copy to best model
                best_path = MODELS_DIR / "nnue_best.pt"
                shutil.copy(checkpoint, best_path)
                self.log(f"Model promoted as new best (win rate: {win_rate:.1%})")
            else:
                self.log(f"Model not promoted (win rate: {win_rate:.1%} < {threshold:.1%})")

            self.current_cycle.status = "completed"
            self.current_cycle.completed_at = datetime.now().isoformat()

        except Exception as e:
            self.current_cycle.status = "failed"
            self.current_cycle.error = str(e)
            self.current_cycle.completed_at = datetime.now().isoformat()
            self.log(f"Cycle failed: {e}")

        # Save cycle info
        cycle_file = LOGS_DIR / f"cycle_{cycle_id}.json"
        with open(cycle_file, "w") as f:
            json.dump({
                "id": self.current_cycle.id,
                "status": self.current_cycle.status,
                "started_at": self.current_cycle.started_at,
                "completed_at": self.current_cycle.completed_at,
                "selfplay_games": self.current_cycle.selfplay_games,
                "training_loss": self.current_cycle.training_loss,
                "validation_winrate": self.current_cycle.validation_winrate,
                "promoted": self.current_cycle.promoted,
                "checkpoint": self.current_cycle.checkpoint_path,
                "error": self.current_cycle.error,
            }, f, indent=2)

        return self.current_cycle

    def run_continuous(self, cycles: int, games_per_cycle: int, epochs_per_cycle: int):
        """Run multiple training cycles."""
        self.log(f"\n{'='*60}")
        self.log(f"  Starting Continuous Training")
        self.log(f"  Cycles: {cycles} | Games/cycle: {games_per_cycle} | Epochs/cycle: {epochs_per_cycle}")
        self.log(f"{'='*60}\n")

        results = []
        for i in range(cycles):
            self.log(f"\n--- Cycle {i+1}/{cycles} ---\n")
            cycle = self.run_cycle(games_per_cycle, epochs_per_cycle)
            results.append(cycle)

            if cycle.status == "failed":
                self.log("Cycle failed, waiting 5 minutes before retry...")
                time.sleep(300)

        # Summary
        self.log("\n=== Training Summary ===")
        completed = sum(1 for c in results if c.status == "completed")
        promoted = sum(1 for c in results if c.promoted)
        self.log(f"Completed: {completed}/{cycles}")
        self.log(f"Promoted: {promoted}")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RingRift Training Orchestrator")
    subparsers = parser.add_subparsers(dest="command")

    # Run full training
    sp = subparsers.add_parser("run", help="Run training cycles")
    sp.add_argument("--cycles", "-c", type=int, default=1, help="Number of cycles")
    sp.add_argument("--games", "-g", type=int, default=5000, help="Games per cycle")
    sp.add_argument("--epochs", "-e", type=int, default=5, help="Epochs per cycle")

    # Self-play only
    sp = subparsers.add_parser("selfplay", help="Generate self-play data")
    sp.add_argument("--games", "-g", type=int, default=10000, help="Number of games")

    # Train only
    sp = subparsers.add_parser("train", help="Train from existing data")
    sp.add_argument("--data", "-d", required=True, help="Training data file")
    sp.add_argument("--epochs", "-e", type=int, default=10, help="Training epochs")

    # Validate
    sp = subparsers.add_parser("validate", help="Validate a checkpoint")
    sp.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path")

    # Status
    subparsers.add_parser("status", help="Show training status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    orchestrator = TrainingOrchestrator()

    if args.command == "run":
        orchestrator.run_continuous(args.cycles, args.games, args.epochs)

    elif args.command == "selfplay":
        orchestrator.current_cycle = TrainingCycle(
            id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat()
        )
        games, path = orchestrator.run_selfplay_phase(args.games)
        print(f"\nCollected {games} games -> {path}")

    elif args.command == "train":
        orchestrator.current_cycle = TrainingCycle(
            id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat()
        )
        loss, checkpoint = orchestrator.run_training_phase(args.data, args.epochs)
        print(f"\nTraining complete. Loss: {loss:.4f} -> {checkpoint}")

    elif args.command == "validate":
        orchestrator.current_cycle = TrainingCycle(
            id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat()
        )
        win_rate = orchestrator.run_validation_phase(args.checkpoint)
        print(f"\nValidation win rate: {win_rate:.1%}")

    elif args.command == "status":
        print("\n=== Recent Training Cycles ===\n")
        cycle_files = sorted(LOGS_DIR.glob("cycle_*.json"), reverse=True)[:10]
        for cf in cycle_files:
            with open(cf) as f:
                data = json.load(f)
            promoted = "PROMOTED" if data.get("promoted") else ""
            print(f"  {data['id']} | {data['status']:<10} | Games: {data.get('selfplay_games', 0):>5} | "
                  f"Loss: {data.get('training_loss', 0):.4f} | WR: {data.get('validation_winrate', 0):.1%} {promoted}")
        print()

if __name__ == "__main__":
    main()
