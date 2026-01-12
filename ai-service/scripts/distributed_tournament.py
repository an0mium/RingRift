#!/usr/bin/env python3
"""Distributed tournament runner that shards models across cluster nodes.

Transfers model shards to cluster nodes, runs games in parallel, and aggregates results.
Designed for 8-24 hour tournament reduction to 2-4 hours.
"""
from __future__ import annotations


import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Available cluster nodes for tournament
# Uses direct SSH IPs where available (more reliable than Tailscale)
CLUSTER_NODES = [
    {
        "name": "lambda-gh200-1",
        "ssh_host": "192.222.57.210",  # Direct SSH
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 16,  # GH200 can handle many parallel games
    },
    {
        "name": "lambda-gh200-2",
        "ssh_host": "192.222.57.140",  # Direct SSH
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 16,
    },
    {
        "name": "lambda-gh200-4",
        "ssh_host": "192.222.57.184",  # Direct SSH
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 16,
    },
    {
        "name": "lambda-gh200-5",
        "ssh_host": "192.222.58.171",  # Direct SSH
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 16,
    },
    {
        "name": "lambda-gh200-8",
        "ssh_host": "100.121.230.110",  # Tailscale (no direct IP known)
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 8,
    },
    {
        "name": "lambda-gh200-11",
        "ssh_host": "100.106.87.89",  # Tailscale (no direct IP known)
        "ssh_user": "ubuntu",
        "gpu": "GH200 (96GB)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 8,
    },
    {
        "name": "vultr-a100",
        "ssh_host": "100.94.201.92",  # Tailscale
        "ssh_user": "ubuntu",
        "gpu": "A100 (20GB vGPU)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 4,
    },
    # mac-studio runs locally, no SSH needed
    {
        "name": "mac-studio",
        "ssh_host": "localhost",
        "ssh_user": "armand",
        "gpu": "M3 Max (MPS)",
        "ringrift_path": "~/ringrift/ai-service",
        "workers": 6,
        "local": True,  # Special flag for local execution
    },
]


@dataclass
class ModelShard:
    """A shard of models assigned to a node."""
    node_name: str
    models: list[Path]
    config: str  # e.g., "hex8_2p"


@dataclass
class NodeResult:
    """Results from a single node."""
    node_name: str
    config: str
    games_played: int
    results: list[dict]
    ratings: dict[str, float]
    duration_seconds: float
    error: str | None = None


def check_node_connectivity(node: dict) -> tuple[str, bool]:
    """Check if a node is reachable via SSH."""
    # Local nodes are always available
    if node.get("local"):
        return node["name"], True

    cmd = [
        "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        f"{node['ssh_user']}@{node['ssh_host']}", "echo", "ok"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return node["name"], result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return node["name"], False


def get_available_nodes() -> list[dict]:
    """Check which nodes are available."""
    logger.info("Checking node connectivity...")
    available = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_node_connectivity, n): n for n in CLUSTER_NODES}
        for future in as_completed(futures):
            name, ok = future.result()
            node = futures[future]
            if ok:
                available.append(node)
                logger.info(f"  ✓ {name}: {node['gpu']}")
            else:
                logger.warning(f"  ✗ {name}: unreachable")

    return available


async def scan_models(source_path: Path) -> dict[str, list[Path]]:
    """Scan models and group by config."""
    from app.utils.model_deduplicator import ModelDeduplicator

    logger.info(f"Scanning models from {source_path}...")
    dedup = ModelDeduplicator()
    unique_models = await dedup.scan_directory(source_path)

    # Group by config (board_type + num_players)
    models_by_config: dict[str, list[Path]] = {}
    for model in unique_models:
        config = f"{model.board_type}_{model.num_players}p"
        if config not in models_by_config:
            models_by_config[config] = []
        models_by_config[config].append(model.canonical_path)

    logger.info(f"Found {len(unique_models)} unique models across {len(models_by_config)} configs")
    for config, models in sorted(models_by_config.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {config}: {len(models)} models")

    return models_by_config


def shard_models(
    models: list[Path],
    nodes: list[dict],
    config: str,
) -> list[ModelShard]:
    """Distribute models across available nodes."""
    shards = []
    n_nodes = len(nodes)

    # Distribute evenly
    for i, node in enumerate(nodes):
        start = i * len(models) // n_nodes
        end = (i + 1) * len(models) // n_nodes
        shard_models = models[start:end]
        if shard_models:
            shards.append(ModelShard(
                node_name=node["name"],
                models=shard_models,
                config=config,
            ))

    return shards


def transfer_shard(node: dict, shard: ModelShard, dest_dir: str = "/tmp/tournament_models") -> bool:
    """Transfer model shard to a node."""
    logger.info(f"Transferring {len(shard.models)} models to {node['name']}...")

    # Create destination directory
    ssh_target = f"{node['ssh_user']}@{node['ssh_host']}"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh_target, f"mkdir -p {dest_dir}/{shard.config}"],
        capture_output=True,
        timeout=30,
    )

    # Transfer models using rsync
    model_list_file = f"/tmp/models_{node['name']}_{shard.config}.txt"
    with open(model_list_file, "w") as f:
        for model in shard.models:
            f.write(f"{model}\n")

    # Use rsync with files-from
    cmd = [
        "rsync", "-avz", "--progress",
        "--files-from", model_list_file,
        "/",  # Root since paths are absolute
        f"{ssh_target}:{dest_dir}/{shard.config}/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"rsync to {node['name']} failed: {result.stderr[:200]}")
            return False
        logger.info(f"  ✓ Transferred to {node['name']}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"rsync to {node['name']} timed out")
        return False
    except Exception as e:
        logger.error(f"rsync to {node['name']} error: {e}")
        return False


def run_node_tournament(
    node: dict,
    shard: ModelShard,
    baselines: list[str],
    games_per_pairing: int,
    model_dir: str = "/tmp/tournament_models",
) -> NodeResult:
    """Run tournament on a single node."""
    start_time = time.time()
    ssh_target = f"{node['ssh_user']}@{node['ssh_host']}"

    board_type, num_players = shard.config.rsplit("_", 1)
    num_players = int(num_players.replace("p", ""))

    # Build remote command
    remote_cmd = f"""
cd {node['ringrift_path']} && \
source venv/bin/activate && \
PYTHONPATH=. python3 -c "
import json
import sys
sys.path.insert(0, '.')

from scripts.run_massive_tournament import run_local_tournament
from pathlib import Path

model_paths = list(Path('{model_dir}/{shard.config}').glob('**/*.pth'))
print(f'Found {{len(model_paths)}} models', file=sys.stderr)

results, ratings = run_local_tournament(
    model_paths=[str(p) for p in model_paths],
    baselines={json.dumps(baselines)},
    games_per_pairing={games_per_pairing},
    board_type='{board_type}',
    num_players={num_players},
    max_workers={node['workers']},
)

print(json.dumps({{'games': len(results), 'ratings': ratings, 'results': results}}))
"
"""

    logger.info(f"Running tournament on {node['name']} ({len(shard.models)} models)...")

    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60", ssh_target, remote_cmd],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per node
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return NodeResult(
                node_name=node["name"],
                config=shard.config,
                games_played=0,
                results=[],
                ratings={},
                duration_seconds=duration,
                error=f"Exit {result.returncode}: {result.stderr[:500]}",
            )

        # Parse JSON output (last line of stdout)
        output_lines = result.stdout.strip().split("\n")
        for line in reversed(output_lines):
            try:
                data = json.loads(line)
                return NodeResult(
                    node_name=node["name"],
                    config=shard.config,
                    games_played=data.get("games", 0),
                    results=data.get("results", []),
                    ratings=data.get("ratings", {}),
                    duration_seconds=duration,
                )
            except json.JSONDecodeError:
                continue

        return NodeResult(
            node_name=node["name"],
            config=shard.config,
            games_played=0,
            results=[],
            ratings={},
            duration_seconds=duration,
            error=f"Could not parse output: {result.stdout[-500:]}",
        )

    except subprocess.TimeoutExpired:
        return NodeResult(
            node_name=node["name"],
            config=shard.config,
            games_played=0,
            results=[],
            ratings={},
            duration_seconds=time.time() - start_time,
            error="Timeout after 2 hours",
        )
    except Exception as e:
        return NodeResult(
            node_name=node["name"],
            config=shard.config,
            games_played=0,
            results=[],
            ratings={},
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


def aggregate_results(node_results: list[NodeResult]) -> dict[str, float]:
    """Aggregate Elo ratings from multiple nodes using combined match results."""
    from scripts.run_massive_tournament import calculate_elo

    # Combine all match results
    all_results = []
    for nr in node_results:
        all_results.extend(nr.results)

    if not all_results:
        return {}

    # Recalculate Elo from combined results
    return calculate_elo(all_results, anchor_player="random", anchor_elo=400)


async def run_distributed_tournament(
    source_path: Path,
    configs: list[str] | None = None,
    baselines: list[str] | None = None,
    games_per_pairing: int = 10,
    skip_transfer: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    """Run distributed tournament across cluster nodes."""

    baselines = baselines or ["random", "heuristic", "mcts_light", "mcts_medium"]

    # Step 1: Check available nodes
    available_nodes = get_available_nodes()
    if not available_nodes:
        logger.error("No cluster nodes available!")
        return {}

    logger.info(f"\n{len(available_nodes)} nodes available for tournament")

    # Step 2: Scan models
    models_by_config = await scan_models(source_path)

    # Filter out invalid configs (unknown board types, 0-player)
    valid_configs = {}
    for k, v in models_by_config.items():
        if "unknown" in k or "_0p" in k:
            logger.info(f"  Skipping invalid config: {k} ({len(v)} models)")
            continue
        valid_configs[k] = v
    models_by_config = valid_configs

    if configs:
        models_by_config = {k: v for k, v in models_by_config.items() if k in configs}

    if dry_run:
        logger.info("\n=== DRY RUN - Tournament Plan ===")
        total_games = 0
        for config, models in models_by_config.items():
            games = len(models) * len(baselines) * games_per_pairing
            total_games += games
            logger.info(f"  {config}: {len(models)} models × {len(baselines)} baselines × {games_per_pairing} = {games} games")

            # Show shard distribution
            shards = shard_models(models, available_nodes, config)
            for shard in shards:
                logger.info(f"    → {shard.node_name}: {len(shard.models)} models")

        logger.info(f"\nTotal: {total_games} games across {len(available_nodes)} nodes")
        est_time = total_games / (sum(n["workers"] for n in available_nodes) * 2)  # ~2 games/worker/min
        logger.info(f"Estimated time: {est_time/60:.1f} hours (with full parallelism)")
        return {}

    # Step 3: Run tournament for each config
    all_ratings: dict[str, dict[str, float]] = {}
    results_dir = Path("results/distributed_tournament")
    results_dir.mkdir(parents=True, exist_ok=True)

    for config, models in models_by_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config} ({len(models)} models)")
        logger.info(f"{'='*60}")

        # Shard models
        shards = shard_models(models, available_nodes, config)

        # Transfer shards (if not skipped)
        if not skip_transfer:
            logger.info("\nTransferring model shards to nodes...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                node_map = {n["name"]: n for n in available_nodes}
                futures = []
                for shard in shards:
                    node = node_map[shard.node_name]
                    futures.append(executor.submit(transfer_shard, node, shard))

                for future in as_completed(futures):
                    future.result()

        # Run tournaments in parallel
        logger.info("\nRunning tournaments on all nodes...")
        node_results: list[NodeResult] = []
        node_map = {n["name"]: n for n in available_nodes}

        with ThreadPoolExecutor(max_workers=len(available_nodes)) as executor:
            futures = {}
            for shard in shards:
                node = node_map[shard.node_name]
                future = executor.submit(
                    run_node_tournament, node, shard, baselines, games_per_pairing
                )
                futures[future] = shard.node_name

            for future in as_completed(futures):
                node_name = futures[future]
                result = future.result()
                node_results.append(result)

                if result.error:
                    logger.error(f"  ✗ {node_name}: {result.error[:100]}")
                else:
                    logger.info(f"  ✓ {node_name}: {result.games_played} games in {result.duration_seconds:.0f}s")

        # Aggregate results
        combined_ratings = aggregate_results(node_results)
        all_ratings[config] = combined_ratings

        # Save intermediate results
        result_file = results_dir / f"{config}_results.json"
        with open(result_file, "w") as f:
            json.dump({
                "config": config,
                "total_games": sum(nr.games_played for nr in node_results),
                "ratings": combined_ratings,
                "node_results": [
                    {
                        "node": nr.node_name,
                        "games": nr.games_played,
                        "duration": nr.duration_seconds,
                        "error": nr.error,
                    }
                    for nr in node_results
                ],
            }, f, indent=2)

        # Print top models
        if combined_ratings:
            logger.info(f"\nTop 10 models for {config}:")
            sorted_ratings = sorted(combined_ratings.items(), key=lambda x: -x[1])[:10]
            for name, elo in sorted_ratings:
                logger.info(f"  {elo:.0f}: {name}")

    return all_ratings


async def main():
    parser = argparse.ArgumentParser(description="Run distributed tournament across cluster")
    parser.add_argument(
        "--source",
        type=str,
        default="/Volumes/RingRift-Data/model_checkpoints",
        help="Path to model archive",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to run (e.g., hex8_2p). If not specified, runs all.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Games per (model, baseline) pair",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="random,heuristic,mcts_light,mcts_medium",
        help="Comma-separated baseline opponents",
    )
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip model transfer (if models already on nodes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without running",
    )

    args = parser.parse_args()

    configs = [args.config] if args.config else None
    baselines = args.baselines.split(",")

    await run_distributed_tournament(
        source_path=Path(args.source),
        configs=configs,
        baselines=baselines,
        games_per_pairing=args.games,
        skip_transfer=args.skip_transfer,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
