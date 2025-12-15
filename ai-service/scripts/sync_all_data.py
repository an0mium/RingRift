#!/usr/bin/env python3
"""Sync all game data from distributed hosts to a central repository.

This script collects game data from all cluster hosts and consolidates
it into a single database for training.

Usage:
    python scripts/sync_all_data.py --target lambda-h100
    python scripts/sync_all_data.py --target local --output data/games/consolidated.db
"""

import argparse
import asyncio
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class HostConfig:
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    remote_path: str = "~/ringrift/ai-service/data/games"


def load_hosts() -> Dict[str, HostConfig]:
    """Load host configuration from YAML."""
    hosts_path = AI_SERVICE_ROOT / "config" / "remote_hosts.yaml"

    with open(hosts_path) as f:
        data = yaml.safe_load(f)

    hosts = {}

    # Standard hosts
    if "standard_hosts" in data:
        for name, config in data["standard_hosts"].items():
            # Skip Tailscale duplicates (lambda_gh200_*)
            if name.startswith("lambda_gh200"):
                continue
            hosts[name] = HostConfig(
                name=name,
                ssh_host=config.get("ssh_host", ""),
                ssh_user=config.get("ssh_user", "ubuntu"),
                ssh_port=config.get("ssh_port", 22),
                remote_path=config.get("remote_path", "~/ringrift/ai-service/data/games"),
            )

    return hosts


async def get_host_game_count(host: HostConfig) -> Tuple[str, int, List[str]]:
    """Get game count and database list from a host."""
    ssh_target = f"{host.ssh_user}@{host.ssh_host}"
    port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

    # Get list of databases and their game counts
    script = """
import sqlite3, glob, json, os
os.chdir(os.path.expanduser('~/ringrift/ai-service'))
dbs = glob.glob('data/games/*.db')
results = []
for db in dbs:
    if 'schema' in db:
        continue
    try:
        count = sqlite3.connect(db).execute('SELECT COUNT(*) FROM games').fetchone()[0]
        results.append({'db': db, 'count': count})
    except:
        pass
print(json.dumps(results))
"""

    cmd = f'ssh -o ConnectTimeout=10 -o BatchMode=yes {port_arg} {ssh_target} "python3 -c \\"{script}\\"" 2>/dev/null'

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)

        import json
        results = json.loads(stdout.decode().strip())
        total = sum(r["count"] for r in results)
        dbs = [r["db"] for r in results if r["count"] > 0]
        return (host.name, total, dbs)
    except Exception as e:
        return (host.name, 0, [])


async def sync_database(
    host: HostConfig,
    remote_db: str,
    local_dir: Path,
    bwlimit_kbps: int = 10000
) -> bool:
    """Sync a single database from remote host."""
    ssh_target = f"{host.ssh_user}@{host.ssh_host}"
    port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

    # Create host-specific directory
    host_dir = local_dir / host.name
    host_dir.mkdir(parents=True, exist_ok=True)

    # Rsync the database
    db_name = Path(remote_db).name
    local_path = host_dir / db_name

    cmd = f'rsync -avz --progress --bwlimit={bwlimit_kbps} -e "ssh -o ConnectTimeout=10 {port_arg}" {ssh_target}:{remote_db} {local_path}'

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=600)
        return proc.returncode == 0
    except Exception as e:
        print(f"  Error syncing {remote_db} from {host.name}: {e}")
        return False


async def merge_databases(source_dir: Path, output_db: Path) -> int:
    """Merge all synced databases into a single consolidated database."""
    print(f"\nMerging databases into {output_db}...")

    # Find all synced databases
    db_files = list(source_dir.rglob("*.db"))
    if not db_files:
        print("No databases found to merge")
        return 0

    # Create output database with schema
    if output_db.exists():
        output_db.unlink()

    # Use the first database as template for schema
    first_db = db_files[0]
    conn_template = sqlite3.connect(first_db)
    schema = conn_template.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='games'"
    ).fetchone()
    conn_template.close()

    if not schema:
        print("Could not find games table schema")
        return 0

    conn_out = sqlite3.connect(output_db)
    conn_out.execute(schema[0])
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_game_id ON games(game_id)")

    # Track unique game IDs to deduplicate
    seen_game_ids = set()
    total_games = 0
    duplicates = 0

    for db_file in db_files:
        try:
            conn_in = sqlite3.connect(db_file)
            cursor = conn_in.execute("SELECT * FROM games")
            columns = [desc[0] for desc in cursor.description]
            game_id_idx = columns.index("game_id") if "game_id" in columns else 0

            batch = []
            for row in cursor:
                game_id = row[game_id_idx]
                if game_id in seen_game_ids:
                    duplicates += 1
                    continue
                seen_game_ids.add(game_id)
                batch.append(row)

                if len(batch) >= 1000:
                    placeholders = ",".join(["?" for _ in columns])
                    conn_out.executemany(
                        f"INSERT INTO games ({','.join(columns)}) VALUES ({placeholders})",
                        batch
                    )
                    total_games += len(batch)
                    batch = []

            if batch:
                placeholders = ",".join(["?" for _ in columns])
                conn_out.executemany(
                    f"INSERT INTO games ({','.join(columns)}) VALUES ({placeholders})",
                    batch
                )
                total_games += len(batch)

            conn_in.close()
            print(f"  Merged {db_file.name}: +{len(batch)} games")

        except Exception as e:
            print(f"  Error merging {db_file}: {e}")

    conn_out.commit()
    conn_out.close()

    print(f"\nMerge complete: {total_games:,} unique games ({duplicates:,} duplicates removed)")
    return total_games


async def main():
    parser = argparse.ArgumentParser(description="Sync all game data from cluster")
    parser.add_argument("--target", choices=["local", "lambda-h100"], default="local",
                       help="Where to sync data to")
    parser.add_argument("--output", type=str, default="data/games/consolidated.db",
                       help="Output database path (for local target)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Max concurrent sync operations")
    parser.add_argument("--bwlimit", type=int, default=10000,
                       help="Bandwidth limit in KB/s per transfer")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only show what would be synced")
    args = parser.parse_args()

    print("=" * 60)
    print("CLUSTER DATA SYNC")
    print("=" * 60)

    # Load hosts
    hosts = load_hosts()
    print(f"\nLoaded {len(hosts)} hosts")

    # Get game counts from all hosts
    print("\nQuerying hosts for game data...")
    start = time.time()
    results = await asyncio.gather(*[get_host_game_count(h) for h in hosts.values()])
    elapsed = time.time() - start

    # Sort by game count
    results = sorted(results, key=lambda x: -x[1])

    print(f"\nHost inventory ({elapsed:.1f}s):")
    total_games = 0
    hosts_with_data = []
    for name, count, dbs in results:
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name}: {count:,} games in {len(dbs)} databases")
        total_games += count
        if count > 0:
            hosts_with_data.append((name, count, dbs))

    print(f"\nTotal: {total_games:,} games across {len(hosts_with_data)} hosts")

    if args.dry_run:
        print("\n[DRY RUN] Would sync the above data")
        return

    # Create temp directory for synced data
    sync_dir = AI_SERVICE_ROOT / "data" / "games" / "sync_temp"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Sync databases with concurrency limit
    print(f"\nSyncing databases (max {args.max_concurrent} concurrent)...")
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def sync_with_limit(host, db):
        async with semaphore:
            return await sync_database(hosts[host], db, sync_dir, args.bwlimit)

    tasks = []
    for name, count, dbs in hosts_with_data:
        for db in dbs:
            tasks.append(sync_with_limit(name, db))

    if tasks:
        sync_start = time.time()
        results = await asyncio.gather(*tasks)
        sync_elapsed = time.time() - sync_start
        successful = sum(1 for r in results if r)
        print(f"\nSync complete: {successful}/{len(tasks)} databases in {sync_elapsed:.1f}s")

    # Merge databases
    output_path = AI_SERVICE_ROOT / args.output
    merged_count = await merge_databases(sync_dir, output_path)

    print(f"\n{'=' * 60}")
    print(f"SYNC COMPLETE")
    print(f"Output: {output_path}")
    print(f"Games: {merged_count:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
