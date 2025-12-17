#!/usr/bin/env python3
"""
Vast.ai P2P Sync - Synchronize Vast instance state with P2P network.

This script:
1. Gets active Vast instances from vastai CLI
2. Compares with P2P network retired nodes
3. Unretires nodes that match active Vast instances
4. Starts P2P orchestrator on nodes missing it
5. Updates distributed_hosts.yaml with current IPs
6. Can provision new instances based on demand
7. Can be run via cron for continuous sync

Usage:
    python scripts/vast_p2p_sync.py --check      # Check status only
    python scripts/vast_p2p_sync.py --sync       # Sync and unretire active instances
    python scripts/vast_p2p_sync.py --start-p2p  # Start P2P on instances missing it
    python scripts/vast_p2p_sync.py --full       # Full sync (check + sync + start)
    python scripts/vast_p2p_sync.py --update-config  # Update distributed_hosts.yaml
    python scripts/vast_p2p_sync.py --provision N    # Provision N new instances
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "vast_p2p_sync.log"
CONFIG_FILE = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VastP2PSync] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _load_p2p_leader_from_config() -> str:
    """Load P2P leader endpoint from config or environment."""
    # Check environment first
    if os.environ.get("P2P_LEADER"):
        return os.environ["P2P_LEADER"]

    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        logger.warning("[P2PSync] Warning: No config found at %s", config_path)
        return ""

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Get coordinator from elo_sync config
        coordinator = config.get("elo_sync", {}).get("coordinator", "mac-studio")
        hosts = config.get("hosts", {})

        if coordinator in hosts:
            coord_info = hosts[coordinator]
            tailscale_ip = coord_info.get("tailscale_ip") or coord_info.get("ssh_host")
            if tailscale_ip and tailscale_ip.startswith("100."):
                return f"http://{tailscale_ip}:8770"

        # Fallback: find first ready p2p_voter node
        for host_id, host_info in hosts.items():
            if host_info.get("status") == "ready" and host_info.get("p2p_voter"):
                tailscale_ip = host_info.get("tailscale_ip")
                if tailscale_ip and tailscale_ip.startswith("100."):
                    return f"http://{tailscale_ip}:8770"

        return ""
    except Exception as e:
        logger.warning("[P2PSync] Error loading config: %s", e)
        return ""

# GPU role mapping for config
GPU_ROLES = {
    "RTX 3070": "gpu_selfplay",
    "RTX 3060": "gpu_selfplay",
    "RTX 3060 Ti": "cpu_selfplay",
    "RTX 2060S": "gpu_selfplay",
    "RTX 2060 SUPER": "gpu_selfplay",
    "RTX 2080 Ti": "gpu_selfplay",
    "RTX 4060 Ti": "gpu_selfplay",
    "RTX 4080S": "nn_training_primary",
    "RTX 4080 SUPER": "nn_training_primary",
    "RTX 5070": "nn_training_primary",
    "RTX 5080": "nn_training_primary",
    "RTX 5090": "nn_training_primary",
    "A10": "nn_training_primary",
    "A40": "nn_training_primary",
    "A100": "nn_training_primary",
    "H100": "nn_training_primary",
}

# Preferred GPU types for auto-provisioning (ordered by preference)
PREFERRED_GPUS = [
    {"name": "RTX 3070", "max_price": 0.08, "role": "gpu_selfplay"},
    {"name": "RTX 3060", "max_price": 0.06, "role": "gpu_selfplay"},
    {"name": "RTX 4060 Ti", "max_price": 0.12, "role": "gpu_selfplay"},
    {"name": "RTX 2080 Ti", "max_price": 0.10, "role": "gpu_selfplay"},
]

# Vast instance ID to Tailscale IP mapping (discovered dynamically)
VAST_TAILSCALE_IPS: Dict[int, str] = {}


@dataclass
class VastInstance:
    """Vast.ai instance info."""
    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    vcpus: float
    ram_gb: float
    ssh_host: str
    ssh_port: int
    status: str
    hourly_cost: float
    uptime_mins: float


@dataclass
class P2PNode:
    """P2P network node info."""
    node_id: str
    host: str
    retired: bool
    selfplay_jobs: int
    healthy: bool
    gpu_name: str


def get_vast_instances() -> List[VastInstance]:
    """Get active Vast instances from vastai CLI."""
    try:
        result = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logger.error(f"vastai CLI error: {result.stderr}")
            return []

        instances = json.loads(result.stdout)
        return [
            VastInstance(
                id=inst.get('id', 0),
                machine_id=inst.get('machine_id', 0),
                gpu_name=inst.get('gpu_name', 'Unknown'),
                num_gpus=inst.get('num_gpus', 0) or 1,
                vcpus=inst.get('cpu_cores_effective', 0) or 0,
                ram_gb=inst.get('cpu_ram', 0) / 1024 if inst.get('cpu_ram') else 0,
                ssh_host=inst.get('ssh_host', ''),
                ssh_port=inst.get('ssh_port', 22),
                status=inst.get('actual_status', 'unknown'),
                hourly_cost=inst.get('dph_total', 0) or 0,
                uptime_mins=inst.get('duration', 0) or 0,
            )
            for inst in instances
            if inst.get('actual_status') == 'running'
        ]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vastai output: {e}")
        return []
    except subprocess.TimeoutExpired:
        logger.error("vastai CLI timeout")
        return []
    except FileNotFoundError:
        logger.error("vastai CLI not found - install with: pip install vastai")
        return []


def get_p2p_nodes() -> List[P2PNode]:
    """Get nodes from P2P network."""
    p2p_leader = _load_p2p_leader_from_config()
    if not p2p_leader:
        logger.error("No P2P leader configured, cannot get P2P nodes")
        return []

    try:
        import urllib.request
        with urllib.request.urlopen(f"{p2p_leader}/status", timeout=10) as response:
            data = json.loads(response.read().decode())

        nodes = []
        peers = data.get('peers', {})
        for node_id, info in peers.items():
            nodes.append(P2PNode(
                node_id=node_id,
                host=info.get('reported_host') or info.get('host', ''),
                retired=info.get('retired', False),
                selfplay_jobs=info.get('selfplay_jobs', 0),
                healthy=not info.get('retired', False),
                gpu_name=info.get('gpu_name', ''),
            ))

        # Add self
        self_info = data.get('self', {})
        if self_info:
            nodes.append(P2PNode(
                node_id=self_info.get('node_id', ''),
                host=self_info.get('host', ''),
                retired=self_info.get('retired', False),
                selfplay_jobs=self_info.get('selfplay_jobs', 0),
                healthy=True,
                gpu_name=self_info.get('gpu_name', ''),
            ))

        return nodes
    except Exception as e:
        logger.error(f"Failed to get P2P nodes: {e}")
        return []


def get_vast_tailscale_ip(instance: VastInstance) -> Optional[str]:
    """Get Tailscale IP for a Vast instance via SSH."""
    if instance.id in VAST_TAILSCALE_IPS:
        return VAST_TAILSCALE_IPS[instance.id]

    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'tailscale ip -4 2>/dev/null || ip route get 1 | grep -oP "src \\K\\S+"'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            ip = result.stdout.strip().split('\n')[0]
            if ip.startswith('100.'):
                VAST_TAILSCALE_IPS[instance.id] = ip
                return ip
    except Exception as e:
        logger.debug(f"Failed to get Tailscale IP for instance {instance.id}: {e}")

    return None


def match_vast_to_p2p(vast_instances: List[VastInstance], p2p_nodes: List[P2PNode]) -> Dict[int, P2PNode]:
    """Match Vast instances to P2P nodes by various criteria."""
    matches: Dict[int, P2PNode] = {}

    for inst in vast_instances:
        # Try matching by Tailscale IP
        ts_ip = get_vast_tailscale_ip(inst)
        if ts_ip:
            for node in p2p_nodes:
                if node.host == ts_ip:
                    matches[inst.id] = node
                    break
            if inst.id in matches:
                continue

        # Try matching by node_id patterns
        for node in p2p_nodes:
            node_lower = node.node_id.lower()
            # Match patterns like vast-28844401, vast-4e19d4df2c83, etc.
            if f"vast-{inst.id}" in node_lower or f"vast{inst.id}" in node_lower:
                matches[inst.id] = node
                break
            # Match by machine ID
            if f"vast-{inst.machine_id}" in node_lower:
                matches[inst.id] = node
                break

    return matches


def unretire_node_via_api(node_id: str) -> bool:
    """Unretire a node by calling P2P API."""
    p2p_leader = _load_p2p_leader_from_config()
    if not p2p_leader:
        logger.error("No P2P leader configured, cannot unretire node")
        return False

    try:
        import urllib.request
        import urllib.parse

        url = f"{p2p_leader}/admin/unretire?node_id={urllib.parse.quote(node_id)}"
        req = urllib.request.Request(url, method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Failed to unretire {node_id} via API: {e}")
        return False


def start_p2p_on_instance(instance: VastInstance) -> bool:
    """Start P2P orchestrator on a Vast instance."""
    # Determine RingRift path
    ringrift_paths = ['/workspace/ringrift', '/root/ringrift']

    try:
        # Check which path exists
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null'],
            capture_output=True, text=True, timeout=15
        )
        ringrift_path = result.stdout.strip() if result.returncode == 0 else '/root/ringrift'

        # Generate node ID
        node_id = f"vast-{instance.id}"

        # Start P2P
        cmd = f"""
cd {ringrift_path}/ai-service
mkdir -p logs
pkill -f p2p_orchestrator 2>/dev/null || true
nohup /opt/conda/bin/python3 scripts/p2p_orchestrator.py \\
    --node-id {node_id} \\
    --port 8770 \\
    --peers https://p2p.ringrift.ai \\
    --ringrift-path {ringrift_path} \\
    > logs/p2p.log 2>&1 &
sleep 2
curl -s http://localhost:8770/health | head -c 100
"""
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}', cmd],
            capture_output=True, text=True, timeout=30
        )

        if 'healthy' in result.stdout or 'node_id' in result.stdout:
            logger.info(f"Started P2P on instance {instance.id} ({instance.gpu_name})")
            return True
        else:
            logger.warning(f"P2P start on {instance.id} unclear: {result.stdout[:200]}")
            return False

    except Exception as e:
        logger.error(f"Failed to start P2P on instance {instance.id}: {e}")
        return False


def check_p2p_running(instance: VastInstance) -> Tuple[bool, int]:
    """Check if P2P is running on instance. Returns (is_running, selfplay_jobs)."""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'curl -s http://localhost:8770/health 2>/dev/null'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return True, data.get('selfplay_jobs', 0)
    except Exception:
        pass
    return False, 0


def update_distributed_hosts_yaml(instances: List[VastInstance]) -> int:
    """Update distributed_hosts.yaml with current Vast instance info."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, skipping config update")
        return 0

    if not CONFIG_FILE.exists():
        logger.warning(f"Config file not found: {CONFIG_FILE}")
        return 0

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    hosts = config.get("hosts", {})
    updated = 0

    for inst in instances:
        if inst.status != "running":
            continue

        # Try to get Tailscale IP
        ts_ip = get_vast_tailscale_ip(inst)

        node_id = f"vast-{inst.id}"
        gpu_desc = f"{inst.num_gpus}x {inst.gpu_name}" if inst.num_gpus > 1 else inst.gpu_name
        role = GPU_ROLES.get(inst.gpu_name, "gpu_selfplay")

        # Check if update needed
        existing = hosts.get(node_id, {})
        needs_update = (
            existing.get("ssh_host") != inst.ssh_host
            or existing.get("ssh_port") != inst.ssh_port
            or (ts_ip and existing.get("tailscale_ip") != ts_ip)
        )

        if needs_update or node_id not in hosts:
            hosts[node_id] = {
                "ssh_host": inst.ssh_host,
                "ssh_port": inst.ssh_port,
                "ssh_user": "root",
                "ssh_key": "~/.ssh/id_cluster",
                "ringrift_path": "~/ringrift/ai-service",
                "venv_activate": "source ~/ringrift/ai-service/venv/bin/activate",
                "memory_gb": int(inst.ram_gb),
                "cpus": int(inst.vcpus),
                "gpu": gpu_desc,
                "role": role,
                "status": "ready",
                "vast_instance_id": str(inst.id),
            }
            if ts_ip:
                hosts[node_id]["tailscale_ip"] = ts_ip
            updated += 1
            logger.info(f"Updated config for {node_id}: {inst.ssh_host}:{inst.ssh_port}")

    if updated:
        config["hosts"] = hosts
        # Backup
        backup_path = CONFIG_FILE.with_suffix(".yaml.bak")
        with open(backup_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        # Write
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated {updated} hosts in {CONFIG_FILE}")

    return updated


def search_gpu_offers(
    gpu_name: str = "RTX 3070",
    max_price: float = 0.10,
    min_reliability: float = 0.95,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search for available GPU offers using vastai CLI."""
    try:
        # Build query - vastai uses a query language
        query = f"gpu_name={gpu_name} reliability>{min_reliability} dph<{max_price}"

        result = subprocess.run(
            ['vastai', 'search', 'offers', query, '-o', 'dph', '--raw'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"vastai search failed: {result.stderr}")
            return []

        offers = json.loads(result.stdout)
        return offers[:limit]
    except Exception as e:
        logger.error(f"Failed to search offers: {e}")
        return []


def create_vast_instance(
    offer_id: int,
    disk_gb: int = 50,
    use_bid: bool = False,
    bid_price: Optional[float] = None,
) -> Optional[str]:
    """Create a new Vast instance from an offer."""
    try:
        args = [
            'vastai', 'create', 'instance', str(offer_id),
            '--image', 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
            '--disk', str(disk_gb),
            '--ssh',
            '--onstart-cmd', 'apt-get update && apt-get install -y git curl',
        ]

        if use_bid and bid_price:
            args.extend(['--price', str(bid_price)])

        result = subprocess.run(args, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            output = result.stdout + result.stderr
            # Parse instance ID from output (format: "new instance created: 12345678")
            for word in output.split():
                if word.isdigit() and len(word) > 6:
                    logger.info(f"Created instance {word}")
                    return word
            logger.info(f"Instance creation output: {output}")
            return output
        else:
            logger.error(f"Failed to create instance: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Failed to create instance: {e}")
        return None


def provision_instances(count: int = 1, max_total_hourly: float = 0.50) -> int:
    """Provision new Vast instances based on preferred GPU list."""
    logger.info(f"Provisioning up to {count} new instances (max ${max_total_hourly}/hr total)...")

    created = 0
    total_cost = 0.0

    for gpu_pref in PREFERRED_GPUS:
        if created >= count:
            break

        offers = search_gpu_offers(
            gpu_name=gpu_pref["name"],
            max_price=gpu_pref["max_price"],
            limit=count - created,
        )

        for offer in offers:
            if created >= count:
                break

            offer_id = offer.get("id")
            price = offer.get("dph_total", 0)

            if total_cost + price > max_total_hourly:
                logger.info(f"Skipping offer {offer_id} - would exceed hourly budget")
                continue

            logger.info(f"Creating instance from offer {offer_id} ({gpu_pref['name']}, ${price:.3f}/hr)...")
            instance_id = create_vast_instance(offer_id, disk_gb=50)

            if instance_id:
                created += 1
                total_cost += price

    logger.info(f"Provisioned {created} instances (${total_cost:.2f}/hr)")
    return created


def sync_code_to_instance(instance: VastInstance) -> bool:
    """Sync git code to an instance."""
    if instance.status != "running" or not instance.ssh_host:
        return False

    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'cd /root/ringrift && git fetch origin && git reset --hard origin/main 2>&1 | tail -1'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.debug(f"Synced code on instance {instance.id}: {result.stdout.strip()}")
            return True
    except Exception as e:
        logger.debug(f"Failed to sync code on {instance.id}: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Sync Vast instances with P2P network")
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--sync', action='store_true', help='Sync and unretire active instances')
    parser.add_argument('--start-p2p', action='store_true', help='Start P2P on instances missing it')
    parser.add_argument('--full', action='store_true', help='Full sync (check + sync + start + config)')
    parser.add_argument('--update-config', action='store_true', help='Update distributed_hosts.yaml')
    parser.add_argument('--provision', type=int, metavar='N', help='Provision N new instances')
    parser.add_argument('--max-hourly', type=float, default=0.50, help='Max hourly budget for provisioning')
    parser.add_argument('--sync-code', action='store_true', help='Sync git code to all instances')
    args = parser.parse_args()

    if not any([args.check, args.sync, args.start_p2p, args.full, args.update_config, args.provision, args.sync_code]):
        args.check = True  # Default to check

    # Get Vast instances
    logger.info("Getting Vast instances...")
    vast_instances = get_vast_instances()
    logger.info(f"Found {len(vast_instances)} active Vast instances")

    if not vast_instances:
        logger.warning("No active Vast instances found")
        return

    # Get P2P nodes
    logger.info("Getting P2P network nodes...")
    p2p_nodes = get_p2p_nodes()
    logger.info(f"Found {len(p2p_nodes)} P2P nodes")

    # Match Vast to P2P
    matches = match_vast_to_p2p(vast_instances, p2p_nodes)
    logger.info(f"Matched {len(matches)} Vast instances to P2P nodes")

    # Status report
    print("\n" + "=" * 80)
    print("VAST INSTANCE STATUS")
    print("=" * 80)
    print(f"{'ID':<10} {'GPU':<20} {'vCPUs':<8} {'P2P Node':<25} {'Status':<15}")
    print("-" * 80)

    for inst in vast_instances:
        p2p_node = matches.get(inst.id)
        if p2p_node:
            status = "RETIRED" if p2p_node.retired else f"OK ({p2p_node.selfplay_jobs} jobs)"
            node_str = p2p_node.node_id[:24]
        else:
            # Check if P2P is running directly
            running, jobs = check_p2p_running(inst)
            if running:
                status = f"P2P OK ({jobs} jobs)"
                node_str = "(not in network yet)"
            else:
                status = "NO P2P"
                node_str = "-"

        gpu_str = f"{inst.num_gpus}x{inst.gpu_name}"[:19]
        print(f"{inst.id:<10} {gpu_str:<20} {inst.vcpus:<8.0f} {node_str:<25} {status:<15}")

    print("=" * 80 + "\n")

    # Sync actions
    if args.sync or args.full:
        logger.info("Syncing retired nodes...")
        for inst_id, node in matches.items():
            if node.retired:
                logger.info(f"Unretiring {node.node_id} (Vast instance {inst_id})")
                if unretire_node_via_api(node.node_id):
                    logger.info(f"  -> Unretired successfully")
                else:
                    logger.warning(f"  -> Failed to unretire via API")

    # Update config
    if args.update_config or args.full:
        logger.info("Updating distributed_hosts.yaml...")
        updated = update_distributed_hosts_yaml(vast_instances)
        logger.info(f"Updated {updated} entries in config")

    # Start P2P actions
    if args.start_p2p or args.full:
        logger.info("Starting P2P on instances without it...")
        for inst in vast_instances:
            running, _ = check_p2p_running(inst)
            if not running:
                logger.info(f"Starting P2P on instance {inst.id} ({inst.gpu_name})")
                start_p2p_on_instance(inst)

    # Sync code
    if args.sync_code:
        logger.info("Syncing code to all instances...")
        synced = 0
        for inst in vast_instances:
            if sync_code_to_instance(inst):
                synced += 1
        logger.info(f"Synced code to {synced}/{len(vast_instances)} instances")

    # Provision new instances
    if args.provision:
        provision_instances(count=args.provision, max_total_hourly=args.max_hourly)


if __name__ == "__main__":
    main()
