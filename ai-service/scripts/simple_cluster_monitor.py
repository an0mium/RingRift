#!/usr/bin/env python3
"""
Simple Cluster Monitor - Lightweight, non-blocking cluster health checks.

Uses concurrent SSH checks with strict timeouts to avoid hanging.
Logs status every cycle and takes corrective actions if needed.

Usage:
    python scripts/simple_cluster_monitor.py --duration-hours 8
"""

import argparse
import concurrent.futures
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

LOG_FILE = Path("logs/simple_monitor.log")
CHECK_INTERVAL = 300  # 5 minutes


def log(msg: str):
    """Log to file and stdout with timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def ssh_check(host: str, port: str = None, cmd: str = "echo OK", timeout: int = 10) -> tuple[str, bool, str]:
    """Quick SSH check with strict timeout."""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    if port:
        ssh_cmd.extend(["-p", port])
    ssh_cmd.extend([host, cmd])

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return (host, result.returncode == 0, result.stdout.strip()[:100])
    except subprocess.TimeoutExpired:
        return (host, False, "TIMEOUT")
    except Exception as e:
        return (host, False, str(e)[:50])


def check_lambda_nodes() -> dict:
    """Check Lambda Slurm nodes in parallel."""
    nodes = [
        ("ubuntu@lambda-gh200-e", None),
        ("ubuntu@lambda-gh200-f", None),
        ("ubuntu@lambda-2xh100", None),
    ]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(ssh_check, h, p, "hostname"): h for h, p in nodes}
        for future in concurrent.futures.as_completed(futures, timeout=20):
            try:
                host, ok, out = future.result()
                results[host] = "OK" if ok else f"FAIL: {out}"
            except:
                results[futures[future]] = "TIMEOUT"
    return results


def check_vast_nodes() -> dict:
    """Check Vast.ai nodes via SSH proxy in parallel."""
    # Get instances from vastai CLI
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return {"error": "vastai CLI failed"}

        import json
        instances = json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

    results = {}

    def check_vast_instance(inst):
        id = inst.get("id")
        host = inst.get("ssh_host")
        port = inst.get("ssh_port")
        if not host:
            return (str(id), False, "NO_SSH")
        return ssh_check(f"root@{host}", str(port), "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_vast_instance, inst): inst["id"] for inst in instances}
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                host, ok, out = future.result()
                id = futures[future]
                if ok:
                    gpu_util = out.strip() if out.strip().isdigit() else "?"
                    results[str(id)] = f"OK (GPU: {gpu_util}%)"
                else:
                    results[str(id)] = f"FAIL: {out}"
            except Exception as e:
                results[str(futures[future])] = f"ERROR: {str(e)[:30]}"

    return results


def check_slurm_queue() -> dict:
    """Check Slurm job queue."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "ubuntu@lambda-gh200-f", "squeue -u $(whoami) | wc -l"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            count = int(result.stdout.strip()) - 1  # subtract header
            return {"jobs": count, "status": "OK"}
        return {"status": "FAIL", "error": result.stderr[:50]}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:50]}


def fill_idle_nodes():
    """Submit jobs to idle nodes if needed."""
    log("Checking for idle nodes to fill...")
    try:
        result = subprocess.run(
            ["python", "scripts/cluster_submit.py", "fill-idle", "--board", "square8", "--players", "2", "--games", "2000", "--gpu", "--max-jobs", "5"],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )
        if "Submitted" in result.stdout:
            log(f"Filled idle nodes: {result.stdout.split('Submitted')[-1][:100]}")
        else:
            log(f"No idle nodes or fill failed")
    except Exception as e:
        log(f"Fill-idle error: {e}")


def run_cycle(cycle: int):
    """Run one monitoring cycle."""
    log(f"=== Cycle {cycle} starting ===")

    # Check Lambda nodes
    log("Checking Lambda nodes...")
    lambda_results = check_lambda_nodes()
    healthy = sum(1 for v in lambda_results.values() if v == "OK")
    log(f"Lambda: {healthy}/{len(lambda_results)} healthy")
    for node, status in lambda_results.items():
        if status != "OK":
            log(f"  WARN: {node} - {status}")

    # Check Vast nodes
    log("Checking Vast.ai nodes...")
    vast_results = check_vast_nodes()
    if "error" in vast_results:
        log(f"Vast check error: {vast_results['error']}")
    else:
        healthy = sum(1 for v in vast_results.values() if "OK" in v)
        log(f"Vast.ai: {healthy}/{len(vast_results)} healthy")
        for id, status in vast_results.items():
            if "FAIL" in status or "ERROR" in status:
                log(f"  WARN: Instance {id} - {status}")

    # Check Slurm
    log("Checking Slurm queue...")
    slurm = check_slurm_queue()
    if slurm.get("status") == "OK":
        log(f"Slurm: {slurm['jobs']} jobs in queue")
        if slurm["jobs"] < 5:
            fill_idle_nodes()
    else:
        log(f"Slurm check failed: {slurm.get('error', 'unknown')}")

    log(f"=== Cycle {cycle} complete ===\n")


def main():
    parser = argparse.ArgumentParser(description="Simple Cluster Monitor")
    parser.add_argument("--duration-hours", type=float, default=8, help="Duration to run")
    args = parser.parse_args()

    end_time = datetime.datetime.now() + datetime.timedelta(hours=args.duration_hours)
    log(f"Starting monitor, will run until {end_time.strftime('%Y-%m-%d %H:%M')}")

    cycle = 0
    while datetime.datetime.now() < end_time:
        cycle += 1
        try:
            run_cycle(cycle)
        except Exception as e:
            log(f"Cycle {cycle} error: {e}")

        # Sleep until next cycle
        sleep_time = CHECK_INTERVAL
        log(f"Sleeping {sleep_time}s until next cycle...")
        time.sleep(sleep_time)

    log("Monitor duration reached, exiting.")


if __name__ == "__main__":
    main()
