#!/usr/bin/env python3
"""Serf event handler for RingRift P2P cluster.

This script is called by Serf when events occur in the cluster.
Serf passes event information via environment variables and stdin.

Environment Variables:
    SERF_EVENT: Event type (member-join, member-leave, member-failed, user, query)
    SERF_SELF_NAME: Name of the local node
    SERF_SELF_ROLE: Role of the local node
    SERF_TAG_*: Node tags
    SERF_USER_EVENT: Name of user event (for user events)
    SERF_USER_LTIME: Lamport time of user event
    SERF_QUERY_NAME: Name of query (for queries)
    SERF_QUERY_LTIME: Lamport time of query

stdin:
    For member events: "node1\\taddr1\\trole1\\ntag1=val1,tag2=val2\\n"
    For user events: raw payload
    For queries: raw payload

Usage:
    # In Serf agent config
    serf agent -event-handler=/path/to/serf_event_handler.py

    # Or with filtering
    serf agent -event-handler="member-join,member-leave:/path/to/serf_event_handler.py"
"""

import json
import logging
import os
import sys
import urllib.request
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - serf_handler - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("/tmp/serf_events.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# P2P orchestrator endpoint for forwarding events
# Dec 2025: Standardized to RINGRIFT_P2P_URL with legacy fallback
P2P_ENDPOINT = os.environ.get("RINGRIFT_P2P_URL") or os.environ.get("RINGRIFT_P2P_ENDPOINT", "http://127.0.0.1:8770")


def notify_p2p(event_type: str, payload: dict) -> bool:
    """Notify P2P orchestrator about Serf event.

    Args:
        event_type: Type of event (member-join, member-leave, etc.)
        payload: Event payload dict

    Returns:
        True if notification succeeded
    """
    try:
        url = f"{P2P_ENDPOINT}/serf/event"
        data = json.dumps({
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200

    except Exception as e:
        logger.error(f"Failed to notify P2P: {e}")
        return False


def handle_member_join(members: list[dict]) -> None:
    """Handle member-join events.

    Args:
        members: List of joined member dicts
    """
    for member in members:
        logger.info(f"Member joined: {member.get('name')} @ {member.get('addr')}")

    notify_p2p("member-join", {"members": members})


def handle_member_leave(members: list[dict]) -> None:
    """Handle member-leave events.

    Args:
        members: List of left member dicts
    """
    for member in members:
        logger.info(f"Member left: {member.get('name')}")

    notify_p2p("member-leave", {"members": members})


def handle_member_failed(members: list[dict]) -> None:
    """Handle member-failed events.

    Args:
        members: List of failed member dicts
    """
    for member in members:
        logger.warning(f"Member failed: {member.get('name')} @ {member.get('addr')}")

    notify_p2p("member-failed", {"members": members})


def handle_member_update(members: list[dict]) -> None:
    """Handle member-update events (tag changes).

    Args:
        members: List of updated member dicts
    """
    for member in members:
        logger.info(f"Member updated: {member.get('name')}")

    notify_p2p("member-update", {"members": members})


def handle_member_reap(members: list[dict]) -> None:
    """Handle member-reap events (final cleanup of failed nodes).

    Args:
        members: List of reaped member dicts
    """
    for member in members:
        logger.info(f"Member reaped: {member.get('name')}")

    notify_p2p("member-reap", {"members": members})


def handle_user_event(event_name: str, payload: bytes) -> None:
    """Handle user-defined events.

    Args:
        event_name: Name of the user event
        payload: Raw event payload
    """
    logger.info(f"User event: {event_name}, payload_len={len(payload)}")

    # Try to decode payload as JSON
    try:
        payload_dict = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload_dict = {"raw": payload.decode("utf-8", errors="replace")}

    notify_p2p("user", {
        "name": event_name,
        "payload": payload_dict,
        "ltime": os.environ.get("SERF_USER_LTIME", "0"),
    })

    # Handle specific RingRift events
    if event_name == "training-complete":
        handle_training_complete(payload_dict)
    elif event_name == "model-promoted":
        handle_model_promoted(payload_dict)
    elif event_name == "selfplay-started":
        handle_selfplay_started(payload_dict)
    elif event_name == "node-status":
        handle_node_status(payload_dict)


def handle_query(query_name: str, payload: bytes) -> str:
    """Handle queries (request-response pattern).

    Args:
        query_name: Name of the query
        payload: Raw query payload

    Returns:
        Response string to send back
    """
    logger.info(f"Query: {query_name}")

    try:
        payload_dict = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload_dict = {}

    # Handle specific queries
    if query_name == "node-status":
        return json.dumps(get_node_status())
    elif query_name == "job-count":
        return json.dumps(get_job_count())
    elif query_name == "can-accept-work":
        return json.dumps(can_accept_work(payload_dict))
    else:
        return json.dumps({"error": f"unknown query: {query_name}"})


# ========== RingRift-specific Event Handlers ==========

def handle_training_complete(payload: dict) -> None:
    """Handle training completion event.

    Args:
        payload: Training completion info
    """
    config_key = payload.get("config_key", "unknown")
    model_path = payload.get("model_path", "")
    metrics = payload.get("metrics", {})

    logger.info(f"Training complete for {config_key}: {model_path}")
    logger.info(f"  Metrics: {metrics}")


def handle_model_promoted(payload: dict) -> None:
    """Handle model promotion event.

    Args:
        payload: Promotion info
    """
    config_key = payload.get("config_key", "unknown")
    model_path = payload.get("model_path", "")
    elo_gain = payload.get("elo_gain", 0)

    logger.info(f"Model promoted for {config_key}: {model_path} (+{elo_gain} Elo)")


def handle_selfplay_started(payload: dict) -> None:
    """Handle selfplay started event.

    Args:
        payload: Selfplay job info
    """
    node = payload.get("node", "unknown")
    config_key = payload.get("config_key", "unknown")
    job_count = payload.get("job_count", 1)

    logger.info(f"Selfplay started on {node}: {config_key} x{job_count}")


def handle_node_status(payload: dict) -> None:
    """Handle node status broadcast.

    Args:
        payload: Node status info
    """
    # Just log, P2P will handle via /serf/event endpoint
    node = payload.get("node_id", "unknown")
    logger.debug(f"Node status from {node}")


# ========== Query Response Functions ==========

def get_node_status() -> dict:
    """Get current node status for query response."""
    import subprocess

    status = {
        "node_id": os.environ.get("SERF_SELF_NAME", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Try to get GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                status["gpu_util"] = float(parts[0])
                status["gpu_mem_used"] = float(parts[1])
                status["gpu_mem_total"] = float(parts[2])
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    # Try to get CPU info
    try:
        import psutil
        status["cpu_percent"] = psutil.cpu_percent()
        status["memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass

    return status


def get_job_count() -> dict:
    """Get current job counts for query response."""
    import subprocess

    counts = {
        "selfplay": 0,
        "training": 0,
    }

    try:
        # Count selfplay processes
        result = subprocess.run(
            ["pgrep", "-f", "selfplay"],
            capture_output=True,
            text=True,
        )
        counts["selfplay"] = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

        # Count training processes
        result = subprocess.run(
            ["pgrep", "-f", "train.py"],
            capture_output=True,
            text=True,
        )
        counts["training"] = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return counts


def can_accept_work(params: dict) -> dict:
    """Check if node can accept work for query response.

    Args:
        params: Work parameters (work_type, resource_requirements, etc.)
    """
    work_type = params.get("work_type", "selfplay")

    # Get current load
    job_counts = get_job_count()

    # Simple acceptance logic
    if work_type == "selfplay":
        can_accept = job_counts["selfplay"] < 10
    elif work_type == "training":
        can_accept = job_counts["training"] < 1
    else:
        can_accept = True

    return {
        "can_accept": can_accept,
        "current_jobs": job_counts,
        "work_type": work_type,
    }


# ========== Member Parsing ==========

def parse_member_line(line: str) -> dict:
    """Parse a member line from Serf stdin.

    Format: "name\\taddr\\trole\\ttags"
    Tags format: "key1=val1,key2=val2"

    Args:
        line: Member line from stdin

    Returns:
        Member dict with name, addr, role, tags
    """
    parts = line.strip().split("\t")
    member = {
        "name": parts[0] if len(parts) > 0 else "",
        "addr": parts[1] if len(parts) > 1 else "",
        "role": parts[2] if len(parts) > 2 else "",
        "tags": {},
    }

    if len(parts) > 3:
        # Parse tags
        tag_str = parts[3]
        for tag in tag_str.split(","):
            if "=" in tag:
                k, v = tag.split("=", 1)
                member["tags"][k] = v

    return member


def main() -> int:
    """Main entry point for Serf event handler.

    Returns:
        Exit code (0 for success)
    """
    event_type = os.environ.get("SERF_EVENT", "")

    if not event_type:
        logger.error("SERF_EVENT not set - not running as Serf handler?")
        return 1

    logger.info(f"Handling Serf event: {event_type}")

    try:
        if event_type == "member-join":
            members = [parse_member_line(line) for line in sys.stdin if line.strip()]
            handle_member_join(members)

        elif event_type == "member-leave":
            members = [parse_member_line(line) for line in sys.stdin if line.strip()]
            handle_member_leave(members)

        elif event_type == "member-failed":
            members = [parse_member_line(line) for line in sys.stdin if line.strip()]
            handle_member_failed(members)

        elif event_type == "member-update":
            members = [parse_member_line(line) for line in sys.stdin if line.strip()]
            handle_member_update(members)

        elif event_type == "member-reap":
            members = [parse_member_line(line) for line in sys.stdin if line.strip()]
            handle_member_reap(members)

        elif event_type == "user":
            event_name = os.environ.get("SERF_USER_EVENT", "")
            payload = sys.stdin.read().encode("utf-8")
            handle_user_event(event_name, payload)

        elif event_type == "query":
            query_name = os.environ.get("SERF_QUERY_NAME", "")
            payload = sys.stdin.read().encode("utf-8")
            response = handle_query(query_name, payload)
            # Write response to stdout for Serf to return
            print(response)

        else:
            logger.warning(f"Unknown event type: {event_type}")

    except Exception as e:
        logger.exception(f"Error handling event {event_type}: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
