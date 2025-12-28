#!/usr/bin/env python3
"""Event Wiring Audit Script - CI Validation for Orphan Events.

This script scans the coordination codebase to detect orphan events (emitters
without subscribers) that would silently break the training pipeline.

Exit Codes:
    0 - All critical events have subscribers
    1 - Critical events are orphaned (blocking for CI)
    2 - Script error

Usage:
    # Quick check (exits 0/1)
    python scripts/audit_event_wiring.py

    # Verbose output with details
    python scripts/audit_event_wiring.py --verbose

    # JSON output for CI integration
    python scripts/audit_event_wiring.py --json

    # Fail only on critical events (default)
    python scripts/audit_event_wiring.py --strict

CI Integration:
    # In GitHub Actions or similar:
    - name: Audit Event Wiring
      run: python scripts/audit_event_wiring.py

December 2025 - Phase 5D: CI validation for orphan events.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Path to source directories
AI_SERVICE_ROOT = Path(__file__).parent.parent
COORDINATION_DIR = AI_SERVICE_ROOT / "app" / "coordination"
P2P_DIR = AI_SERVICE_ROOT / "scripts" / "p2p"
DISTRIBUTED_DIR = AI_SERVICE_ROOT / "app" / "distributed"

# Critical events that MUST have subscribers (pipeline would break without them)
# From docs/runbooks/EVENT_WIRING_VERIFICATION.md
CRITICAL_EVENTS = {
    "training_started": ["SyncRouter", "IdleShutdown", "DataPipeline"],
    "training_completed": ["FeedbackLoop", "DataPipeline", "ModelDistribution"],
    "evaluation_completed": ["FeedbackLoop", "CurriculumIntegration"],
    "model_promoted": ["ModelDistribution", "FeedbackLoop"],
    "data_sync_completed": ["DataPipelineOrchestrator"],
    "new_games_available": ["SelfplayScheduler"],
    "regression_detected": ["ModelLifecycleCoordinator", "DataPipeline"],
    "orphan_games_detected": ["DataPipelineOrchestrator"],
    "backpressure_activated": ["SyncRouter"],
}

# Events intentionally orphaned (no subscribers expected)
INTENTIONALLY_ORPHANED = {
    # Metrics/logging events observed externally
    "metrics_updated",
    "daemon_heartbeat",
    "coordinator_heartbeat",
    # Status broadcast events
    "idle_state_broadcast",
    # Events consumed by external systems
    "webhook_sent",
    # Test/debug events
    "test_event",
    "debug_event",
}


def scan_file_for_patterns(
    file_path: Path, patterns: List[str]
) -> List[Tuple[str, int, str]]:
    """Scan a file for patterns and return matches with line numbers."""
    matches = []
    try:
        content = file_path.read_text()
        for line_num, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    matches.append((match.group(1), line_num, str(file_path.name)))
    except Exception:
        pass
    return matches


def find_emitters() -> Dict[str, List[Tuple[str, int]]]:
    """Find all event emitters in coordination code."""
    emitters: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    emit_patterns = [
        r'emit\(["\']([a-z_]+)["\']',
        r'emit_event\(["\']([a-z_]+)["\']',
        r'\.emit\(["\']([a-z_]+)["\']',
        r'_emit_([a-z_]+)_event',
        r'publish\(["\']([a-z_]+)["\']',
        r'emit_([a-z_]+)\(',
        r'DataEventType\.([A-Z_]+)\.value',
    ]

    for search_dir in [COORDINATION_DIR, P2P_DIR, DISTRIBUTED_DIR]:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            for event_name, line_num, filename in scan_file_for_patterns(
                py_file, emit_patterns
            ):
                normalized = event_name.lower().replace("-", "_")
                emitters[normalized].append((filename, line_num))

    return emitters


def find_subscribers() -> Dict[str, List[Tuple[str, int]]]:
    """Find all event subscribers in coordination code."""
    subscribers: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    subscribe_patterns = [
        r'subscribe\(["\']([a-z_]+)["\']',
        r'subscribe_to_event\(["\']([a-z_]+)["\']',
        r'\.subscribe\(["\']([a-z_]+)["\']',
        r'on_event\(["\']([a-z_]+)["\']',
        r'["\']([a-z_]+)["\']\s*:\s*self\._on_',
        r'DataEventType\.([A-Z_]+)\.value.*subscribe',
        r'DataEventType\.([A-Z_]+)[,\)]',
        r'_on_([a-z_]+)\s*\(',
        # Additional patterns for event handler dictionaries
        r'"([a-z_]+)":\s*self\.',
        r"'([a-z_]+)':\s*self\.",
    ]

    for search_dir in [COORDINATION_DIR, P2P_DIR, DISTRIBUTED_DIR]:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            for event_name, line_num, filename in scan_file_for_patterns(
                py_file, subscribe_patterns
            ):
                normalized = event_name.lower().replace("-", "_")
                subscribers[normalized].append((filename, line_num))

    return subscribers


def find_orphan_events(
    emitters: Dict[str, List[Tuple[str, int]]],
    subscribers: Dict[str, List[Tuple[str, int]]],
) -> Set[str]:
    """Find events that are emitted but have no subscribers."""
    emitted_events = set(emitters.keys())
    subscribed_events = set(subscribers.keys())
    orphans = emitted_events - subscribed_events - INTENTIONALLY_ORPHANED
    # Filter out test patterns
    orphans = {o for o in orphans if not o.startswith("test_") and not o.startswith("mock_")}
    return orphans


def find_critical_orphans(orphans: Set[str]) -> Set[str]:
    """Find critical events that are orphaned."""
    return set(CRITICAL_EVENTS.keys()) & orphans


def format_output(
    emitters: Dict[str, List[Tuple[str, int]]],
    subscribers: Dict[str, List[Tuple[str, int]]],
    orphans: Set[str],
    critical_orphans: Set[str],
    verbose: bool = False,
    as_json: bool = False,
) -> str:
    """Format the audit results."""
    if as_json:
        return json.dumps({
            "total_events_emitted": len(emitters),
            "total_events_subscribed": len(subscribers),
            "orphan_events": sorted(orphans),
            "critical_orphans": sorted(critical_orphans),
            "critical_events_defined": list(CRITICAL_EVENTS.keys()),
            "intentionally_orphaned": list(INTENTIONALLY_ORPHANED),
            "status": "FAIL" if critical_orphans else "PASS",
        }, indent=2)

    lines = []
    lines.append("=" * 60)
    lines.append("Event Wiring Audit Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Events emitted:      {len(emitters)}")
    lines.append(f"Events subscribed:   {len(subscribers)}")
    lines.append(f"Orphan events:       {len(orphans)}")
    lines.append(f"Critical orphans:    {len(critical_orphans)}")
    lines.append("")

    if critical_orphans:
        lines.append("CRITICAL: The following critical events have no subscribers:")
        for event in sorted(critical_orphans):
            lines.append(f"  - {event}")
            if verbose and event in emitters:
                for filename, line_num in emitters[event][:3]:
                    lines.append(f"      Emitted at: {filename}:{line_num}")
        lines.append("")
        lines.append("This will break the training pipeline. Add subscribers immediately.")
        lines.append("")

    if orphans and verbose:
        non_critical = orphans - critical_orphans
        if non_critical:
            lines.append("Non-critical orphan events (informational):")
            for event in sorted(non_critical)[:15]:
                lines.append(f"  - {event}")
            if len(non_critical) > 15:
                lines.append(f"  ... and {len(non_critical) - 15} more")
            lines.append("")

    if critical_orphans:
        lines.append("STATUS: FAIL - Critical events are orphaned")
    else:
        lines.append("STATUS: PASS - All critical events have subscribers")

    return "\n".join(lines)


def main():
    """Run the event wiring audit."""
    parser = argparse.ArgumentParser(
        description="Audit event wiring to detect orphan events"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON for CI integration"
    )
    parser.add_argument(
        "--strict", action="store_true", default=True,
        help="Fail only on critical events (default: True)"
    )
    parser.add_argument(
        "--fail-on-any", action="store_true",
        help="Fail on any orphan event (not just critical)"
    )
    args = parser.parse_args()

    try:
        emitters = find_emitters()
        subscribers = find_subscribers()
        orphans = find_orphan_events(emitters, subscribers)
        critical_orphans = find_critical_orphans(orphans)

        output = format_output(
            emitters, subscribers, orphans, critical_orphans,
            verbose=args.verbose, as_json=args.json
        )
        print(output)

        # Exit code logic
        if critical_orphans:
            sys.exit(1)
        elif args.fail_on_any and orphans:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        if args.json:
            print(json.dumps({"status": "ERROR", "error": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
