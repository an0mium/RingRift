#!/usr/bin/env python
from __future__ import annotations

"""
Canonical guardrail script for rules/engine/training changes.

This script is designed to be a single, CI-friendly entrypoint that runs:

  1. Canonical phase-history validator on all canonical_*.db.
  2. TS↔Python parity checker on a small sample of games per canonical DB.
  3. Invariant soak across Python GameEngine and (optionally) TS orchestrator.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/run_canonical_guards.py

  # Include TS orchestrator invariant soak as well:
  PYTHONPATH=. python scripts/run_canonical_guards.py --with-ts

Notes:
  - For square19/hex DBs, parity samples default to
    RINGRIFT_USE_MAKE_UNMAKE=true and RINGRIFT_USE_FAST_TERRITORY=false
    unless those environment variables are already set.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.db import GameReplayDB
from app.rules.history_validation import (
    validate_canonical_config_for_game,
    validate_canonical_history_for_game,
)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(f"[canonical-guards] {msg}")


def _canonical_db_paths() -> list[Path]:
    games_dir = AI_SERVICE_ROOT / "data" / "games"
    if not games_dir.exists():
        return []
    return sorted(p for p in games_dir.glob("canonical_*.db") if p.is_file())


def _infer_board_type(db_path: Path) -> str | None:
    try:
        with GameReplayDB(str(db_path))._get_conn() as conn:  # type: ignore[attr-defined]
            row = conn.execute("SELECT board_type FROM games LIMIT 1").fetchone()
    except (OSError, RuntimeError):
        return None

    if not row:
        return None
    board_type = row["board_type"] if isinstance(row, dict) else row[0]
    return str(board_type).strip().lower()


def _canonical_gate_env(board_type: str | None) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    if board_type in {"square19", "hex", "hexagonal"}:
        if "RINGRIFT_USE_FAST_TERRITORY" not in os.environ:
            env_overrides["RINGRIFT_USE_FAST_TERRITORY"] = "false"
        if "RINGRIFT_USE_MAKE_UNMAKE" not in os.environ:
            env_overrides["RINGRIFT_USE_MAKE_UNMAKE"] = "true"
    return env_overrides


def run_history_validator() -> bool:
    """Run validate_canonical_history_for_game over all games in all canonical DBs."""
    db_paths = _canonical_db_paths()
    if not db_paths:
        _log("No canonical_*.db files found; skipping history validator.")
        return True

    all_ok = True
    for db_path in db_paths:
        _log(f"History validator: {db_path}")
        db = GameReplayDB(str(db_path))
        with db._get_conn() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT game_id FROM games").fetchall()
            game_ids = [row["game_id"] for row in rows]

        for gid in game_ids:
            cfg_report = validate_canonical_config_for_game(db, gid)
            if not cfg_report.is_canonical:
                _log(f"  NON-CANONICAL config for game {gid}:")
                for issue in cfg_report.issues[:5]:
                    _log(
                        "    "
                        f"reason={issue.reason} observed={issue.observed} expected={issue.expected}"
                    )
                all_ok = False
                break

            report = validate_canonical_history_for_game(db, gid)
            if not report.is_canonical:
                _log(f"  NON-CANONICAL game {gid}:")
                for issue in report.issues[:5]:
                    _log(
                        f"    move={issue.move_number} phase={issue.phase} "
                        f"type={issue.move_type} reason={issue.reason}"
                    )
                all_ok = False
                break
        if not all_ok:
            break

    return all_ok


def run_parity_sample(max_games_per_db: int) -> bool:
    """Run the parity checker on a small sample of games per canonical DB."""
    db_paths = _canonical_db_paths()
    if not db_paths:
        _log("No canonical_*.db files found; skipping parity sample.")
        return True

    all_ok = True
    for db_path in db_paths:
        _log(f"Parity sample: {db_path}")
        db = GameReplayDB(str(db_path))
        with db._get_conn() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT game_id FROM games").fetchall()
            game_ids = [row["game_id"] for row in rows]

        if not game_ids:
            _log("  Skipping (no games).")
            continue

        random.shuffle(game_ids)
        sample_ids = game_ids[: max_games_per_db or len(game_ids)]
        board_type = _infer_board_type(db_path)
        env_overrides = _canonical_gate_env(board_type)
        if env_overrides:
            _log(f"  Env overrides: {env_overrides}")

        summary = _run_parity_sample(
            db_path=db_path,
            game_ids=sample_ids,
            env_overrides=env_overrides,
        )

        sem = int(summary.get("games_with_semantic_divergence", 0) or 0)
        struct = int(summary.get("games_with_structural_issues", 0) or 0)

        if sem != 0 or struct != 0:
            _log(f"  Parity issues: semantic_divergence={sem}, " f"structural_issues={struct}")
            all_ok = False
            break
        _log(f"  OK (checked {summary.get('total_games_checked', 0)} games; " "no semantic or structural issues).")

    return all_ok


def _run_parity_sample(
    *,
    db_path: Path,
    game_ids: list[str],
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        game_ids_path = tmp_dir / "game_ids.txt"
        game_ids_path.write_text("\n".join(game_ids), encoding="utf-8")
        summary_path = tmp_dir / "parity_summary.json"

        cmd = [
            sys.executable,
            "scripts/check_ts_python_replay_parity.py",
            "--db",
            str(db_path),
            "--mode",
            "canonical",
            "--view",
            "post_move",
            "--include-game-ids-file",
            str(game_ids_path),
            "--summary-json",
            str(summary_path),
        ]

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(AI_SERVICE_ROOT))
        env.update(env_overrides or {})

        proc = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            env=env,
            text=True,
            capture_output=True,
        )

        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                return {
                    "error": "failed_to_load_parity_summary",
                    "summary_path": str(summary_path),
                    "message": str(exc),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

        return {
            "error": "parity_summary_missing",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


def run_invariant_soak(with_ts: bool) -> bool:
    """Delegate to scripts/run_invariant_soak.py."""
    cmd = [
        sys.executable,
        "scripts/run_invariant_soak.py",
        "--max-games-per-db",
        "3",
    ]
    if with_ts:
        cmd.append("--with-ts")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(AI_SERVICE_ROOT))
    proc = subprocess.run(
        cmd,
        cwd=str(AI_SERVICE_ROOT),
        env=env,
        text=True,
    )
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run canonical guardrails: history validator, parity sample, and invariant soak."
    )
    parser.add_argument(
        "--parity-max-games-per-db",
        type=int,
        default=5,
        help="Maximum number of games per canonical DB to parity-check (default: 5).",
    )
    parser.add_argument(
        "--with-ts",
        action="store_true",
        help="Also run TS orchestrator invariant soak via run_invariant_soak.py --with-ts.",
    )
    args = parser.parse_args(argv)

    ok_hist = run_history_validator()
    ok_parity = run_parity_sample(max_games_per_db=args.parity_max_games_per_db)
    ok_soak = run_invariant_soak(with_ts=args.with_ts)

    if not ok_hist:
        _log("History validator FAILED.")
    if not ok_parity:
        _log("Parity sample FAILED.")
    if not ok_soak:
        _log("Invariant soak FAILED.")

    return 0 if (ok_hist and ok_parity and ok_soak) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
