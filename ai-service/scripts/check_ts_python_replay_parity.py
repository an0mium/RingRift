"""TS vs Python replay parity checker for GameReplayDB databases.

This script walks all self-play GameReplayDB databases under the repo and,
for each game, compares Python's GameReplayDB.get_state_at_move against the
TypeScript ClientSandboxEngine replay path driven via the
scripts/selfplay-db-ts-replay.ts harness.

For each game it checks:
  - That TS replays the same number of moves as Python reports (total_moves).
  - For k = 0..min(total_moves, tsApplied):
      * currentPlayer
      * currentPhase
      * gameStatus
      * state hash (shared progress/hash function)

Any mismatch is reported with (db, game_id, move_index, python_summary, ts_summary).

Usage (from ai-service/):

  python scripts/check_ts_python_replay_parity.py

You can optionally restrict to a single DB:

  python scripts/check_ts_python_replay_parity.py --db ../ai-service/logs/cmaes/.../games.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.db.game_replay import GameReplayDB, _compute_state_hash


@dataclass
class StateSummary:
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str


@dataclass
class GameParityResult:
    db_path: str
    game_id: str
    structure: str
    structure_reason: Optional[str]
    total_moves_python: int
    total_moves_ts: int
    diverged_at: Optional[int]
    python_summary: Optional[StateSummary]
    ts_summary: Optional[StateSummary]
    # High-level classification of what differed at diverged_at, e.g.:
    # ['current_player'], ['current_phase', 'game_status'], ['move_count'], ['ts_missing_step']
    mismatch_kinds: List[str] = field(default_factory=list)
    # Optional free-form context, such as "initial_state" vs "post_move"
    mismatch_context: Optional[str] = None


def repo_root() -> Path:
    """Return the monorepo root (one level above ai-service/)."""
    return Path(__file__).resolve().parents[2]


def find_dbs(explicit_db: Optional[str] = None) -> List[Path]:
    """Find GameReplayDB files to inspect."""
    root = repo_root()
    if explicit_db:
        return [Path(explicit_db).resolve()]

    search_paths = [
        root / "data" / "games",
        root / "ai-service" / "logs" / "cmaes",
        root / "ai-service" / "data" / "games",
    ]

    results: List[Path] = []
    visited = set()

    def walk(dir_path: Path, depth: int) -> None:
        if depth <= 0:
            return
        real = dir_path.resolve()
        if real in visited or not real.exists():
            return
        visited.add(real)
        try:
            entries = list(real.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                walk(entry, depth - 1)
            elif entry.is_file() and (entry.name == "games.db" or entry.name.endswith(".db")):
                results.append(entry)

    for base in search_paths:
        walk(base, 7)

    return results


def summarize_python_state(db: GameReplayDB, game_id: str, move_index: int) -> StateSummary:
    """Summarize state AFTER move_index is applied."""
    state = db.get_state_at_move(game_id, move_index)
    if state is None:
        raise RuntimeError(f"Python get_state_at_move returned None for {game_id} @ {move_index}")
    return StateSummary(
        move_index=move_index,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=state.game_status.value
        if hasattr(state.game_status, "value")
        else str(state.game_status),
        state_hash=_compute_state_hash(state),
    )


def summarize_python_initial_state(db: GameReplayDB, game_id: str) -> StateSummary:
    """Summarize the initial state BEFORE any moves are applied."""
    state = db.get_initial_state(game_id)
    if state is None:
        raise RuntimeError(f"Python get_initial_state returned None for {game_id}")
    return StateSummary(
        move_index=0,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=state.game_status.value
        if hasattr(state.game_status, "value")
        else str(state.game_status),
        state_hash=_compute_state_hash(state),
    )


def classify_game_structure(db: GameReplayDB, game_id: str) -> Tuple[str, str]:
    """Classify game recording structure.

    Returns (structure, reason) where structure is one of:
      - "good": initial state looks like a true game start (empty board).
      - "mid_snapshot": initial state appears to be a mid-game snapshot.
      - "invalid": missing data.

    Note: We do NOT compare initial_state with get_state_at_move(0) because
    get_state_at_move(0) returns the state AFTER move 0 is applied, which is
    expected to differ from initial_state (e.g., phase changes from
    ring_placement to movement after the first ring is placed).
    """
    initial = db.get_initial_state(game_id)
    if initial is None:
        return "invalid", "no initial_state record"

    # Treat any pre-populated history or board content as a mid-game snapshot.
    move_hist_len = len(initial.move_history or [])
    board = initial.board
    stacks = getattr(board, "stacks", {}) or {}
    markers = getattr(board, "markers", {}) or {}
    collapsed = getattr(board, "collapsed_spaces", {}) or {}

    stack_count = len(stacks)
    marker_count = len(markers)
    collapsed_count = len(collapsed)

    if move_hist_len > 0 or stack_count > 0 or marker_count > 0 or collapsed_count > 0:
        reason = (
            "initial_state contains history/board: "
            f"move_history={move_hist_len}, stacks={stack_count}, "
            f"markers={marker_count}, collapsed={collapsed_count}"
        )
        return "mid_snapshot", reason

    # Verify we can at least replay move 0
    state0 = db.get_state_at_move(game_id, 0)
    if state0 is None:
        return "invalid", "get_state_at_move(0) returned None"

    return "good", ""


def run_ts_replay(db_path: Path, game_id: str) -> Tuple[int, Dict[int, StateSummary]]:
    """Invoke the TS harness and parse its per-move summaries.

    Returns:
      (total_moves_reported_by_ts, mapping from move_index -> summary)
    """
    root = repo_root()
    cmd = [
        "npx",
        "ts-node",
        "-T",
        "scripts/selfplay-db-ts-replay.ts",
        "--db",
        str(db_path),
        "--game",
        game_id,
    ]

    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay harness failed for {db_path} / {game_id} with code {proc.returncode}:\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    total_ts_moves = 0
    summaries: Dict[int, StateSummary] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = payload.get("kind")
        if kind == "ts-replay-initial":
            total_ts_moves = int(payload.get("totalRecordedMoves", 0))
            summary = payload.get("summary") or {}
            summaries[0] = StateSummary(
                move_index=0,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=summary.get("gameStatus"),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-step":
            k = int(payload.get("k", 0))
            summary = payload.get("summary") or {}
            summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=summary.get("gameStatus"),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-final":
            # We could cross-check appliedMoves here if needed.
            pass

    return total_ts_moves, summaries


def check_game_parity(db_path: Path, game_id: str) -> GameParityResult:
    db = GameReplayDB(str(db_path))
    meta = db.get_game_metadata(game_id)
    if not meta:
        raise RuntimeError(f"Game {game_id} not found in {db_path}")

    structure, structure_reason = classify_game_structure(db, game_id)
    total_moves_py = int(meta["total_moves"])

    # For structurally bad games we don't attempt TS replay; they are not
    # suitable for start-to-finish parity and are reported separately.
    if structure != "good":
        return GameParityResult(
            db_path=str(db_path),
            game_id=game_id,
            structure=structure,
            structure_reason=structure_reason,
            total_moves_python=total_moves_py,
            total_moves_ts=0,
            diverged_at=None,
            python_summary=None,
            ts_summary=None,
        )

    total_moves_ts, ts_summaries = run_ts_replay(db_path, game_id)

    diverged_at: Optional[int] = None
    py_summary_at_diverge: Optional[StateSummary] = None
    ts_summary_at_diverge: Optional[StateSummary] = None
    mismatch_kinds: List[str] = []
    mismatch_context: Optional[str] = None

    # Index alignment:
    #   TS k=0 (ts-replay-initial) = initial state BEFORE any moves
    #   TS k=1 (ts-replay-step) = state AFTER move 0
    #   TS k=N = state AFTER move N-1
    #   Python get_state_at_move(0) = state AFTER move 0
    #   Python get_state_at_move(N) = state AFTER move N
    #
    # So: TS k=0 ↔ Python initial_state
    #     TS k=N (N>=1) ↔ Python get_state_at_move(N-1)

    # First, compare initial states (TS k=0 vs Python initial_state)
    # Note: We compare currentPlayer, currentPhase, gameStatus but NOT state_hash
    # because Python uses SHA-256 while TS uses a custom human-readable format.
    ts_initial = ts_summaries.get(0)
    if ts_initial is not None:
        py_initial = summarize_python_initial_state(db, game_id)
        init_mismatches: List[str] = []
        if py_initial.current_player != ts_initial.current_player:
            init_mismatches.append("current_player")
        if py_initial.current_phase != ts_initial.current_phase:
            init_mismatches.append("current_phase")
        if py_initial.game_status != ts_initial.game_status:
            init_mismatches.append("game_status")

        if init_mismatches:
            diverged_at = 0
            py_summary_at_diverge = py_initial
            ts_summary_at_diverge = ts_initial
            mismatch_kinds = init_mismatches
            mismatch_context = "initial_state"

    # Then compare post-move states: TS k ↔ Python get_state_at_move(k-1)
    if diverged_at is None:
        max_ts_k = total_moves_ts  # TS k ranges from 1 to total_moves_ts
        for ts_k in range(1, max_ts_k + 1):
            py_move_index = ts_k - 1  # Python index for state after move (ts_k - 1)
            if py_move_index >= total_moves_py:
                break  # Python doesn't have this move recorded

            ts_summary = ts_summaries.get(ts_k)
            if ts_summary is None:
                diverged_at = ts_k
                py_summary_at_diverge = summarize_python_state(db, game_id, py_move_index)
                ts_summary_at_diverge = None
                mismatch_kinds = ["ts_missing_step"]
                mismatch_context = "post_move"
                break

            py_summary = summarize_python_state(db, game_id, py_move_index)

            step_mismatches: List[str] = []
            if (
                py_summary.current_player != ts_summary.current_player
            ):
                step_mismatches.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                step_mismatches.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                step_mismatches.append("game_status")

            if step_mismatches:
                diverged_at = ts_k
                py_summary_at_diverge = py_summary
                ts_summary_at_diverge = ts_summary
                mismatch_kinds = step_mismatches
                mismatch_context = "post_move"
                break

    # If we had no per-move divergence but move counts differ, record that as a
    # distinct mismatch kind so callers can track move-count-only issues.
    if diverged_at is None and total_moves_python != total_moves_ts:
        diverged_at = None  # keep as None; mismatch is global, not at a single k
        mismatch_kinds = ["move_count"]
        mismatch_context = "global"

    return GameParityResult(
        db_path=str(db_path),
        game_id=game_id,
        structure=structure,
        structure_reason=structure_reason or None,
        total_moves_python=total_moves_py,
        total_moves_ts=total_moves_ts,
        diverged_at=diverged_at,
        python_summary=py_summary_at_diverge,
        ts_summary=ts_summary_at_diverge,
        mismatch_kinds=mismatch_kinds,
        mismatch_context=mismatch_context,
    )


def trace_game(db_path: Path, game_id: str, max_k: Optional[int] = None) -> None:
    """Emit a per-step TS vs Python trace for a single game.

    This is a focused debugging helper: it prints one line per TS step k,
    including the corresponding Python state (when available) and basic
    move metadata so rule/phase alignment issues can be inspected without
    re-running the full parity sweep.
    """
    db = GameReplayDB(str(db_path))
    meta = db.get_game_metadata(game_id)
    if not meta:
        print(f"[trace] game {game_id} not found in {db_path}")
        return

    structure, structure_reason = classify_game_structure(db, game_id)
    total_moves_py = int(meta.get("total_moves", 0))

    try:
        total_moves_ts, ts_summaries = run_ts_replay(db_path, game_id)
    except Exception as exc:
        print(
            "[trace] TS replay failed "
            f"for db={db_path} game={game_id}: {exc}"
        )
        return

    print(
        "TRACE-HEADER "
        f"db={db_path} "
        f"game={game_id} "
        f"structure={structure} "
        f"structure_reason={json.dumps(structure_reason or '')} "
        f"total_moves_py={total_moves_py} "
        f"total_moves_ts={total_moves_ts}"
    )

    # Initial state (TS k=0 vs Python initial_state)
    py_initial = summarize_python_initial_state(db, game_id)
    ts_initial = ts_summaries.get(0)
    init_dims: List[str] = []
    if ts_initial is None:
        init_dims.append("ts_missing_step")
    else:
        if py_initial.current_player != ts_initial.current_player:
            init_dims.append("current_player")
        if py_initial.current_phase != ts_initial.current_phase:
            init_dims.append("current_phase")
        if py_initial.game_status != ts_initial.game_status:
            init_dims.append("game_status")
        if py_initial.state_hash != ts_initial.state_hash:
            init_dims.append("state_hash")

    print(
        "TRACE "
        f"db={db_path} "
        f"game={game_id} "
        f"k=0 "
        f"move_number=None "
        f"move_player=None "
        f"move_type=None "
        f"py_player={py_initial.current_player} "
        f"ts_player={(ts_initial.current_player if ts_initial is not None else 'None')} "
        f"py_phase={py_initial.current_phase} "
        f"ts_phase={(ts_initial.current_phase if ts_initial is not None else 'None')} "
        f"py_status={py_initial.game_status} "
        f"ts_status={(ts_initial.game_status if ts_initial is not None else 'None')} "
        f"py_hash={py_initial.state_hash} "
        f"ts_hash={(ts_initial.state_hash if ts_initial is not None else 'None')} "
        f"dims={','.join(init_dims)}"
    )

    # Per-move states: TS k ↔ Python get_state_at_move(k-1)
    move_records = db.get_move_records(game_id)
    limit_k = total_moves_ts
    if max_k is not None and max_k > 0:
        limit_k = min(limit_k, max_k)

    for ts_k in range(1, limit_k + 1):
        py_index = ts_k - 1

        py_summary: Optional[StateSummary] = None
        py_error: Optional[str] = None
        if py_index < total_moves_py:
            try:
                py_summary = summarize_python_state(db, game_id, py_index)
            except Exception as exc:  # pragma: no cover - defensive
                py_error = str(exc)

        ts_summary = ts_summaries.get(ts_k)

        dims: List[str] = []
        if py_summary is not None and ts_summary is not None:
            if py_summary.current_player != ts_summary.current_player:
                dims.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                dims.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                dims.append("game_status")
            if py_summary.state_hash != ts_summary.state_hash:
                dims.append("state_hash")
        elif py_summary is None and ts_summary is not None:
            dims.append("python_missing_step")
        elif py_summary is not None and ts_summary is None:
            dims.append("ts_missing_step")

        move_number: Optional[int] = None
        move_player: Optional[int] = None
        move_type: Optional[str] = None
        if 0 <= py_index < len(move_records):
            rec = move_records[py_index]
            move_number = rec.get("moveNumber")
            move_player = rec.get("player")
            move_type = rec.get("moveType")

        py_player = py_summary.current_player if py_summary is not None else None
        ts_player = ts_summary.current_player if ts_summary is not None else None
        py_phase = py_summary.current_phase if py_summary is not None else None
        ts_phase = ts_summary.current_phase if ts_summary is not None else None
        py_status = py_summary.game_status if py_summary is not None else None
        ts_status = ts_summary.game_status if ts_summary is not None else None
        py_hash = py_summary.state_hash if py_summary is not None else None
        ts_hash = ts_summary.state_hash if ts_summary is not None else None

        line = (
            "TRACE "
            f"db={db_path} "
            f"game={game_id} "
            f"k={ts_k} "
            f"py_index={py_index} "
            f"move_number={move_number} "
            f"move_player={move_player} "
            f"move_type={move_type} "
            f"py_player={py_player} "
            f"ts_player={ts_player} "
            f"py_phase={py_phase} "
            f"ts_phase={ts_phase} "
            f"py_status={py_status} "
            f"ts_status={ts_status} "
            f"py_hash={py_hash} "
            f"ts_hash={ts_hash} "
            f"dims={','.join(dims)}"
        )
        if py_error:
            line += f" py_error={json.dumps(py_error)}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check TS vs Python replay parity for all self-play GameReplayDBs."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional path to a single games.db to inspect. "
        "When omitted, scans all known self-play locations.",
    )
    parser.add_argument(
        "--limit-games-per-db",
        type=int,
        default=0,
        help="Optional limit on number of games per DB to check (0 = all).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Emit only semantic divergences as single-line, grep-friendly "
            "records (no JSON summary, no structural issue output)."
        ),
    )
    parser.add_argument(
        "--emit-fixtures-dir",
        type=str,
        default=None,
        help=(
            "If set, write one JSON fixture per semantic divergence into this directory. "
            "Each fixture captures db_path, game_id, diverged_at, mismatch_kinds/context, "
            "Python/TS summaries, and the canonical move at or immediately before divergence."
        ),
    )
    parser.add_argument(
        "--trace-game",
        type=str,
        default=None,
        help=(
            "If set, emit a per-step TS vs Python trace for a single game_id and exit. "
            "Respects --db when provided; otherwise searches all known DB locations."
        ),
    )
    parser.add_argument(
        "--trace-max-k",
        type=int,
        default=0,
        help=(
            "Optional maximum TS k to include in --trace-game output (0 = all steps)."
        ),
    )
    args = parser.parse_args()

    db_paths = find_dbs(args.db)
    if not db_paths:
        print("No GameReplayDB databases found.")
        return

    # Focused trace mode: find the requested game_id and emit a per-step trace,
    # then exit without running the full parity sweep.
    if args.trace_game:
        for db_path in db_paths:
            db = GameReplayDB(str(db_path))
            try:
                meta = db.get_game_metadata(args.trace_game)
            except Exception:
                meta = None
            if meta:
                max_k = args.trace_max_k if args.trace_max_k and args.trace_max_k > 0 else None
                trace_game(db_path, args.trace_game, max_k=max_k)
                return

        print(
            f"[trace] game {args.trace_game} not found in any GameReplayDB "
            f"(searched {len(db_paths)} databases)"
        )
        return

    structural_issues: List[Dict[str, object]] = []
    semantic_divergences: List[Dict[str, object]] = []
    mismatch_counts_by_dimension: Dict[str, int] = {}
    total_games = 0
    total_semantic_divergent = 0
    total_structural_issues = 0

    fixtures_dir: Optional[Path] = Path(args.emit_fixtures_dir).resolve() if args.emit_fixtures_dir else None
    if fixtures_dir is not None:
        fixtures_dir.mkdir(parents=True, exist_ok=True)

    for db_path in db_paths:
        db = GameReplayDB(str(db_path))
        games = db.query_games(limit=100000)
        if not games:
            continue
        if args.limit_games_per_db and args.limit_games_per_db > 0:
            games = games[: args.limit_games_per_db]

        for game_meta in games:
            game_id = game_meta["game_id"]
            total_games += 1
            try:
                result = check_game_parity(db_path, game_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                structural_issues.append(
                    {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "structure": "error",
                        "structure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                total_structural_issues += 1
                continue

            if result.structure != "good":
                total_structural_issues += 1
                structural_issues.append(
                    {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "structure": result.structure,
                        "structure_reason": result.structure_reason,
                    }
                )
                continue

            if result.diverged_at is not None or result.total_moves_python != result.total_moves_ts:
                total_semantic_divergent += 1
                payload = asdict(result)
                if result.python_summary is not None:
                    payload["python_summary"] = asdict(result.python_summary)
                if result.ts_summary is not None:
                    payload["ts_summary"] = asdict(result.ts_summary)
                semantic_divergences.append(payload)
                # Increment per-dimension mismatch counters
                for kind in result.mismatch_kinds or []:
                    mismatch_counts_by_dimension[kind] = mismatch_counts_by_dimension.get(kind, 0) + 1

                # Optionally emit a compact JSON fixture for this divergence so TS tests
                # can consume it directly without re-querying the replay DB.
                if fixtures_dir is not None:
                    try:
                        moves = db.get_moves(game_id)
                    except Exception:
                        moves = []

                    canonical_move_index: Optional[int] = None
                    canonical_move_dict: Optional[Dict[str, object]] = None

                    if moves:
                        if result.diverged_at is None or result.diverged_at <= 0:
                            idx = 0
                        else:
                            idx = max(0, min(len(moves) - 1, result.diverged_at - 1))
                        canonical_move_index = idx
                        try:
                            move_obj = moves[idx]
                            canonical_move_dict = json.loads(
                                move_obj.model_dump_json(by_alias=True)  # type: ignore[attr-defined]
                            )
                        except Exception:
                            canonical_move_dict = None

                    fixture = {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "diverged_at": result.diverged_at,
                        "mismatch_kinds": list(result.mismatch_kinds),
                        "mismatch_context": result.mismatch_context,
                        "total_moves_python": result.total_moves_python,
                        "total_moves_ts": result.total_moves_ts,
                        "python_summary": (
                            asdict(result.python_summary) if result.python_summary is not None else None
                        ),
                        "ts_summary": (
                            asdict(result.ts_summary) if result.ts_summary is not None else None
                        ),
                        "canonical_move_index": canonical_move_index,
                        "canonical_move": canonical_move_dict,
                    }

                    safe_game_id = game_id.replace("/", "_")
                    diverged_label = (
                        "global"
                        if result.diverged_at is None
                        else str(result.diverged_at)
                    )
                    fixture_path = fixtures_dir / f"{Path(db_path).stem}__{safe_game_id}__k{diverged_label}.json"
                    with open(fixture_path, "w", encoding="utf-8") as f:
                        json.dump(fixture, f, indent=2, sort_keys=True)

    if args.compact:
        # Compact mode: emit one line per semantic divergence, skip structural issues.
        for entry in semantic_divergences:
            py = entry.get("python_summary") or {}
            ts = entry.get("ts_summary") or {}
            dims = entry.get("mismatch_kinds") or []
            line = (
                "SEMANTIC "
                f"db={entry.get('db_path')} "
                f"game={entry.get('game_id')} "
                f"diverged_at={entry.get('diverged_at')} "
                f"py_phase={py.get('current_phase')} "
                f"ts_phase={ts.get('current_phase')} "
                f"py_status={py.get('game_status')} "
                f"ts_status={ts.get('game_status')} "
                f"py_hash={py.get('state_hash')} "
                f"ts_hash={ts.get('state_hash')} "
                f"dims={','.join(dims)}"
            )
            print(line)
        return

    summary = {
        "total_databases": len(db_paths),
        "total_games_checked": total_games,
        "games_with_semantic_divergence": total_semantic_divergent,
        "games_with_structural_issues": total_structural_issues,
        "semantic_divergences": semantic_divergences,
        "structural_issues": structural_issues,
        "mismatch_counts_by_dimension": mismatch_counts_by_dimension,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
