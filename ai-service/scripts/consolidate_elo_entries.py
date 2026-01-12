#!/usr/bin/env python3
"""Consolidate duplicate Elo entries for identical model files.

This script scans the unified_elo.db database and model files to:
1. Compute SHA256 hashes for all model files
2. Find participant IDs that reference the same model content
3. Create aliases from secondary IDs to the primary (most games)
4. Optionally merge Elo ratings (weighted average by games)

This fixes the stale Elo problem where canonical models appear weak because
their Elo was computed with an older model version, while ringrift_best_*
(same file!) has the current Elo.

Usage:
    # Dry run - preview what would be done
    python scripts/consolidate_elo_entries.py --dry-run

    # Execute consolidation
    python scripts/consolidate_elo_entries.py

    # Verbose output
    python scripts/consolidate_elo_entries.py --verbose

    # Specify custom database
    python scripts/consolidate_elo_entries.py --db data/unified_elo.db

January 2026: Created for Elo/Model Identity Tracking fix (Priority 0).
"""
from __future__ import annotations


import argparse
import hashlib
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Model directories to scan
MODEL_DIRS = [
    AI_SERVICE_ROOT / "models",
    AI_SERVICE_ROOT / "models" / "production",
]


@dataclass
class ModelInfo:
    """Information about a model file."""
    path: Path
    content_hash: str
    file_size: int


@dataclass
class ParticipantInfo:
    """Information about a participant in the Elo database."""
    participant_id: str
    board_type: str
    num_players: int
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int


@dataclass
class ConsolidationGroup:
    """A group of participants that reference the same model content."""
    content_hash: str
    model_paths: list[Path] = field(default_factory=list)
    participants: list[ParticipantInfo] = field(default_factory=list)
    primary_participant: ParticipantInfo | None = None
    aliases: list[ParticipantInfo] = field(default_factory=list)


def compute_model_hash(path: Path) -> str | None:
    """Compute SHA256 hash of model file content."""
    if not path.exists() or not path.is_file():
        return None

    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Warning: Could not hash {path}: {e}")
        return None


def discover_model_files(verbose: bool = False) -> dict[str, list[ModelInfo]]:
    """Discover all model files and compute their hashes.

    Returns:
        Dict mapping content_hash -> list of ModelInfo with that hash
    """
    hash_to_models: dict[str, list[ModelInfo]] = defaultdict(list)

    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            continue

        for path in model_dir.rglob("*.pth"):
            # Skip symlinks (they'll point to the same content)
            if path.is_symlink():
                if verbose:
                    print(f"  Skipping symlink: {path}")
                continue

            content_hash = compute_model_hash(path)
            if content_hash:
                info = ModelInfo(
                    path=path,
                    content_hash=content_hash,
                    file_size=path.stat().st_size,
                )
                hash_to_models[content_hash].append(info)
                if verbose:
                    print(f"  {path.name}: {content_hash[:12]}...")

    return hash_to_models


def get_participants_from_db(db_path: Path) -> list[ParticipantInfo]:
    """Get all participants from the Elo database."""
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return []

    participants = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("""
            SELECT participant_id, board_type, num_players,
                   rating, games_played, wins, losses, draws
            FROM elo_ratings
            WHERE games_played > 0
            ORDER BY games_played DESC
        """)
        for row in cursor:
            participants.append(ParticipantInfo(
                participant_id=row["participant_id"],
                board_type=row["board_type"],
                num_players=row["num_players"],
                rating=row["rating"],
                games_played=row["games_played"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
            ))
    finally:
        conn.close()

    return participants


def participant_to_model_path(participant_id: str) -> Path | None:
    """Convert a participant ID to a model path (heuristic).

    Participant IDs typically follow patterns like:
    - canonical_hex8_2p
    - ringrift_best_square8_4p
    - models/production/hex8_2p/model_xyz.pth

    Returns the most likely model path, or None if not determinable.
    """
    # Try direct path first (for full-path participant IDs)
    if "/" in participant_id or participant_id.endswith(".pth"):
        path = Path(participant_id)
        if not path.is_absolute():
            path = AI_SERVICE_ROOT / participant_id
        if path.exists():
            return path

    # Try canonical pattern
    if participant_id.startswith("canonical_"):
        config = participant_id.replace("canonical_", "")
        path = AI_SERVICE_ROOT / "models" / f"canonical_{config}.pth"
        if path.exists():
            return path

    # Try ringrift_best pattern (resolves symlink to canonical)
    if participant_id.startswith("ringrift_best_"):
        config = participant_id.replace("ringrift_best_", "")
        path = AI_SERVICE_ROOT / "models" / f"ringrift_best_{config}.pth"
        if path.exists():
            # Resolve symlink to get actual file
            return path.resolve() if path.is_symlink() else path

    return None


def find_consolidation_groups(
    hash_to_models: dict[str, list[ModelInfo]],
    participants: list[ParticipantInfo],
    verbose: bool = False,
) -> list[ConsolidationGroup]:
    """Find groups of participants that need consolidation.

    Returns groups where multiple participant IDs reference the same model content.
    """
    # Build mapping from participant_id to model hash
    participant_to_hash: dict[str, str] = {}
    for participant in participants:
        model_path = participant_to_model_path(participant.participant_id)
        if model_path:
            # Find hash for this path
            for content_hash, model_infos in hash_to_models.items():
                for model_info in model_infos:
                    if model_info.path.resolve() == model_path.resolve():
                        participant_to_hash[participant.participant_id] = content_hash
                        if verbose:
                            print(f"  {participant.participant_id} -> {content_hash[:12]}...")
                        break

    # Group participants by (hash, board_type, num_players)
    groups: dict[tuple[str, str, int], ConsolidationGroup] = {}

    for participant in participants:
        content_hash = participant_to_hash.get(participant.participant_id)
        if not content_hash:
            continue

        key = (content_hash, participant.board_type, participant.num_players)
        if key not in groups:
            groups[key] = ConsolidationGroup(content_hash=content_hash)
            # Add model paths for this hash
            for model_info in hash_to_models.get(content_hash, []):
                groups[key].model_paths.append(model_info.path)

        groups[key].participants.append(participant)

    # Filter to groups with multiple participants (need consolidation)
    consolidation_groups = []
    for key, group in groups.items():
        if len(group.participants) > 1:
            # Sort by games_played descending, pick first as primary
            group.participants.sort(key=lambda p: p.games_played, reverse=True)
            group.primary_participant = group.participants[0]
            group.aliases = group.participants[1:]
            consolidation_groups.append(group)

    return consolidation_groups


def apply_consolidation(
    db_path: Path,
    groups: list[ConsolidationGroup],
    dry_run: bool = True,
    verbose: bool = False,
) -> dict:
    """Apply consolidation by creating aliases in the database.

    Returns:
        Dict with consolidation statistics
    """
    stats = {
        "groups_processed": 0,
        "aliases_created": 0,
        "skipped": 0,
        "errors": 0,
    }

    if not groups:
        print("No consolidation needed - all participants are unique.")
        return stats

    if dry_run:
        print("\n[DRY RUN] Would create the following aliases:\n")
    else:
        print("\nCreating aliases...\n")

    conn = None
    if not dry_run:
        conn = sqlite3.connect(db_path)
        # Ensure tables exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS participant_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_participant_id TEXT NOT NULL,
                alias_participant_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(primary_participant_id, alias_participant_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_participant_aliases_primary
            ON participant_aliases(primary_participant_id)
        """)

    try:
        for group in groups:
            stats["groups_processed"] += 1
            primary = group.primary_participant
            if not primary:
                continue

            print(f"Group: {group.content_hash[:12]}... ({primary.board_type}_{primary.num_players}p)")
            print(f"  Primary: {primary.participant_id}")
            print(f"    Elo: {primary.rating:.0f}, Games: {primary.games_played}")

            for alias in group.aliases:
                print(f"  Alias: {alias.participant_id}")
                print(f"    Elo: {alias.rating:.0f}, Games: {alias.games_played}")

                if not dry_run and conn:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO participant_aliases
                            (primary_participant_id, alias_participant_id, content_sha256)
                            VALUES (?, ?, ?)
                        """, (primary.participant_id, alias.participant_id, group.content_hash))
                        stats["aliases_created"] += 1
                        if verbose:
                            print(f"    ✓ Alias created")
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"    ✗ Error: {e}")
                else:
                    stats["aliases_created"] += 1

            print()

        if conn:
            conn.commit()

    finally:
        if conn:
            conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate duplicate Elo entries for identical model files."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=AI_SERVICE_ROOT / "data" / "unified_elo.db",
        help="Path to unified_elo.db database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Elo Entry Consolidation Tool")
    print("=" * 60)
    print(f"\nDatabase: {args.db}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print()

    # Step 1: Discover model files and compute hashes
    print("Step 1: Discovering model files...")
    hash_to_models = discover_model_files(verbose=args.verbose)
    total_models = sum(len(v) for v in hash_to_models.values())
    unique_hashes = len(hash_to_models)
    print(f"  Found {total_models} model files with {unique_hashes} unique hashes")

    # Step 2: Get participants from database
    print("\nStep 2: Loading participants from database...")
    participants = get_participants_from_db(args.db)
    print(f"  Found {len(participants)} participants with games")

    # Step 3: Find consolidation groups
    print("\nStep 3: Finding participants with same model content...")
    groups = find_consolidation_groups(hash_to_models, participants, verbose=args.verbose)
    if groups:
        print(f"  Found {len(groups)} groups needing consolidation")
    else:
        print("  No duplicate entries found - database is clean!")

    # Step 4: Apply consolidation
    print("\nStep 4: Applying consolidation...")
    stats = apply_consolidation(args.db, groups, dry_run=args.dry_run, verbose=args.verbose)

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Groups processed: {stats['groups_processed']}")
    print(f"  Aliases created: {stats['aliases_created']}")
    print(f"  Errors: {stats['errors']}")

    if args.dry_run and stats['aliases_created'] > 0:
        print("\n⚠ This was a dry run. Run without --dry-run to apply changes.")

    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
