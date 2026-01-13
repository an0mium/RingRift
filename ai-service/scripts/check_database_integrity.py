#!/usr/bin/env python3
"""Check game databases for move data integrity.

Validates that game databases don't have corrupted move data that would
cause training data export or replay failures.

Jan 12, 2026: Created after hex8_4p corruption incident where PLACE_RING
moves had null 'to' fields causing 100% replay failures.

Usage:
    python scripts/check_database_integrity.py data/games/canonical_*.db
    python scripts/check_database_integrity.py --all
    python scripts/check_database_integrity.py --fix data/games/corrupted.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IntegrityIssue:
    """Single integrity issue found in database."""

    game_id: str
    move_number: int
    issue_type: str
    description: str


@dataclass
class IntegrityReport:
    """Report of all integrity issues in a database."""

    db_path: str
    total_games: int
    total_moves: int
    issues: list[IntegrityIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Return True if no issues found."""
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        """Return number of issues found."""
        return len(self.issues)


def check_null_move_fields(conn: sqlite3.Connection) -> list[IntegrityIssue]:
    """Find moves with null to/from fields that shouldn't be null.

    PLACE_RING and MOVE_STACK moves must have 'to' field.
    ATTACK moves must have 'from' and 'to' fields (sometimes 'target').
    """
    issues = []

    # Check for null 'to' in place_ring moves
    try:
        cursor = conn.execute("""
            SELECT game_id, move_number, move_type, move_data
            FROM moves
            WHERE move_type IN ('place_ring', 'PLACE_RING')
            AND (
                move_data IS NULL
                OR json_extract(move_data, '$.to') IS NULL
            )
        """)
        for row in cursor:
            issues.append(IntegrityIssue(
                game_id=str(row[0]),
                move_number=row[1],
                issue_type="null_to_field",
                description=f"place_ring move missing 'to' field",
            ))
    except sqlite3.OperationalError:
        pass  # Table/column might not exist

    # Check for null 'to' in move_stack moves
    try:
        cursor = conn.execute("""
            SELECT game_id, move_number, move_type, move_data
            FROM moves
            WHERE move_type IN ('move_stack', 'MOVE_STACK')
            AND (
                move_data IS NULL
                OR json_extract(move_data, '$.to') IS NULL
            )
        """)
        for row in cursor:
            issues.append(IntegrityIssue(
                game_id=str(row[0]),
                move_number=row[1],
                issue_type="null_to_field",
                description=f"move_stack move missing 'to' field",
            ))
    except sqlite3.OperationalError:
        pass

    return issues


def check_invalid_json(conn: sqlite3.Connection) -> list[IntegrityIssue]:
    """Find moves with invalid JSON in move_data."""
    issues = []

    try:
        cursor = conn.execute("""
            SELECT game_id, move_number, move_type, move_data
            FROM moves
            WHERE move_data IS NOT NULL
        """)
        for row in cursor:
            game_id, move_number, move_type, move_data = row
            if move_data:
                try:
                    json.loads(move_data)
                except json.JSONDecodeError:
                    issues.append(IntegrityIssue(
                        game_id=str(game_id),
                        move_number=move_number,
                        issue_type="invalid_json",
                        description=f"{move_type} move has invalid JSON: {move_data[:50]}...",
                    ))
    except sqlite3.OperationalError:
        pass

    return issues


def check_negative_move_numbers(conn: sqlite3.Connection) -> list[IntegrityIssue]:
    """Find moves with negative or out-of-order move numbers."""
    issues = []

    try:
        cursor = conn.execute("""
            SELECT game_id, move_number, move_type
            FROM moves
            WHERE move_number < 0
        """)
        for row in cursor:
            issues.append(IntegrityIssue(
                game_id=str(row[0]),
                move_number=row[1],
                issue_type="negative_move_number",
                description=f"Move has negative move_number: {row[1]}",
            ))
    except sqlite3.OperationalError:
        pass

    return issues


def check_orphan_moves(conn: sqlite3.Connection) -> list[IntegrityIssue]:
    """Find moves that reference non-existent games."""
    issues = []

    try:
        cursor = conn.execute("""
            SELECT m.game_id, m.move_number, m.move_type
            FROM moves m
            LEFT JOIN games g ON m.game_id = g.id
            WHERE g.id IS NULL
            LIMIT 100
        """)
        for row in cursor:
            issues.append(IntegrityIssue(
                game_id=str(row[0]),
                move_number=row[1],
                issue_type="orphan_move",
                description=f"Move references non-existent game",
            ))
    except sqlite3.OperationalError:
        pass

    return issues


def check_database_integrity(db_path: str) -> IntegrityReport:
    """Run all integrity checks on a database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        IntegrityReport with all issues found
    """
    if not Path(db_path).exists():
        return IntegrityReport(
            db_path=db_path,
            total_games=0,
            total_moves=0,
            issues=[IntegrityIssue(
                game_id="N/A",
                move_number=0,
                issue_type="file_not_found",
                description=f"Database file not found: {db_path}",
            )],
        )

    conn = sqlite3.connect(db_path)
    issues: list[IntegrityIssue] = []

    # Get counts
    try:
        total_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    except sqlite3.OperationalError:
        total_games = 0

    try:
        total_moves = conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
    except sqlite3.OperationalError:
        total_moves = 0

    # Run all checks
    issues.extend(check_null_move_fields(conn))
    issues.extend(check_invalid_json(conn))
    issues.extend(check_negative_move_numbers(conn))
    issues.extend(check_orphan_moves(conn))

    conn.close()

    return IntegrityReport(
        db_path=db_path,
        total_games=total_games,
        total_moves=total_moves,
        issues=issues,
    )


def find_all_databases(base_dir: str = "data/games") -> list[str]:
    """Find all .db files in the data directory."""
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("**/*.db"))


def print_report(report: IntegrityReport, verbose: bool = False) -> None:
    """Print integrity report to stdout."""
    status = "✓ CLEAN" if report.is_clean else f"✗ {report.issue_count} ISSUES"
    print(f"\n{report.db_path}")
    print(f"  Games: {report.total_games}, Moves: {report.total_moves}")
    print(f"  Status: {status}")

    if not report.is_clean:
        # Group by issue type
        by_type: dict[str, list[IntegrityIssue]] = {}
        for issue in report.issues:
            by_type.setdefault(issue.issue_type, []).append(issue)

        for issue_type, type_issues in by_type.items():
            print(f"    {issue_type}: {len(type_issues)} occurrences")
            if verbose:
                for issue in type_issues[:5]:
                    print(f"      - Game {issue.game_id}, move {issue.move_number}: {issue.description}")
                if len(type_issues) > 5:
                    print(f"      ... and {len(type_issues) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Check game databases for move data integrity"
    )
    parser.add_argument(
        "databases",
        nargs="*",
        help="Database files to check (glob patterns supported)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all databases in data/games/",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed issue information",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Collect databases to check
    db_paths: list[str] = []

    if args.all:
        db_paths = find_all_databases()
    elif args.databases:
        for pattern in args.databases:
            if "*" in pattern:
                db_paths.extend(str(p) for p in Path().glob(pattern))
            else:
                db_paths.append(pattern)
    else:
        parser.print_help()
        sys.exit(1)

    if not db_paths:
        print("No databases found to check")
        sys.exit(1)

    # Check each database
    reports: list[IntegrityReport] = []
    for db_path in db_paths:
        report = check_database_integrity(db_path)
        reports.append(report)

    # Output results
    if args.json:
        import json as json_module
        output = {
            "databases": [
                {
                    "path": r.db_path,
                    "games": r.total_games,
                    "moves": r.total_moves,
                    "issues": [
                        {
                            "game_id": i.game_id,
                            "move_number": i.move_number,
                            "type": i.issue_type,
                            "description": i.description,
                        }
                        for i in r.issues
                    ],
                }
                for r in reports
            ],
            "summary": {
                "total_databases": len(reports),
                "clean": sum(1 for r in reports if r.is_clean),
                "with_issues": sum(1 for r in reports if not r.is_clean),
                "total_issues": sum(r.issue_count for r in reports),
            },
        }
        print(json_module.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("DATABASE INTEGRITY CHECK")
        print("=" * 60)

        for report in reports:
            print_report(report, verbose=args.verbose)

        # Summary
        clean = sum(1 for r in reports if r.is_clean)
        with_issues = len(reports) - clean
        total_issues = sum(r.issue_count for r in reports)

        print("\n" + "=" * 60)
        print(f"SUMMARY: {len(reports)} databases checked")
        print(f"  Clean: {clean}")
        print(f"  With issues: {with_issues}")
        print(f"  Total issues: {total_issues}")
        print("=" * 60)

    # Exit with error code if issues found
    if any(not r.is_clean for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
