"""Test that threshold values are imported from SSoT (app.config.thresholds).

This test ensures that all scripts and modules use the canonical threshold
values from app/config/thresholds.py rather than hardcoding values.

This follows the RULES_CANONICAL_SPEC.md SSoT principle:
"Single Source of Truth - Each piece of knowledge should have one
authoritative location in the codebase."
"""

import ast
import re
from pathlib import Path

import pytest

AI_SERVICE_ROOT = Path(__file__).parent.parent

# Patterns that indicate hardcoded threshold values that should be imported
SUSPICIOUS_PATTERNS = [
    # ELO thresholds - these exact values should come from thresholds.py
    (r"\b1650\b", "PRODUCTION_ELO_THRESHOLD"),  # Production ELO threshold
    (r"\b1200\b", "ELO_TIER_NOVICE"),  # Novice tier
    (r"\b1400\b", "ELO_TIER_INTERMEDIATE"),  # Intermediate tier
    (r"\b1500\b", "ELO_TIER_ADVANCED"),  # Advanced tier (also initial ELO)
    (r"\b1600\b", "ELO_TIER_EXPERT"),  # Expert tier
    (r"\b1800\b", "ELO_TIER_MASTER"),  # Master tier
    (r"\b2000\b", "ELO_TIER_GRANDMASTER"),  # Grandmaster tier
]

# Files that are allowed to define these values (the SSoT sources)
ALLOWED_DEFINITION_FILES = {
    "app/config/thresholds.py",  # The canonical SSoT
    "scripts/lib/elo_queries.py",  # Re-exports from thresholds.py
}

# Files that are known to have legitimate uses of these numbers
# (e.g., tests that verify the values, documentation, etc.)
ALLOWED_USAGE_FILES = {
    "tests/test_thresholds_usage.py",  # This test file
    "tests/test_thresholds.py",  # Tests for the thresholds module
}


def get_python_files() -> list[Path]:
    """Get all Python files in the ai-service directory."""
    files = []
    for pattern in ["app/**/*.py", "scripts/**/*.py"]:
        files.extend(AI_SERVICE_ROOT.glob(pattern))
    return [f for f in files if "__pycache__" not in str(f)]


def check_file_for_hardcoded_values(
    file_path: Path,
) -> list[tuple[int, str, str]]:
    """Check a file for hardcoded threshold values.

    Returns:
        List of (line_number, matched_value, suggested_import) tuples
    """
    relative_path = str(file_path.relative_to(AI_SERVICE_ROOT))

    # Skip allowed files
    if relative_path in ALLOWED_DEFINITION_FILES | ALLOWED_USAGE_FILES:
        return []

    issues = []

    try:
        content = file_path.read_text()
    except Exception:
        return []

    # Check if file imports from thresholds
    imports_thresholds = (
        "from app.config.thresholds import" in content
        or "from app.config import thresholds" in content
        or "from scripts.lib.elo_queries import" in content
    )

    # Parse the AST to find numeric literals
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = node.value
            line_no = node.lineno
            line_content = lines[line_no - 1] if line_no <= len(lines) else ""

            # Skip if this is clearly in a context where the value is being
            # compared to an imported threshold or is a different use
            if imports_thresholds and any(
                const in line_content
                for const in [
                    "PRODUCTION_ELO_THRESHOLD",
                    "ELO_TIER_",
                    "PRODUCTION_MIN_",
                ]
            ):
                continue

            # Check against suspicious patterns
            for pattern, suggested in SUSPICIOUS_PATTERNS:
                if re.match(pattern, str(int(value) if isinstance(value, float) else value)):
                    # Skip if it's in a comment
                    if line_content.strip().startswith("#"):
                        continue

                    # Skip if it looks like it's being used as a range or limit
                    # that's not an ELO threshold (e.g., "range(0, 2000)")
                    if "range(" in line_content and value in [1200, 1400, 1500, 2000]:
                        continue

                    # Skip if it's in a string literal (documentation)
                    if f'"{value}"' in line_content or f"'{value}'" in line_content:
                        continue

                    issues.append((line_no, str(value), suggested))

    return issues


def test_no_hardcoded_production_elo_threshold():
    """Ensure PRODUCTION_ELO_THRESHOLD (1650) is not hardcoded."""
    issues = []

    for file_path in get_python_files():
        file_issues = check_file_for_hardcoded_values(file_path)
        for line_no, value, suggested in file_issues:
            if suggested == "PRODUCTION_ELO_THRESHOLD":
                relative = file_path.relative_to(AI_SERVICE_ROOT)
                issues.append(f"{relative}:{line_no} - hardcoded {value}, use {suggested}")

    if issues:
        pytest.fail(
            f"Found hardcoded PRODUCTION_ELO_THRESHOLD values:\n"
            + "\n".join(issues)
            + "\n\nImport from app.config.thresholds instead."
        )


def test_thresholds_module_exists():
    """Ensure the canonical thresholds module exists."""
    thresholds_path = AI_SERVICE_ROOT / "app" / "config" / "thresholds.py"
    assert thresholds_path.exists(), "app/config/thresholds.py must exist as SSoT"


def test_thresholds_exports_required_values():
    """Ensure thresholds.py exports all required threshold constants."""
    from app.config.thresholds import (
        ELO_TIER_ADVANCED,
        ELO_TIER_EXPERT,
        ELO_TIER_GRANDMASTER,
        ELO_TIER_INTERMEDIATE,
        ELO_TIER_MASTER,
        ELO_TIER_NOVICE,
        PRODUCTION_ELO_THRESHOLD,
        PRODUCTION_MIN_GAMES,
    )

    # Verify values are reasonable (sanity check)
    assert PRODUCTION_ELO_THRESHOLD > 1500, "Production threshold should be above initial ELO"
    assert PRODUCTION_MIN_GAMES > 0, "Production min games should be positive"

    # Verify tier ordering
    assert ELO_TIER_NOVICE < ELO_TIER_INTERMEDIATE < ELO_TIER_ADVANCED
    assert ELO_TIER_ADVANCED < ELO_TIER_EXPERT < ELO_TIER_MASTER
    assert ELO_TIER_MASTER < ELO_TIER_GRANDMASTER


def test_elo_queries_uses_thresholds():
    """Ensure elo_queries.py imports from canonical thresholds."""
    elo_queries_path = AI_SERVICE_ROOT / "scripts" / "lib" / "elo_queries.py"

    if not elo_queries_path.exists():
        pytest.skip("elo_queries.py not yet created")

    content = elo_queries_path.read_text()

    assert "from app.config.thresholds import" in content, (
        "elo_queries.py should import from app.config.thresholds"
    )


def test_dashboard_scripts_import_from_ssot():
    """Ensure ELO dashboard scripts import thresholds from SSoT."""
    dashboard_scripts = [
        "scripts/elo_dashboard.py",
        "scripts/elo_leaderboard.py",
        "scripts/elo_alerts.py",
        "scripts/check_production_candidates.py",
        "scripts/auto_promote.py",
    ]

    for script_rel in dashboard_scripts:
        script_path = AI_SERVICE_ROOT / script_rel
        if not script_path.exists():
            continue

        content = script_path.read_text()

        # Should import from thresholds or elo_queries
        has_ssot_import = (
            "from app.config.thresholds import" in content
            or "from scripts.lib.elo_queries import" in content
        )

        assert has_ssot_import, (
            f"{script_rel} should import threshold constants from "
            "app.config.thresholds or scripts.lib.elo_queries"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
