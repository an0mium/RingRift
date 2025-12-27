#!/usr/bin/env python3
"""Fail if app/ imports from scripts/ (layering violation)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


_IMPORT_RE = re.compile(r"^\s*(from|import)\s+scripts(\.|\\b)")


def _find_layer_violations(app_root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []

    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", r"^\s*(from|import)\s+scripts(\.|\\b)", str(app_root)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().splitlines():
                path_str, lineno_str, content = line.split(":", 2)
                violations.append((Path(path_str), int(lineno_str), content.strip()))
        return violations

    for path in app_root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _IMPORT_RE.match(line):
                violations.append((path, lineno, line.strip()))
    return violations


def main() -> int:
    ai_service_root = Path(__file__).resolve().parents[1]
    app_root = ai_service_root / "app"
    violations = _find_layer_violations(app_root)
    if not violations:
        return 0

    print("Layer violations detected (app -> scripts):")
    for path, lineno, module in sorted(violations):
        rel_path = path.relative_to(ai_service_root)
        print(f"- {rel_path}:{lineno}: {module}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
