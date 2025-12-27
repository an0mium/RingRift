#!/usr/bin/env python3
"""Fail if app/ imports from scripts/ (layering violation)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _find_layer_violations(app_root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in app_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("scripts"):
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("scripts"):
                    violations.append((path, node.lineno, node.module))
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
