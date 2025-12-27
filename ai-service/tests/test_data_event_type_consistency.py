"""Validate DataEventType references stay in sync with the enum definition."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA_EVENTS_PATH = ROOT / "app" / "distributed" / "data_events.py"


def _iter_python_files(root: Path) -> list[Path]:
    skip_dirs = {
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def _load_data_event_types() -> set[str]:
    text = DATA_EVENTS_PATH.read_text()
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*\"", line)
        if match:
            names.add(match.group(1))
    return names


def test_data_event_type_references_are_defined() -> None:
    enum_names = _load_data_event_types()
    pattern = re.compile(r"DataEventType\.([A-Z0-9_]+)")
    unknown: dict[str, list[str]] = {}

    for path in _iter_python_files(ROOT):
        text = path.read_text()
        for match in pattern.finditer(text):
            name = match.group(1)
            if name in enum_names:
                continue
            unknown.setdefault(name, []).append(str(path.relative_to(ROOT)))

    assert not unknown, f"Unknown DataEventType references: {unknown}"
