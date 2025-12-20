"""Tests for JSON utilities."""

import json
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest

from app.utils.json_utils import (
    JSONEncoder,
    dump,
    dumps,
    json_default,
    load_json,
    pretty_dumps,
    save_json,
)


class TestJSONEncoder:
    """Tests for JSONEncoder class."""

    def test_datetime(self):
        dt = datetime(2025, 12, 19, 10, 30, 0, tzinfo=timezone.utc)
        result = json.dumps({"time": dt}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["time"] == "2025-12-19T10:30:00+00:00"

    def test_date(self):
        d = date(2025, 12, 19)
        result = json.dumps({"date": d}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["date"] == "2025-12-19"

    def test_timedelta(self):
        td = timedelta(hours=2, minutes=30)
        result = json.dumps({"duration": td}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["duration"] == 9000.0  # 2.5 hours in seconds

    def test_path(self):
        p = Path("/tmp/test.txt")
        result = json.dumps({"path": p}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["path"] == "/tmp/test.txt"

    def test_enum(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = json.dumps({"color": Color.RED}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["color"] == "red"

    def test_uuid(self):
        u = UUID("12345678-1234-5678-1234-567812345678")
        result = json.dumps({"id": u}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["id"] == "12345678-1234-5678-1234-567812345678"

    def test_set(self):
        s = {1, 2, 3}
        result = json.dumps({"items": s}, cls=JSONEncoder)
        data = json.loads(result)
        assert sorted(data["items"]) == [1, 2, 3]

    def test_frozenset(self):
        s = frozenset([1, 2, 3])
        result = json.dumps({"items": s}, cls=JSONEncoder)
        data = json.loads(result)
        assert sorted(data["items"]) == [1, 2, 3]

    def test_bytes(self):
        b = b"\x00\x01\x02"
        result = json.dumps({"data": b}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["data"] == "AAEC"  # base64 encoded

    def test_object_with_to_dict(self):
        class MyObj:
            def to_dict(self):
                return {"value": 42}

        result = json.dumps({"obj": MyObj()}, cls=JSONEncoder)
        data = json.loads(result)
        assert data["obj"] == {"value": 42}

    def test_nested_special_types(self):
        data = {
            "time": datetime(2025, 12, 19, tzinfo=timezone.utc),
            "items": {1, 2},
            "path": Path("/tmp"),
        }
        result = json.dumps(data, cls=JSONEncoder)
        parsed = json.loads(result)
        assert "2025-12-19" in parsed["time"]
        assert sorted(parsed["items"]) == [1, 2]
        assert parsed["path"] == "/tmp"


class TestDumps:
    """Tests for dumps() function."""

    def test_basic_dict(self):
        result = dumps({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_with_indent(self):
        result = dumps({"key": "value"}, indent=2)
        assert "\n" in result
        assert "  " in result

    def test_with_sort_keys(self):
        result = dumps({"b": 1, "a": 2}, sort_keys=True)
        # 'a' should come before 'b' in output
        assert result.index('"a"') < result.index('"b"')

    def test_handles_datetime(self):
        dt = datetime(2025, 12, 19, tzinfo=timezone.utc)
        result = dumps({"time": dt})
        data = json.loads(result)
        assert "2025-12-19" in data["time"]

    def test_handles_path(self):
        result = dumps({"path": Path("/home/user")})
        data = json.loads(result)
        assert data["path"] == "/home/user"


class TestDump:
    """Tests for dump() function."""

    def test_writes_to_file(self, tmp_path):
        filepath = tmp_path / "test.json"
        with filepath.open("w") as f:
            dump({"key": "value"}, f)

        data = json.loads(filepath.read_text())
        assert data == {"key": "value"}

    def test_with_indent(self, tmp_path):
        filepath = tmp_path / "test.json"
        with filepath.open("w") as f:
            dump({"key": "value"}, f, indent=2)

        content = filepath.read_text()
        assert "\n" in content


class TestPrettyDumps:
    """Tests for pretty_dumps() function."""

    def test_indented(self):
        result = pretty_dumps({"key": "value"})
        assert "\n" in result
        assert "  " in result  # 2-space indent

    def test_handles_special_types(self):
        result = pretty_dumps({"time": datetime.now(timezone.utc)})
        assert isinstance(result, str)


class TestJsonDefault:
    """Tests for json_default() function."""

    def test_datetime(self):
        dt = datetime(2025, 12, 19, tzinfo=timezone.utc)
        result = json_default(dt)
        assert "2025-12-19" in result

    def test_path(self):
        result = json_default(Path("/tmp"))
        assert result == "/tmp"

    def test_unsupported_raises(self):
        class Unsupported:
            pass

        with pytest.raises(TypeError):
            json_default(Unsupported())

    def test_can_use_with_json_dumps(self):
        data = {"path": Path("/tmp"), "time": datetime.now(timezone.utc)}
        result = json.dumps(data, default=json_default)
        parsed = json.loads(result)
        assert parsed["path"] == "/tmp"


class TestLoadJson:
    """Tests for load_json() function."""

    def test_load_valid_json(self, tmp_path):
        filepath = tmp_path / "test.json"
        filepath.write_text('{"key": "value"}')

        result = load_json(filepath)
        assert result == {"key": "value"}

    def test_file_not_found_returns_default(self, tmp_path):
        filepath = tmp_path / "nonexistent.json"

        result = load_json(filepath, default={"default": True})
        assert result == {"default": True}

    def test_file_not_found_returns_none_by_default(self, tmp_path):
        filepath = tmp_path / "nonexistent.json"

        result = load_json(filepath)
        assert result is None

    def test_invalid_json_returns_default(self, tmp_path):
        filepath = tmp_path / "invalid.json"
        filepath.write_text("not valid json {{{")

        result = load_json(filepath, default=[])
        assert result == []

    def test_accepts_string_path(self, tmp_path):
        filepath = tmp_path / "test.json"
        filepath.write_text('{"key": "value"}')

        result = load_json(str(filepath))
        assert result == {"key": "value"}


class TestSaveJson:
    """Tests for save_json() function."""

    def test_save_basic(self, tmp_path):
        filepath = tmp_path / "test.json"
        save_json(filepath, {"key": "value"})

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "nested" / "dir" / "test.json"
        save_json(filepath, {"key": "value"})

        assert filepath.exists()

    def test_pretty_prints_by_default(self, tmp_path):
        filepath = tmp_path / "test.json"
        save_json(filepath, {"key": "value"})

        content = filepath.read_text()
        assert "\n" in content  # Has newlines (pretty printed)

    def test_atomic_write(self, tmp_path):
        filepath = tmp_path / "test.json"

        # Write initial content
        save_json(filepath, {"initial": True}, atomic=True)

        # Update atomically
        save_json(filepath, {"updated": True}, atomic=True)

        data = json.loads(filepath.read_text())
        assert data == {"updated": True}

    def test_non_atomic_write(self, tmp_path):
        filepath = tmp_path / "test.json"
        save_json(filepath, {"key": "value"}, atomic=False)

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data == {"key": "value"}

    def test_handles_special_types(self, tmp_path):
        filepath = tmp_path / "test.json"
        save_json(filepath, {
            "time": datetime(2025, 12, 19, tzinfo=timezone.utc),
            "path": Path("/tmp"),
        })

        data = json.loads(filepath.read_text())
        assert "2025-12-19" in data["time"]
        assert data["path"] == "/tmp"

    def test_accepts_string_path(self, tmp_path):
        filepath = tmp_path / "test.json"
        save_json(str(filepath), {"key": "value"})

        assert filepath.exists()

    def test_roundtrip(self, tmp_path):
        filepath = tmp_path / "test.json"
        original = {"string": "hello", "number": 42, "list": [1, 2, 3]}

        save_json(filepath, original)
        loaded = load_json(filepath)

        assert loaded == original
