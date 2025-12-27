"""Tests for app.config.loader - Configuration Loading Utilities.

This module tests the configuration loading system including:
- ConfigSource class for source metadata
- load_config function for loading files
- save_config function for saving files
- env_override for environment variable overrides
- merge_configs for config merging
- validate_config for config validation
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config.loader import (
    ConfigLoadError,
    ConfigLoader,
    ConfigSource,
    env_override,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)


# =============================================================================
# ConfigSource Tests
# =============================================================================


class TestConfigSource:
    """Tests for ConfigSource class."""

    def test_detect_yaml_format(self):
        """Should detect YAML format from extension."""
        source = ConfigSource("config.yaml")
        assert source.format == "yaml"

    def test_detect_yml_format(self):
        """Should detect YAML format from .yml extension."""
        source = ConfigSource("config.yml")
        assert source.format == "yaml"

    def test_detect_json_format(self):
        """Should detect JSON format from extension."""
        source = ConfigSource("config.json")
        assert source.format == "json"

    def test_default_yaml_for_unknown(self):
        """Should default to YAML for unknown extensions."""
        source = ConfigSource("config.conf")
        assert source.format == "yaml"

    def test_explicit_format_override(self):
        """Should use explicit format over auto-detection."""
        source = ConfigSource("config.yaml", format="json")
        assert source.format == "json"

    def test_get_path_basic(self):
        """Should return basic path."""
        source = ConfigSource("/path/to/config.yaml")
        assert source.get_path() == Path("/path/to/config.yaml")

    def test_get_path_env_override(self):
        """Should use environment variable if set."""
        with patch.dict(os.environ, {"MY_CONFIG_PATH": "/override/path.yaml"}):
            source = ConfigSource("/default/path.yaml", env_var="MY_CONFIG_PATH")
            assert source.get_path() == Path("/override/path.yaml")

    def test_get_path_env_not_set(self):
        """Should use default path if env var not set."""
        source = ConfigSource("/default/path.yaml", env_var="NONEXISTENT_VAR")
        assert source.get_path() == Path("/default/path.yaml")

    def test_required_default_true(self):
        """Should default to required=True."""
        source = ConfigSource("config.yaml")
        assert source.required is True

    def test_required_can_be_false(self):
        """Should allow required=False."""
        source = ConfigSource("config.yaml", required=False)
        assert source.required is False


# =============================================================================
# load_config Tests
# =============================================================================


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_yaml_file(self):
        """Should load YAML config file."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            f.flush()
            try:
                config = load_config(f.name)
                assert config["key"] == "value"
                assert config["number"] == 42
            finally:
                os.unlink(f.name)

    def test_load_json_file(self):
        """Should load JSON config file."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('{"key": "value", "number": 42}')
            f.flush()
            try:
                config = load_config(f.name)
                assert config["key"] == "value"
                assert config["number"] == 42
            finally:
                os.unlink(f.name)

    def test_load_missing_required_file(self):
        """Should raise error for missing required file."""
        with pytest.raises(ConfigLoadError, match="not found"):
            load_config("/nonexistent/path.yaml", required=True)

    def test_load_missing_optional_file(self):
        """Should return empty dict for missing optional file."""
        config = load_config("/nonexistent/path.yaml", required=False)
        assert config == {}

    def test_load_with_defaults(self):
        """Should merge defaults with loaded config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            try:
                config = load_config(
                    f.name,
                    defaults={"key": "default", "other": "default_other"},
                )
                assert config["key"] == "value"  # Overridden
                assert config["other"] == "default_other"  # From defaults
            finally:
                os.unlink(f.name)

    def test_load_into_dataclass(self):
        """Should convert config to dataclass."""

        @dataclass
        class MyConfig:
            host: str = "localhost"
            port: int = 8080

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("host: remote.server\nport: 9000\n")
            f.flush()
            try:
                config = load_config(f.name, target=MyConfig)
                assert isinstance(config, MyConfig)
                assert config.host == "remote.server"
                assert config.port == 9000
            finally:
                os.unlink(f.name)

    def test_load_missing_optional_into_dataclass(self):
        """Should return default dataclass for missing optional file."""

        @dataclass
        class MyConfig:
            host: str = "localhost"
            port: int = 8080

        config = load_config(
            "/nonexistent/path.yaml",
            target=MyConfig,
            required=False,
        )
        assert isinstance(config, MyConfig)
        assert config.host == "localhost"
        assert config.port == 8080


# =============================================================================
# save_config Tests
# =============================================================================


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_yaml(self):
        """Should save config as YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config({"key": "value", "number": 42}, path)
            config = load_config(path)
            assert config["key"] == "value"
            assert config["number"] == 42
        finally:
            os.unlink(path)

    def test_save_json(self):
        """Should save config as JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_config({"key": "value", "number": 42}, path)
            config = load_config(path)
            assert config["key"] == "value"
            assert config["number"] == 42
        finally:
            os.unlink(path)

    def test_save_dataclass(self):
        """Should save dataclass as config."""

        @dataclass
        class MyConfig:
            host: str = "localhost"
            port: int = 8080

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(MyConfig(host="remote", port=9000), path)
            config = load_config(path)
            assert config["host"] == "remote"
            assert config["port"] == 9000
        finally:
            os.unlink(path)


# =============================================================================
# env_override Tests
# =============================================================================


class TestEnvOverride:
    """Tests for env_override function."""

    def test_override_simple_value(self):
        """Should override simple config value from env."""
        config = {"host": "localhost", "port": 8080}
        with patch.dict(os.environ, {"APP_HOST": "remote.server"}):
            result = env_override(config, "APP_")
            assert result["host"] == "remote.server"
            assert result["port"] == 8080

    def test_override_replaces_value(self):
        """Should replace config value with env var."""
        config = {"port": 8080}
        with patch.dict(os.environ, {"APP_PORT": "9000"}):
            result = env_override(config, "APP_")
            # Value is coerced to match original type
            assert result["port"] == 9000

    def test_no_override_when_env_not_set(self):
        """Should not modify config when env vars not set."""
        config = {"host": "localhost", "port": 8080}
        result = env_override(config, "APP_")
        assert result["host"] == "localhost"
        assert result["port"] == 8080


# =============================================================================
# merge_configs Tests
# =============================================================================


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_simple(self):
        """Should merge simple configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_merge_nested(self):
        """Should merge nested configs."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = merge_configs(base, override)
        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3
        assert result["outer"]["c"] == 4

    def test_merge_preserves_base(self):
        """Should not modify original configs."""
        base = {"a": 1}
        override = {"b": 2}
        merge_configs(base, override)
        assert "b" not in base

    def test_merge_override_replaces_non_dict(self):
        """Should replace non-dict values."""
        base = {"key": [1, 2, 3]}
        override = {"key": [4, 5, 6]}
        result = merge_configs(base, override)
        assert result["key"] == [4, 5, 6]


# =============================================================================
# validate_config Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_with_required_keys(self):
        """Should pass when required keys present."""
        config = {"host": "localhost", "port": 8080}
        is_valid, errors = validate_config(config, required_keys=["host", "port"])
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_key(self):
        """Should fail when required key missing."""
        config = {"host": "localhost"}
        is_valid, errors = validate_config(config, required_keys=["host", "port"])
        assert is_valid is False
        assert "port" in str(errors)

    def test_validate_with_custom_validator(self):
        """Should run custom validators."""
        config = {"port": 8080}
        is_valid, errors = validate_config(
            config,
            validators={"port": lambda x: x > 0 and x < 65536},
        )
        assert is_valid is True

    def test_validate_failing_custom_validator(self):
        """Should fail when custom validator fails."""
        config = {"port": 99999}
        is_valid, errors = validate_config(
            config,
            validators={"port": lambda x: x > 0 and x < 65536},
        )
        assert is_valid is False


# =============================================================================
# ConfigLoader Tests
# =============================================================================


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_loader_caches_result(self):
        """Should cache loaded config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            try:
                loader = ConfigLoader(f.name)
                config1 = loader.load()
                config2 = loader.load()
                assert config1 == config2
            finally:
                os.unlink(f.name)

    def test_loader_reloads_on_change(self):
        """Should reload when file changes."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value1\n")
            f.flush()
            path = f.name

        try:
            loader = ConfigLoader(path)
            config1 = loader.load()
            assert config1["key"] == "value1"

            # Modify file
            with open(path, "w") as f:
                f.write("key: value2\n")

            # Force reload
            config2 = loader.load(force_reload=True)
            assert config2["key"] == "value2"
        finally:
            os.unlink(path)
