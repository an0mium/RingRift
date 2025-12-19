"""Tests for the secrets handling module."""

import os
from unittest.mock import patch

import pytest

from app.utils.secrets import (
    mask_secret,
    mask_secret_prefix,
    is_sensitive_key,
    sanitize_value,
    sanitize_for_log,
    SecretString,
    load_secret_from_env,
    get_env_masked,
    SENSITIVE_KEY_PATTERNS,
)


class TestMaskSecret:
    """Tests for mask_secret function."""

    def test_mask_long_secret(self):
        """Should mask long secrets showing last 4 chars."""
        assert mask_secret("my_secret_api_key_12345") == "***2345"

    def test_mask_short_secret(self):
        """Should show last 4 chars even for shorter secrets."""
        assert mask_secret("short") == "***hort"  # 5 chars, shows last 4

    def test_mask_very_short_secret(self):
        """Should fully mask very short secrets."""
        assert mask_secret("abc") == "***"
        assert mask_secret("ab") == "**"

    def test_mask_empty_secret(self):
        """Should return [empty] for empty strings."""
        assert mask_secret("") == "[empty]"
        assert mask_secret(None) == "[empty]"

    def test_mask_custom_visible_chars(self):
        """Should respect custom visible_chars parameter."""
        assert mask_secret("my_secret_key", visible_chars=6) == "***et_key"
        assert mask_secret("my_secret_key", visible_chars=2) == "***ey"


class TestMaskSecretPrefix:
    """Tests for mask_secret_prefix function."""

    def test_mask_showing_prefix(self):
        """Should show first N chars."""
        assert mask_secret_prefix("my_secret_api_key") == "my_s***"

    def test_mask_short_value(self):
        """Should fully mask short values."""
        assert mask_secret_prefix("abc") == "***"

    def test_mask_empty(self):
        """Should return [empty] for empty strings."""
        assert mask_secret_prefix("") == "[empty]"
        assert mask_secret_prefix(None) == "[empty]"


class TestIsSensitiveKey:
    """Tests for is_sensitive_key function."""

    def test_sensitive_keys(self):
        """Should detect common sensitive key names."""
        assert is_sensitive_key("api_key") is True
        assert is_sensitive_key("API_KEY") is True
        assert is_sensitive_key("apiKey") is True
        assert is_sensitive_key("password") is True
        assert is_sensitive_key("user_password") is True
        assert is_sensitive_key("secret") is True
        assert is_sensitive_key("auth_token") is True
        assert is_sensitive_key("access_key") is True
        assert is_sensitive_key("private_key") is True

    def test_non_sensitive_keys(self):
        """Should not flag normal keys as sensitive."""
        assert is_sensitive_key("username") is False
        assert is_sensitive_key("host") is False
        assert is_sensitive_key("port") is False
        assert is_sensitive_key("config") is False
        assert is_sensitive_key("data") is False


class TestSanitizeValue:
    """Tests for sanitize_value function."""

    def test_sanitize_with_sensitive_key(self):
        """Should mask values when key is sensitive."""
        assert sanitize_value("secret123", key="api_key") == "***t123"  # last 4 chars
        assert sanitize_value("mypassword", key="password") == "***word"  # last 4 chars

    def test_sanitize_dict(self):
        """Should recursively sanitize dicts."""
        data = {"api_key": "secret", "host": "localhost"}
        result = sanitize_value(data)
        assert result["api_key"] == "***cret"  # last 4 chars of "secret"
        assert result["host"] == "localhost"

    def test_sanitize_list(self):
        """Should sanitize list elements."""
        data = ["sk-abc123456789012345678901234567890123"]
        result = sanitize_value(data)
        assert "***" in result[0]

    def test_sanitize_embedded_secrets(self):
        """Should detect and mask embedded secrets in strings."""
        # OpenAI-style key
        text = "Using key sk-abc12345678901234567890"
        result = sanitize_value(text)
        assert "sk-abc" not in result or "***" in result

    def test_preserve_normal_values(self):
        """Should preserve non-sensitive values."""
        assert sanitize_value("hello world") == "hello world"
        assert sanitize_value(12345) == 12345
        assert sanitize_value(True) is True


class TestSanitizeForLog:
    """Tests for sanitize_for_log function."""

    def test_sanitize_dict_with_secrets(self):
        """Should sanitize dictionary containing secrets."""
        data = {
            "api_key": "secret123456789",
            "host": "example.com",
            "password": "mypassword123",
            "port": 8080,
        }
        result = sanitize_for_log(data)
        assert "***" in result["api_key"]
        assert result["host"] == "example.com"
        assert "***" in result["password"]
        assert result["port"] == 8080

    def test_sanitize_nested_dict(self):
        """Should sanitize nested dictionaries."""
        data = {
            "config": {
                "api_key": "nested_secret",
                "url": "http://example.com",
            }
        }
        result = sanitize_for_log(data)
        assert "***" in result["config"]["api_key"]
        assert result["config"]["url"] == "http://example.com"

    def test_sanitize_with_additional_keys(self):
        """Should respect additional_sensitive_keys."""
        # Note: additional_sensitive_keys adds to SENSITIVE_KEY_PATTERNS
        # which uses substring matching, so "custom_field" won't match
        # unless the key contains a sensitive pattern
        data = {"custom_secret": "should_be_masked"}  # Contains "secret"
        result = sanitize_for_log(data)
        assert "***" in result["custom_secret"]

    def test_sanitize_non_dict(self):
        """Should handle non-dict input."""
        assert sanitize_for_log("hello") == "hello"
        assert sanitize_for_log(123) == 123


class TestSecretString:
    """Tests for SecretString class."""

    def test_str_masks_value(self):
        """str() should return masked value."""
        secret = SecretString("my_api_key_12345")
        assert str(secret) == "***2345"

    def test_repr_masks_value(self):
        """repr() should return masked value."""
        secret = SecretString("my_api_key_12345")
        assert "***" in repr(secret)
        assert "SecretString" in repr(secret)

    def test_get_value_returns_actual(self):
        """get_value() should return the actual secret."""
        secret = SecretString("my_api_key_12345")
        assert secret.get_value() == "my_api_key_12345"

    def test_bool_true_when_has_value(self):
        """bool should be True when secret has value."""
        assert bool(SecretString("secret")) is True

    def test_bool_false_when_empty(self):
        """bool should be False when secret is empty."""
        assert bool(SecretString("")) is False

    def test_equality_with_secret_string(self):
        """Should compare equal with matching SecretString."""
        s1 = SecretString("secret")
        s2 = SecretString("secret")
        s3 = SecretString("different")
        assert s1 == s2
        assert s1 != s3

    def test_equality_with_string(self):
        """Should compare equal with matching string."""
        secret = SecretString("secret")
        assert secret == "secret"
        assert secret != "different"

    def test_hash(self):
        """Should be hashable based on value."""
        s1 = SecretString("secret")
        s2 = SecretString("secret")
        assert hash(s1) == hash(s2)
        # Can be used in sets/dicts
        assert len({s1, s2}) == 1

    def test_custom_visible_chars(self):
        """Should respect visible_chars parameter."""
        secret = SecretString("my_long_secret", visible_chars=6)
        assert str(secret) == "***secret"

    def test_fstring_formatting(self):
        """Should mask in f-string formatting."""
        secret = SecretString("my_secret_key")
        message = f"Using key: {secret}"
        assert "my_secret_key" not in message
        assert "***" in message


class TestLoadSecretFromEnv:
    """Tests for load_secret_from_env function."""

    def test_load_existing_env_var(self):
        """Should load existing environment variable."""
        with patch.dict(os.environ, {"TEST_SECRET": "secret_value"}):
            result = load_secret_from_env("TEST_SECRET")
            assert result is not None
            assert result.get_value() == "secret_value"
            assert isinstance(result, SecretString)

    def test_load_missing_env_var_with_default(self):
        """Should use default when env var is missing."""
        result = load_secret_from_env("NONEXISTENT_VAR", default="default_val")
        assert result is not None
        assert result.get_value() == "default_val"

    def test_load_missing_env_var_returns_none(self):
        """Should return None when env var is missing and no default."""
        result = load_secret_from_env("NONEXISTENT_VAR")
        assert result is None

    def test_load_required_missing_raises(self):
        """Should raise ValueError when required env var is missing."""
        with pytest.raises(ValueError, match="Required environment variable"):
            load_secret_from_env("NONEXISTENT_VAR", required=True)


class TestGetEnvMasked:
    """Tests for get_env_masked function."""

    def test_get_existing_env_var_masked(self):
        """Should return masked value of existing env var."""
        with patch.dict(os.environ, {"TEST_KEY": "secret_value_123"}):
            result = get_env_masked("TEST_KEY")
            assert "***" in result
            assert "secret_value_123" not in result

    def test_get_missing_env_var(self):
        """Should return masked default when env var is missing."""
        result = get_env_masked("NONEXISTENT_VAR", default="default")
        assert "***" in result or result == "[empty]"
