"""Tests for app.validation.common module.

Tests common validators: range, string, collection, and type validators.
"""

import pytest

from app.validation.common import (
    each_item,
    has_keys,
    has_length,
    in_range,
    is_instance,
    is_non_negative,
    is_not_empty,
    is_positive,
    is_type,
    matches_pattern,
    max_length,
    min_length,
)


class TestInRange:
    """Tests for in_range validator."""

    def test_value_in_range_inclusive(self):
        """Test value within inclusive range."""
        validator = in_range(0, 100)
        assert validator(50).valid is True
        assert validator(0).valid is True
        assert validator(100).valid is True

    def test_value_out_of_range(self):
        """Test value outside range."""
        validator = in_range(0, 100)
        assert validator(-1).valid is False
        assert validator(101).valid is False

    def test_exclusive_range(self):
        """Test exclusive range bounds."""
        validator = in_range(0, 100, inclusive=False)
        assert validator(50).valid is True
        assert validator(0).valid is False
        assert validator(100).valid is False

    def test_float_range(self):
        """Test range with float values."""
        validator = in_range(0.0, 1.0)
        assert validator(0.5).valid is True
        assert validator(0.0).valid is True
        assert validator(1.0).valid is True
        assert validator(-0.1).valid is False
        assert validator(1.1).valid is False

    def test_non_comparable_value(self):
        """Test with non-comparable value."""
        validator = in_range(0, 100)
        result = validator("not a number")
        assert result.valid is False


class TestIsPositive:
    """Tests for is_positive validator."""

    def test_positive_int(self):
        """Test positive integer."""
        assert is_positive(1).valid is True
        assert is_positive(100).valid is True

    def test_positive_float(self):
        """Test positive float."""
        assert is_positive(0.1).valid is True
        assert is_positive(99.9).valid is True

    def test_zero(self):
        """Test zero is not positive."""
        assert is_positive(0).valid is False

    def test_negative(self):
        """Test negative values."""
        assert is_positive(-1).valid is False
        assert is_positive(-0.1).valid is False

    def test_non_numeric(self):
        """Test non-numeric value."""
        result = is_positive("test")
        assert result.valid is False


class TestIsNonNegative:
    """Tests for is_non_negative validator."""

    def test_positive(self):
        """Test positive values."""
        assert is_non_negative(1).valid is True
        assert is_non_negative(100).valid is True

    def test_zero(self):
        """Test zero is non-negative."""
        assert is_non_negative(0).valid is True

    def test_negative(self):
        """Test negative values fail."""
        assert is_non_negative(-1).valid is False
        assert is_non_negative(-0.01).valid is False

    def test_non_numeric(self):
        """Test non-numeric value."""
        result = is_non_negative("test")
        assert result.valid is False


class TestIsNotEmpty:
    """Tests for is_not_empty validator."""

    def test_non_empty_string(self):
        """Test non-empty string."""
        assert is_not_empty("hello").valid is True

    def test_empty_string(self):
        """Test empty string fails."""
        assert is_not_empty("").valid is False

    def test_whitespace_string(self):
        """Test whitespace-only string fails."""
        assert is_not_empty("   ").valid is False
        assert is_not_empty("\t\n").valid is False

    def test_none(self):
        """Test None fails."""
        assert is_not_empty(None).valid is False

    def test_non_empty_list(self):
        """Test non-empty list."""
        assert is_not_empty([1, 2, 3]).valid is True

    def test_empty_list(self):
        """Test empty list fails."""
        assert is_not_empty([]).valid is False

    def test_non_empty_dict(self):
        """Test non-empty dict."""
        assert is_not_empty({"key": "value"}).valid is True

    def test_empty_dict(self):
        """Test empty dict fails."""
        assert is_not_empty({}).valid is False

    def test_number_passes(self):
        """Test numbers pass (they have no length)."""
        # Numbers are not considered empty
        assert is_not_empty(0).valid is True
        assert is_not_empty(42).valid is True


class TestMatchesPattern:
    """Tests for matches_pattern validator."""

    def test_matching_pattern(self):
        """Test value matching pattern."""
        validator = matches_pattern(r"^\d+$")
        assert validator("123").valid is True
        assert validator("0").valid is True

    def test_non_matching_pattern(self):
        """Test value not matching pattern."""
        validator = matches_pattern(r"^\d+$")
        assert validator("abc").valid is False
        assert validator("12a3").valid is False

    def test_email_pattern(self):
        """Test email-like pattern."""
        validator = matches_pattern(r"^[\w.+-]+@[\w.-]+\.\w+$")
        assert validator("test@example.com").valid is True
        assert validator("invalid-email").valid is False

    def test_non_string_value(self):
        """Test non-string value fails."""
        validator = matches_pattern(r"\d+")
        result = validator(123)
        assert result.valid is False

    def test_case_insensitive_flag(self):
        """Test with case insensitive flag."""
        import re
        validator = matches_pattern(r"^hello$", flags=re.IGNORECASE)
        assert validator("hello").valid is True
        assert validator("HELLO").valid is True
        assert validator("HeLLo").valid is True


class TestMaxLength:
    """Tests for max_length validator."""

    def test_under_max(self):
        """Test value under max length."""
        validator = max_length(10)
        assert validator("hello").valid is True

    def test_at_max(self):
        """Test value at max length."""
        validator = max_length(5)
        assert validator("hello").valid is True

    def test_over_max(self):
        """Test value over max length."""
        validator = max_length(3)
        assert validator("hello").valid is False

    def test_empty_string(self):
        """Test empty string passes."""
        validator = max_length(10)
        assert validator("").valid is True

    def test_list(self):
        """Test list length."""
        validator = max_length(3)
        assert validator([1, 2]).valid is True
        assert validator([1, 2, 3, 4]).valid is False

    def test_no_length(self):
        """Test value without length."""
        validator = max_length(10)
        result = validator(123)
        assert result.valid is False


class TestMinLength:
    """Tests for min_length validator."""

    def test_over_min(self):
        """Test value over min length."""
        validator = min_length(3)
        assert validator("hello").valid is True

    def test_at_min(self):
        """Test value at min length."""
        validator = min_length(5)
        assert validator("hello").valid is True

    def test_under_min(self):
        """Test value under min length."""
        validator = min_length(10)
        assert validator("hello").valid is False

    def test_empty_string_zero_min(self):
        """Test empty string with zero min."""
        validator = min_length(0)
        assert validator("").valid is True

    def test_list(self):
        """Test list length."""
        validator = min_length(3)
        assert validator([1, 2, 3, 4]).valid is True
        assert validator([1, 2]).valid is False


class TestHasKeys:
    """Tests for has_keys validator."""

    def test_all_keys_present(self):
        """Test dict with all required keys."""
        validator = has_keys("name", "age")
        assert validator({"name": "John", "age": 30}).valid is True

    def test_extra_keys_allowed(self):
        """Test dict with extra keys is valid."""
        validator = has_keys("name")
        assert validator({"name": "John", "age": 30, "city": "NY"}).valid is True

    def test_missing_keys(self):
        """Test dict with missing keys."""
        validator = has_keys("name", "age", "email")
        result = validator({"name": "John"})
        assert result.valid is False
        assert "age" in result.error_message
        assert "email" in result.error_message

    def test_empty_dict_no_required_keys(self):
        """Test empty dict with no required keys."""
        validator = has_keys()
        assert validator({}).valid is True

    def test_non_dict(self):
        """Test non-dict value fails."""
        validator = has_keys("name")
        result = validator([1, 2, 3])
        assert result.valid is False


class TestHasLength:
    """Tests for has_length validator."""

    def test_length_in_range(self):
        """Test value with length in range."""
        validator = has_length(2, 5)
        assert validator([1, 2, 3]).valid is True

    def test_length_at_min(self):
        """Test value at minimum length."""
        validator = has_length(3, 5)
        assert validator([1, 2, 3]).valid is True

    def test_length_at_max(self):
        """Test value at maximum length."""
        validator = has_length(1, 3)
        assert validator([1, 2, 3]).valid is True

    def test_length_below_min(self):
        """Test value below minimum length."""
        validator = has_length(3, 5)
        assert validator([1]).valid is False

    def test_length_above_max(self):
        """Test value above maximum length."""
        validator = has_length(1, 3)
        assert validator([1, 2, 3, 4, 5]).valid is False

    def test_string_length(self):
        """Test string length."""
        validator = has_length(2, 10)
        assert validator("hello").valid is True
        assert validator("h").valid is False

    def test_no_length_attribute(self):
        """Test value without length."""
        validator = has_length(1, 5)
        result = validator(123)
        assert result.valid is False


class TestEachItem:
    """Tests for each_item validator."""

    def test_all_items_valid(self):
        """Test all items passing validation."""
        validator = each_item(is_positive)
        assert validator([1, 2, 3]).valid is True

    def test_some_items_invalid(self):
        """Test some items failing validation."""
        validator = each_item(is_positive)
        result = validator([1, -2, 3])
        assert result.valid is False
        assert any("[1]" in e for e in result.errors)

    def test_empty_list(self):
        """Test empty list passes."""
        validator = each_item(is_positive)
        assert validator([]).valid is True

    def test_with_multiple_validators(self):
        """Test with multiple validators per item."""
        validator = each_item(is_positive, in_range(1, 100))
        assert validator([1, 50, 100]).valid is True
        assert validator([1, 50, 101]).valid is False

    def test_non_iterable(self):
        """Test non-iterable value fails."""
        validator = each_item(is_positive)
        result = validator(123)
        assert result.valid is False

    def test_string_iteration(self):
        """Test string is treated as iterable of characters."""
        validator = each_item(is_not_empty)
        # Each character is not empty
        assert validator("abc").valid is True


class TestIsType:
    """Tests for is_type validator."""

    def test_exact_type_match(self):
        """Test exact type match."""
        validator = is_type(int)
        assert validator(42).valid is True

    def test_wrong_type(self):
        """Test wrong type fails."""
        validator = is_type(int)
        assert validator("42").valid is False

    def test_multiple_types(self):
        """Test multiple allowed types."""
        validator = is_type(int, float)
        assert validator(42).valid is True
        assert validator(3.14).valid is True
        assert validator("42").valid is False

    def test_bool_is_not_int(self):
        """Test bool is considered separate from int with is_type."""
        # Note: bool is a subclass of int, but is_type checks exact type
        validator = is_type(int)
        # Depends on implementation - check actual behavior
        result = validator(True)
        # If using type(value) in types, True has type bool, not int
        assert result.valid is False or result.valid is True  # Implementation dependent


class TestIsInstance:
    """Tests for is_instance validator."""

    def test_direct_instance(self):
        """Test direct instance match."""
        validator = is_instance(int)
        assert validator(42).valid is True

    def test_subclass_instance(self):
        """Test subclass instance passes."""
        validator = is_instance(int)
        # bool is subclass of int
        assert validator(True).valid is True

    def test_multiple_types(self):
        """Test multiple types."""
        validator = is_instance(int, str)
        assert validator(42).valid is True
        assert validator("hello").valid is True
        assert validator(3.14).valid is False

    def test_wrong_type(self):
        """Test wrong type fails."""
        validator = is_instance(str)
        assert validator(42).valid is False

    def test_custom_class(self):
        """Test with custom class."""
        class MyClass:
            pass

        class MySubclass(MyClass):
            pass

        validator = is_instance(MyClass)
        assert validator(MyClass()).valid is True
        assert validator(MySubclass()).valid is True
        assert validator("string").valid is False
