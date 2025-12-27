"""Tests for app.validation.core module.

Tests the core validation primitives: ValidationError, ValidationResult,
validate(), and validate_all().
"""

import pytest

from app.validation.core import (
    ValidationError,
    ValidationResult,
    Validator,
    validate,
    validate_all,
)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_creation(self):
        """Test creating a ValidationError with message."""
        error = ValidationError("test error")
        assert error.message == "test error"
        assert error.field is None
        assert str(error) == "test error"

    def test_with_field(self):
        """Test creating a ValidationError with field."""
        error = ValidationError("invalid value", field="username")
        assert error.message == "invalid value"
        assert error.field == "username"

    def test_is_exception(self):
        """Test that ValidationError is an Exception."""
        error = ValidationError("test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("test error")
        assert exc_info.value.message == "test error"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok_result(self):
        """Test creating a successful result."""
        result = ValidationResult.ok("test value")
        assert result.valid is True
        assert result.errors == []
        assert result.value == "test value"

    def test_fail_result(self):
        """Test creating a failed result."""
        result = ValidationResult.fail("error message")
        assert result.valid is False
        assert result.errors == ["error message"]

    def test_fail_with_field(self):
        """Test creating a failed result with field."""
        result = ValidationResult.fail("invalid", field="email")
        assert result.valid is False
        assert result.field == "email"

    def test_bool_true_when_valid(self):
        """Test bool conversion for valid result."""
        result = ValidationResult.ok()
        assert bool(result) is True

    def test_bool_false_when_invalid(self):
        """Test bool conversion for invalid result."""
        result = ValidationResult.fail("error")
        assert bool(result) is False

    def test_error_message_single(self):
        """Test error_message property with single error."""
        result = ValidationResult.fail("single error")
        assert result.error_message == "single error"

    def test_error_message_multiple(self):
        """Test error_message property with multiple errors."""
        result = ValidationResult(valid=False, errors=["error1", "error2", "error3"])
        assert result.error_message == "error1; error2; error3"

    def test_error_message_empty(self):
        """Test error_message property with no errors."""
        result = ValidationResult.ok()
        assert result.error_message == ""

    def test_merge_two_valid(self):
        """Test merging two valid results."""
        r1 = ValidationResult.ok("value1")
        r2 = ValidationResult.ok("value2")
        merged = r1.merge(r2)
        assert merged.valid is True
        assert merged.errors == []

    def test_merge_valid_with_invalid(self):
        """Test merging valid with invalid result."""
        r1 = ValidationResult.ok("value")
        r2 = ValidationResult.fail("error")
        merged = r1.merge(r2)
        assert merged.valid is False
        assert merged.errors == ["error"]

    def test_merge_two_invalid(self):
        """Test merging two invalid results."""
        r1 = ValidationResult.fail("error1")
        r2 = ValidationResult.fail("error2")
        merged = r1.merge(r2)
        assert merged.valid is False
        assert merged.errors == ["error1", "error2"]

    def test_merge_preserves_field(self):
        """Test that merge preserves field from first non-None."""
        r1 = ValidationResult(valid=True, field="field1")
        r2 = ValidationResult(valid=True, field="field2")
        merged = r1.merge(r2)
        assert merged.field == "field1"

    def test_merge_uses_second_field_if_first_none(self):
        """Test that merge uses second field if first is None."""
        r1 = ValidationResult(valid=True, field=None)
        r2 = ValidationResult(valid=True, field="field2")
        merged = r1.merge(r2)
        assert merged.field == "field2"


class TestValidate:
    """Tests for validate() function."""

    def test_no_validators(self):
        """Test validate with no validators returns ok."""
        result = validate("any value")
        assert result.valid is True
        assert result.value == "any value"

    def test_single_passing_validator(self):
        """Test validate with single passing validator."""
        def always_valid(v):
            return ValidationResult.ok(v)

        result = validate("test", always_valid)
        assert result.valid is True

    def test_single_failing_validator(self):
        """Test validate with single failing validator."""
        def always_fail(v):
            return ValidationResult.fail("always fails")

        result = validate("test", always_fail)
        assert result.valid is False
        assert "always fails" in result.errors

    def test_multiple_passing_validators(self):
        """Test validate with multiple passing validators."""
        def pass1(v):
            return ValidationResult.ok(v)

        def pass2(v):
            return ValidationResult.ok(v)

        result = validate("test", pass1, pass2)
        assert result.valid is True

    def test_multiple_one_fails(self):
        """Test validate with multiple validators where one fails."""
        def pass_validator(v):
            return ValidationResult.ok(v)

        def fail_validator(v):
            return ValidationResult.fail("failed")

        result = validate("test", pass_validator, fail_validator)
        assert result.valid is False
        assert "failed" in result.errors

    def test_validator_returning_bool_true(self):
        """Test validator returning True."""
        def bool_validator(v):
            return True

        result = validate("test", bool_validator)
        assert result.valid is True

    def test_validator_returning_bool_false(self):
        """Test validator returning False."""
        def bool_validator(v):
            return False

        result = validate("test", bool_validator)
        assert result.valid is False

    def test_validator_returning_string(self):
        """Test validator returning error string."""
        def string_validator(v):
            return "error message"

        result = validate("test", string_validator)
        assert result.valid is False
        assert "error message" in result.errors

    def test_validator_returning_none(self):
        """Test validator returning None (treated as ok)."""
        def none_validator(v):
            return None

        result = validate("test", none_validator)
        assert result.valid is True

    def test_validator_raising_validation_error(self):
        """Test validator that raises ValidationError."""
        def raising_validator(v):
            raise ValidationError("raised error", field="test_field")

        result = validate("test", raising_validator)
        assert result.valid is False
        assert "raised error" in result.errors

    def test_validator_raising_generic_exception(self):
        """Test validator that raises generic exception."""
        def raising_validator(v):
            raise ValueError("generic error")

        result = validate("test", raising_validator)
        assert result.valid is False
        assert "generic error" in result.errors

    def test_collects_all_errors(self):
        """Test that validate collects all errors from validators."""
        def fail1(v):
            return ValidationResult.fail("error1")

        def fail2(v):
            return ValidationResult.fail("error2")

        result = validate("test", fail1, fail2)
        assert result.valid is False
        assert len(result.errors) == 2
        assert "error1" in result.errors
        assert "error2" in result.errors


class TestValidateAll:
    """Tests for validate_all() function."""

    def test_empty_list(self):
        """Test validate_all with empty list."""
        result = validate_all([])
        assert result.valid is True

    def test_all_items_valid(self):
        """Test validate_all where all items pass."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        result = validate_all([1, 2, 3], is_positive)
        assert result.valid is True

    def test_one_item_invalid(self):
        """Test validate_all where one item fails."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        result = validate_all([1, -1, 3], is_positive)
        assert result.valid is False
        assert any("Item 1" in e for e in result.errors)

    def test_multiple_items_invalid(self):
        """Test validate_all where multiple items fail."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        result = validate_all([-1, 2, -3], is_positive)
        assert result.valid is False
        assert any("Item 0" in e for e in result.errors)
        assert any("Item 2" in e for e in result.errors)

    def test_stop_on_first(self):
        """Test validate_all with stop_on_first=True."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        result = validate_all([-1, -2, -3], is_positive, stop_on_first=True)
        assert result.valid is False
        # Should only have one error (stopped after first)
        assert len(result.errors) == 1

    def test_without_stop_on_first(self):
        """Test validate_all without stop_on_first."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        result = validate_all([-1, -2, -3], is_positive, stop_on_first=False)
        assert result.valid is False
        # Should have all errors
        assert len(result.errors) == 3

    def test_with_multiple_validators(self):
        """Test validate_all with multiple validators per item."""
        def is_positive(v):
            return ValidationResult.ok(v) if v > 0 else ValidationResult.fail("not positive")

        def is_even(v):
            return ValidationResult.ok(v) if v % 2 == 0 else ValidationResult.fail("not even")

        result = validate_all([2, 4, 6], is_positive, is_even)
        assert result.valid is True

        result = validate_all([2, 3, 6], is_positive, is_even)
        assert result.valid is False
