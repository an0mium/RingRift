"""Data Validation Package for RingRift AI Service.

Provides standardized validation utilities:
- Common validators (range, pattern, enum)
- Validation result handling
- Pydantic integration helpers
- Domain-specific validators

Usage:
    from app.validation import (
        Validator,
        validate,
        ValidationError,
        # Common validators
        in_range,
        matches_pattern,
        is_positive,
        is_not_empty,
    )

    # Basic validation
    result = validate(value, in_range(0, 100))
    if not result:
        print(result.errors)

    # Chained validation
    result = validate(
        config,
        has_keys("host", "port"),
        each_value(is_not_empty),
    )

    # Domain validators
    from app.validation import (
        is_valid_config_key,
        is_valid_model_path,
        is_valid_elo,
    )
"""

from app.validation.core import (
    Validator,
    ValidationResult,
    ValidationError,
    validate,
    validate_all,
)
from app.validation.common import (
    # Range validators
    in_range,
    is_positive,
    is_non_negative,
    # String validators
    is_not_empty,
    matches_pattern,
    max_length,
    # Collection validators
    has_keys,
    has_length,
    each_item,
    # Type validators
    is_type,
    is_instance,
)
from app.validation.domain import (
    is_valid_config_key,
    is_valid_board_type,
    is_valid_elo,
    is_valid_model_path,
)

__all__ = [
    # Core
    "Validator",
    "ValidationResult",
    "ValidationError",
    "validate",
    "validate_all",
    # Common
    "in_range",
    "is_positive",
    "is_non_negative",
    "is_not_empty",
    "matches_pattern",
    "max_length",
    "has_keys",
    "has_length",
    "each_item",
    "is_type",
    "is_instance",
    # Domain
    "is_valid_config_key",
    "is_valid_board_type",
    "is_valid_elo",
    "is_valid_model_path",
]
