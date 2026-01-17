"""Validated Model Reference - Type-safe binding of model path to configuration.

This module provides ValidatedModelRef, an immutable dataclass that guarantees
the model path has been validated against a specific board configuration.

Usage:
    from app.training.validated_model_ref import ValidatedModelRef

    # Create from path (extracts config from filename)
    ref = ValidatedModelRef.from_path("models/canonical_hex8_2p.pth")
    print(ref.config_key)  # "hex8_2p"
    print(ref.board_type)  # "hex8"
    print(ref.num_players)  # 2

    # Validate against expected config
    if ref.matches_config("hex8_2p"):
        # Safe to use for hex8_2p evaluation
        pass

    # Create with explicit config (validates match)
    ref = ValidatedModelRef.from_path_with_config(
        "models/canonical_hex8_2p.pth",
        expected_board_type="hex8",
        expected_num_players=2,
    )

January 2026: Created as part of model-config validation system to prevent
Elo tracking corruption from model-config mismatches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Valid board types
VALID_BOARD_TYPES = {"hex8", "hexagonal", "square8", "square19"}

# Valid player counts
VALID_PLAYER_COUNTS = {2, 3, 4}


class ModelConfigError(Exception):
    """Raised when model configuration is invalid or mismatched."""

    pass


@dataclass(frozen=True)
class ValidatedModelRef:
    """Immutable reference to a model with verified configuration.

    This class guarantees that:
    - model_path exists (or is a valid path string)
    - board_type and num_players are extracted/verified from the path
    - config_key is derived consistently

    Attributes:
        model_path: Absolute or relative path to the model file
        board_type: Board type (hex8, hexagonal, square8, square19)
        num_players: Number of players (2, 3, or 4)
    """

    model_path: str
    board_type: str
    num_players: int

    def __post_init__(self) -> None:
        """Validate the model reference after creation."""
        if self.board_type not in VALID_BOARD_TYPES:
            raise ModelConfigError(
                f"Invalid board_type '{self.board_type}'. "
                f"Must be one of: {sorted(VALID_BOARD_TYPES)}"
            )
        if self.num_players not in VALID_PLAYER_COUNTS:
            raise ModelConfigError(
                f"Invalid num_players {self.num_players}. "
                f"Must be one of: {sorted(VALID_PLAYER_COUNTS)}"
            )

    @property
    def config_key(self) -> str:
        """Return config key like 'hex8_2p'."""
        return f"{self.board_type}_{self.num_players}p"

    def matches_config(self, expected_config: str) -> bool:
        """Check if model matches expected config.

        Args:
            expected_config: Expected config key (e.g., "hex8_2p")

        Returns:
            True if this model's config matches the expected config
        """
        return self.config_key == expected_config

    def matches_board_config(
        self, board_type: str, num_players: int
    ) -> bool:
        """Check if model matches expected board type and player count.

        Args:
            board_type: Expected board type
            num_players: Expected number of players

        Returns:
            True if this model matches the expected configuration
        """
        return self.board_type == board_type and self.num_players == num_players

    @classmethod
    def from_path(cls, path: str | Path) -> ValidatedModelRef:
        """Create from model path, extracting config from filename.

        Args:
            path: Path to model file

        Returns:
            ValidatedModelRef with extracted configuration

        Raises:
            ModelConfigError: If config cannot be extracted from path
        """
        path_str = str(path)
        board_type, num_players = _extract_config_from_path(path_str)

        if board_type is None or num_players is None:
            raise ModelConfigError(
                f"Cannot extract config from path: {path_str}. "
                "Expected pattern like 'canonical_hex8_2p.pth' or 'hex8_2p/model.pth'"
            )

        return cls(
            model_path=path_str,
            board_type=board_type,
            num_players=num_players,
        )

    @classmethod
    def from_path_with_config(
        cls,
        path: str | Path,
        expected_board_type: str,
        expected_num_players: int,
        strict: bool = True,
    ) -> ValidatedModelRef:
        """Create from path with explicit config validation.

        Args:
            path: Path to model file
            expected_board_type: Expected board type
            expected_num_players: Expected number of players
            strict: If True, raise error on mismatch; if False, log warning

        Returns:
            ValidatedModelRef with validated configuration

        Raises:
            ModelConfigError: If strict=True and config doesn't match
        """
        path_str = str(path)
        detected_board, detected_players = _extract_config_from_path(path_str)

        # If we can extract config from path, validate it matches
        if detected_board is not None and detected_players is not None:
            if detected_board != expected_board_type or detected_players != expected_num_players:
                msg = (
                    f"Model path config mismatch: {path_str} appears to be for "
                    f"'{detected_board}_{detected_players}p' but expected "
                    f"'{expected_board_type}_{expected_num_players}p'"
                )
                if strict:
                    raise ModelConfigError(msg)
                else:
                    logger.warning(f"[ValidatedModelRef] {msg}")

        return cls(
            model_path=path_str,
            board_type=expected_board_type,
            num_players=expected_num_players,
        )

    @classmethod
    def try_from_path(cls, path: str | Path) -> ValidatedModelRef | None:
        """Try to create from path, returning None on failure.

        Args:
            path: Path to model file

        Returns:
            ValidatedModelRef if successful, None if config cannot be extracted
        """
        try:
            return cls.from_path(path)
        except ModelConfigError:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_path": self.model_path,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
        }

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"ValidatedModelRef({self.config_key}: {self.model_path})"


def _extract_config_from_path(model_path: str) -> tuple[str | None, int | None]:
    """Extract board type and num_players from a model file path.

    Handles patterns like:
    - models/canonical_hex8_2p.pth -> ("hex8", 2)
    - /path/to/ringrift_best_square8_4p.pth -> ("square8", 4)
    - models/hex8_2p/checkpoint_epoch10.pth -> ("hex8", 2)
    - hex8_3p_v5heavy.pth -> ("hex8", 3)

    Args:
        model_path: Path to model file

    Returns:
        Tuple of (board_type, num_players) or (None, None) if not found
    """
    # Build pattern for valid board types
    board_pattern = "|".join(sorted(VALID_BOARD_TYPES, key=len, reverse=True))

    # Pattern to match: board_type followed by _Np where N is 2, 3, or 4
    config_pattern = rf"({board_pattern})_([234])p"

    # Try filename first
    filename = Path(model_path).stem
    match = re.search(config_pattern, filename)
    if match:
        return match.group(1), int(match.group(2))

    # Try parent directory
    parent = Path(model_path).parent.name
    match = re.search(config_pattern, parent)
    if match:
        return match.group(1), int(match.group(2))

    # Try grandparent directory (for nested structures)
    grandparent = Path(model_path).parent.parent.name
    match = re.search(config_pattern, grandparent)
    if match:
        return match.group(1), int(match.group(2))

    return None, None


def validate_model_for_config(
    model_path: str | Path,
    expected_board_type: str,
    expected_num_players: int,
) -> tuple[bool, str]:
    """Validate that a model path is appropriate for the given config.

    This is a convenience function that wraps ValidatedModelRef creation.

    Args:
        model_path: Path to model file
        expected_board_type: Expected board type
        expected_num_players: Expected number of players

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        ref = ValidatedModelRef.from_path_with_config(
            model_path,
            expected_board_type,
            expected_num_players,
            strict=True,
        )
        return True, ""
    except ModelConfigError as e:
        return False, str(e)


def is_valid_config_key(value: str) -> bool:
    """Check if a string is a valid config key.

    Args:
        value: String to check

    Returns:
        True if it matches the config key pattern (e.g., "hex8_2p")
    """
    pattern = rf"^({'|'.join(VALID_BOARD_TYPES)})_[234]p$"
    return bool(re.match(pattern, value))


def parse_config_key(config_key: str) -> tuple[str, int] | None:
    """Parse a config key into board_type and num_players.

    Args:
        config_key: Config key like "hex8_2p"

    Returns:
        Tuple of (board_type, num_players) or None if invalid
    """
    pattern = rf"^({'|'.join(VALID_BOARD_TYPES)})_([234])p$"
    match = re.match(pattern, config_key)
    if match:
        return match.group(1), int(match.group(2))
    return None
