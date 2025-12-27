"""Protocol Utilities for P2P Mixins.

Provides common utilities for protocol mixins (SWIM, Raft, HTTP fallback).
Used by consensus_mixin.py and membership_mixin.py.

December 2025: Created to consolidate common patterns across protocol mixins.

Usage:
    from scripts.p2p.protocol_utils import (
        safe_import,
        ProtocolAvailability,
        get_feature_flag,
        log_protocol_status,
    )

    # Safe import with fallback
    pysyncobj, RAFT_AVAILABLE = safe_import("pysyncobj", "SyncObj")

    # Check feature flag with default
    raft_enabled = get_feature_flag("RINGRIFT_RAFT_ENABLED", False)

    # Log protocol availability
    log_protocol_status("raft", RAFT_AVAILABLE, raft_enabled)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Safe Import Utilities
# =============================================================================

def safe_import(module_name: str, *attr_names: str) -> tuple[Any, ...]:
    """Safely import a module and optional attributes.

    Returns tuple of (module_or_None, attr1_or_None, attr2_or_None, ..., available_bool)

    Args:
        module_name: Module to import (e.g., "pysyncobj")
        attr_names: Attributes to extract from module (e.g., "SyncObj", "replicated")

    Returns:
        Tuple of (module, *attrs, is_available) where is_available is always last

    Example:
        # Import module with multiple attributes
        pysyncobj, SyncObj, replicated, available = safe_import(
            "pysyncobj", "SyncObj", "replicated"
        )

        # Import just module
        torch, available = safe_import("torch")
    """
    try:
        import importlib
        module = importlib.import_module(module_name)

        attrs = []
        for attr_name in attr_names:
            try:
                attrs.append(getattr(module, attr_name))
            except AttributeError:
                attrs.append(None)

        return (module, *attrs, True)

    except ImportError:
        # Return None for module and all attrs, plus False for availability
        return (None,) * (len(attr_names) + 1) + (False,)


def safe_import_from(module_name: str, name: str) -> tuple[Any, bool]:
    """Safely import a single item from a module.

    Args:
        module_name: Module path (e.g., "pysyncobj.batteries")
        name: Name to import (e.g., "ReplDict")

    Returns:
        (imported_item_or_None, is_available)

    Example:
        ReplDict, available = safe_import_from("pysyncobj.batteries", "ReplDict")
    """
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, name), True
    except (ImportError, AttributeError):
        return None, False


# =============================================================================
# Feature Flag Utilities
# =============================================================================

def get_feature_flag(env_var: str, default: bool = False) -> bool:
    """Get boolean feature flag from environment variable.

    Treats "true", "1", "yes", "on" (case-insensitive) as True.

    Args:
        env_var: Environment variable name (e.g., "RINGRIFT_RAFT_ENABLED")
        default: Default value if not set

    Returns:
        Boolean value of the flag
    """
    value = os.environ.get(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_string(env_var: str, default: str = "") -> str:
    """Get string environment variable with default.

    Args:
        env_var: Environment variable name
        default: Default value if not set

    Returns:
        String value
    """
    return os.environ.get(env_var, default)


def get_env_int(env_var: str, default: int) -> int:
    """Get integer environment variable with default.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value
    """
    try:
        return int(os.environ.get(env_var, ""))
    except ValueError:
        return default


def get_env_float(env_var: str, default: float) -> float:
    """Get float environment variable with default.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Float value
    """
    try:
        return float(os.environ.get(env_var, ""))
    except ValueError:
        return default


# =============================================================================
# Protocol Status Tracking
# =============================================================================

@dataclass
class ProtocolAvailability:
    """Tracks availability status for a protocol.

    Attributes:
        name: Protocol name (e.g., "raft", "swim")
        dependency_available: Whether required package is installed
        feature_enabled: Whether feature flag is set
        is_active: Whether protocol is actually being used

    Example:
        raft = ProtocolAvailability(
            name="raft",
            dependency_available=PYSYNCOBJ_AVAILABLE,
            feature_enabled=RAFT_ENABLED,
        )
        if raft.can_use:
            # Initialize Raft
    """

    name: str
    dependency_available: bool = False
    feature_enabled: bool = False
    is_active: bool = False

    @property
    def can_use(self) -> bool:
        """Check if protocol can be used (dep installed + feature enabled)."""
        return self.dependency_available and self.feature_enabled

    @property
    def status_message(self) -> str:
        """Get human-readable status message."""
        if self.is_active:
            return f"{self.name}: active"
        if self.can_use:
            return f"{self.name}: available (not active)"
        if self.dependency_available:
            return f"{self.name}: disabled (feature flag off)"
        return f"{self.name}: unavailable (dependency missing)"


def log_protocol_status(
    name: str,
    dependency_available: bool,
    feature_enabled: bool,
    is_active: bool = False,
) -> None:
    """Log protocol availability status.

    Args:
        name: Protocol name
        dependency_available: Whether dependency is installed
        feature_enabled: Whether feature flag is set
        is_active: Whether protocol is actively being used
    """
    status = ProtocolAvailability(
        name=name,
        dependency_available=dependency_available,
        feature_enabled=feature_enabled,
        is_active=is_active,
    )

    if is_active:
        logger.info(f"[Protocol] {status.status_message}")
    elif status.can_use:
        logger.debug(f"[Protocol] {status.status_message}")
    elif not dependency_available:
        logger.debug(f"[Protocol] {status.status_message}")
    else:
        logger.debug(f"[Protocol] {status.status_message}")


# =============================================================================
# Fallback Handler
# =============================================================================

class FallbackHandler:
    """Handler for graceful protocol fallbacks.

    Provides context for trying a primary protocol with automatic fallback.

    Example:
        async def get_work_item(self):
            with FallbackHandler("raft", "sqlite") as fb:
                if fb.try_primary(self.raft_available):
                    return await self.raft_get_work_item()
                else:
                    return await self.sqlite_get_work_item()
    """

    def __init__(self, primary: str, fallback: str):
        """Initialize fallback handler.

        Args:
            primary: Name of primary protocol
            fallback: Name of fallback protocol
        """
        self.primary = primary
        self.fallback = fallback
        self._used_fallback = False

    def __enter__(self) -> "FallbackHandler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def try_primary(self, available: bool) -> bool:
        """Check if primary protocol should be used.

        Args:
            available: Whether primary protocol is available

        Returns:
            True if should use primary, False if should use fallback
        """
        if available:
            return True
        self._used_fallback = True
        return False

    @property
    def used_fallback(self) -> bool:
        """Check if fallback was used."""
        return self._used_fallback


# =============================================================================
# Constants Loader
# =============================================================================

def load_constants(*names: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load constants from scripts.p2p.constants with fallbacks.

    Args:
        names: Constant names to load
        defaults: Default values for each constant

    Returns:
        Dict mapping constant name to value

    Example:
        consts = load_constants(
            "RAFT_ENABLED", "CONSENSUS_MODE",
            defaults={"RAFT_ENABLED": False, "CONSENSUS_MODE": "bully"}
        )
    """
    defaults = defaults or {}
    result = {}

    try:
        from scripts.p2p import constants as p2p_constants

        for name in names:
            if hasattr(p2p_constants, name):
                result[name] = getattr(p2p_constants, name)
            else:
                result[name] = defaults.get(name)
    except ImportError:
        # Use all defaults
        for name in names:
            result[name] = defaults.get(name)

    return result


__all__ = [
    # Safe imports
    "safe_import",
    "safe_import_from",
    # Feature flags
    "get_feature_flag",
    "get_env_string",
    "get_env_int",
    "get_env_float",
    # Protocol status
    "ProtocolAvailability",
    "log_protocol_status",
    # Fallback handling
    "FallbackHandler",
    # Constants loading
    "load_constants",
]
