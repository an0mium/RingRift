"""Utility modules for RingRift AI.

This package provides reusable utilities harvested from archived debug scripts
and consolidated for maintainability.

Modules:
    debug_utils: State comparison and parity debugging utilities
    torch_utils: Safe PyTorch operations including device detection (canonical)
    env_config: Typed environment variable access

Device Management (Canonical Exports):
    get_device: Auto-detect best compute device (CUDA/MPS/CPU)
    get_device_info: Get detailed device information

Environment Configuration (Canonical Exports):
    env: Singleton EnvConfig instance for typed env var access
    get_str, get_int, get_float, get_bool, get_list: Direct env var getters
"""

from __future__ import annotations

# Canonical device management exports
from app.utils.torch_utils import get_device, get_device_info

# Canonical environment configuration exports
from app.utils.env_config import (
    EnvConfig,
    env,
    get_bool,
    get_float,
    get_int,
    get_list,
    get_str,
)

__all__ = [
    "debug_utils",
    "env_config",
    "torch_utils",
    # Device management
    "get_device",
    "get_device_info",
    # Environment configuration
    "EnvConfig",
    "env",
    "get_bool",
    "get_float",
    "get_int",
    "get_list",
    "get_str",
]
