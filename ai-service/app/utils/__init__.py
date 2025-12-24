"""Utility modules for RingRift AI.

This package provides reusable utilities harvested from archived debug scripts
and consolidated for maintainability.

Modules:
    debug_utils: State comparison and parity debugging utilities
    torch_utils: Safe PyTorch operations including device detection (canonical)
    env_config: Typed environment variable access
    game_discovery: Unified game database discovery across all storage patterns

Device Management (Canonical Exports):
    get_device: Auto-detect best compute device (CUDA/MPS/CPU)
    get_device_info: Get detailed device information

Environment Configuration (Canonical Exports):
    env: Singleton EnvConfig instance for typed env var access
    get_str, get_int, get_float, get_bool, get_list: Direct env var getters

Game Discovery (Canonical Exports):
    GameDiscovery: Find all game databases across cluster storage patterns
    find_all_game_databases: Quick function to find all databases
    count_games_for_config: Count games for a board/player configuration
    get_game_counts_summary: Get summary of all game counts
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

# Canonical game discovery exports
from app.utils.game_discovery import (
    GameDiscovery,
    count_games_for_config,
    find_all_game_databases,
    get_game_counts_summary,
)

__all__ = [
    "debug_utils",
    "env_config",
    "torch_utils",
    "game_discovery",
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
    # Game discovery
    "GameDiscovery",
    "find_all_game_databases",
    "count_games_for_config",
    "get_game_counts_summary",
]
