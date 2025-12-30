"""NNUE (Efficiently Updatable Neural Network) model management.

This package provides:
- registry: Canonical path management for NNUE models
- Core NNUE classes and utilities (from nnue.py module)
"""

import importlib.util
import sys
from pathlib import Path

# Import registry functions
from .registry import (
    CANONICAL_CONFIGS,
    NNUEModelInfo,
    NNUERegistryStats,
    get_nnue_canonical_path,
    get_nnue_config_key,
    get_nnue_model_info,
    get_all_nnue_paths,
    get_existing_nnue_models,
    get_missing_nnue_models,
    get_nnue_registry_stats,
    get_nnue_output_path,
    promote_nnue_model,
    print_nnue_registry_status,
)

# Load nnue.py module (sibling file) using importlib to avoid naming conflict
# The module app.ai.nnue.py is shadowed by this package (app.ai.nnue/)
_nnue_module_path = Path(__file__).parent.parent / "nnue.py"
_spec = importlib.util.spec_from_file_location("_nnue_core", _nnue_module_path)
if _spec is not None and _spec.loader is not None:
    _nnue_core = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_nnue_core)

    # Re-export core NNUE classes and functions
    # Constants
    FEATURE_PLANES = _nnue_core.FEATURE_PLANES
    GLOBAL_FEATURES = _nnue_core.GLOBAL_FEATURES
    FEATURE_DIMS = _nnue_core.FEATURE_DIMS
    FEATURE_PLANES_V2 = _nnue_core.FEATURE_PLANES_V2
    GLOBAL_FEATURES_V2 = _nnue_core.GLOBAL_FEATURES_V2
    FEATURE_DIMS_V2 = _nnue_core.FEATURE_DIMS_V2
    BOARD_SIZES = _nnue_core.BOARD_SIZES
    CURRENT_NNUE_FEATURE_VERSION = _nnue_core.CURRENT_NNUE_FEATURE_VERSION
    NNUE_FEATURE_V1 = _nnue_core.NNUE_FEATURE_V1
    NNUE_FEATURE_V2 = _nnue_core.NNUE_FEATURE_V2
    NNUE_FEATURE_V3 = _nnue_core.NNUE_FEATURE_V3

    # Functions
    clear_nnue_cache = _nnue_core.clear_nnue_cache
    get_feature_dim = _nnue_core.get_feature_dim
    get_feature_dim_for_version = _nnue_core.get_feature_dim_for_version
    detect_feature_version_from_accumulator = _nnue_core.detect_feature_version_from_accumulator
    get_board_size = _nnue_core.get_board_size
    extract_features_from_gamestate = _nnue_core.extract_features_from_gamestate
    extract_features_from_mutable = _nnue_core.extract_features_from_mutable
    extract_features_from_gpu_batch_vectorized = _nnue_core.extract_features_from_gpu_batch_vectorized
    get_nnue_model_path = _nnue_core.get_nnue_model_path
    load_nnue_model = _nnue_core.load_nnue_model

    # Classes
    ClippedReLU = _nnue_core.ClippedReLU
    StochasticDepthLayer = _nnue_core.StochasticDepthLayer
    ResidualBlock = _nnue_core.ResidualBlock
    RingRiftNNUE = _nnue_core.RingRiftNNUE
    MultiPlayerNNUE = _nnue_core.MultiPlayerNNUE
    NNUEEvaluator = _nnue_core.NNUEEvaluator
    BatchNNUEEvaluator = _nnue_core.BatchNNUEEvaluator

    # Internal functions (needed by tests)
    _NNUE_CACHE = _nnue_core._NNUE_CACHE
    _rotate_player_perspective = _nnue_core._rotate_player_perspective
    _migrate_legacy_state_dict = _nnue_core._migrate_legacy_state_dict
else:
    raise ImportError("Failed to load nnue.py module")

__all__ = [
    # Registry exports
    "CANONICAL_CONFIGS",
    "NNUEModelInfo",
    "NNUERegistryStats",
    "get_nnue_canonical_path",
    "get_nnue_config_key",
    "get_nnue_model_info",
    "get_all_nnue_paths",
    "get_existing_nnue_models",
    "get_missing_nnue_models",
    "get_nnue_registry_stats",
    "get_nnue_output_path",
    "promote_nnue_model",
    "print_nnue_registry_status",
    # Core NNUE exports
    "FEATURE_PLANES",
    "GLOBAL_FEATURES",
    "FEATURE_DIMS",
    "FEATURE_PLANES_V2",
    "GLOBAL_FEATURES_V2",
    "FEATURE_DIMS_V2",
    "BOARD_SIZES",
    "CURRENT_NNUE_FEATURE_VERSION",
    "NNUE_FEATURE_V1",
    "NNUE_FEATURE_V2",
    "NNUE_FEATURE_V3",
    "clear_nnue_cache",
    "get_feature_dim",
    "get_feature_dim_for_version",
    "detect_feature_version_from_accumulator",
    "get_board_size",
    "extract_features_from_gamestate",
    "extract_features_from_mutable",
    "extract_features_from_gpu_batch_vectorized",
    "get_nnue_model_path",
    "load_nnue_model",
    "ClippedReLU",
    "StochasticDepthLayer",
    "ResidualBlock",
    "RingRiftNNUE",
    "MultiPlayerNNUE",
    "NNUEEvaluator",
    "BatchNNUEEvaluator",
    "_NNUE_CACHE",
    "_rotate_player_perspective",
    "_migrate_legacy_state_dict",
]
