"""Unified metrics system for RingRift AI service.

This package consolidates all metrics collection for the AI service,
including:
- Prometheus metrics for monitoring and alerting
- Training metrics for experiment tracking
- Orchestrator metrics for pipeline observability

Usage:
    from app.metrics import (
        # Prometheus metrics
        record_game_outcome,
        record_selfplay_batch,
        record_training_run,
        record_model_promotion,
        # Metrics server
        start_metrics_server,
        # Training logger
        create_training_logger,
    )

    # Record selfplay progress
    record_selfplay_batch(
        board_type="square8",
        num_players=2,
        games=100,
        duration_seconds=60.5,
    )

    # Start metrics server for Prometheus scraping
    start_metrics_server(port=9090)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Re-export from existing metrics module
# Note: Import from parent module using absolute path
import sys
import importlib.util

# Load app/metrics.py as a separate module to avoid circular import
_metrics_path = __file__.replace("__init__.py", "").rstrip("/").rsplit("/", 1)[0] + "/metrics.py"
_spec = importlib.util.spec_from_file_location("_app_metrics_base", _metrics_path)
_metrics_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics_base)

# Re-export from the base metrics module
from _app_metrics_base import (
    # Core metrics
    AI_MOVE_REQUESTS,
    AI_MOVE_LATENCY,
    AI_INSTANCE_CACHE_LOOKUPS,
    AI_INSTANCE_CACHE_SIZE,
    PYTHON_INVARIANT_VIOLATIONS,
    # Game metrics
    GAME_OUTCOMES,
    GAMES_COMPLETED,
    GAMES_MOVES_TOTAL,
    GAME_DURATION_SECONDS,
    WIN_RATE_BY_PLAYER,
    DRAW_RATE,
    # Cluster metrics
    CLUSTER_NODE_UP,
    CLUSTER_NODE_COST_PER_HOUR,
    CLUSTER_GPU_UTILIZATION,
    CLUSTER_CPU_UTILIZATION,
    CLUSTER_GPU_MEMORY_USED_BYTES,
    CLUSTER_MEMORY_USED_BYTES,
    GPU_HOURLY_RATES,
    # Training data metrics
    TRAINING_SAMPLES_BY_PHASE,
    TRAINING_SAMPLES_BY_MOVE_NUMBER,
    TRAINING_DATA_RECENCY,
    TRAINING_UNIQUE_POSITIONS,
    TRAINING_POSITION_ENTROPY,
    # Helper functions
    observe_ai_move_start,
    record_game_outcome,
    record_training_sample,
    report_cluster_node,
)

# Import orchestrator metrics
from app.metrics.orchestrator import (
    # Selfplay metrics
    SELFPLAY_GAMES_TOTAL,
    SELFPLAY_GAMES_PER_SECOND,
    SELFPLAY_BATCH_DURATION,
    SELFPLAY_ERRORS_TOTAL,
    # Training metrics
    TRAINING_RUNS_TOTAL,
    TRAINING_RUN_DURATION,
    TRAINING_LOSS,
    TRAINING_ACCURACY,
    TRAINING_SAMPLES_PROCESSED,
    # Evaluation metrics
    EVALUATION_GAMES_TOTAL,
    EVALUATION_ELO_DELTA,
    # Promotion metrics
    MODEL_PROMOTIONS_TOTAL,
    MODEL_PROMOTION_ELO_GAIN,
    # Pipeline metrics
    PIPELINE_STAGE_DURATION,
    PIPELINE_ITERATIONS_TOTAL,
    PIPELINE_ERRORS_TOTAL,
    # Sync metrics
    DATA_SYNC_DURATION,
    DATA_SYNC_GAMES,
    MODEL_SYNC_DURATION,
    # Helper functions
    record_selfplay_batch,
    record_training_run,
    record_evaluation,
    record_model_promotion,
    record_pipeline_stage,
    record_data_sync,
    record_model_sync,
)

# Re-export training logger
from app.training.metrics_logger import (
    MetricsLogger,
    MetricsBackend,
    TensorBoardBackend,
    WandBBackend,
    ConsoleBackend,
    JSONFileBackend,
    create_training_logger,
)

# Metrics server management
_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = 9090) -> bool:
    """Start the Prometheus metrics HTTP server.

    This should be called once at application startup to expose
    metrics for Prometheus scraping.

    Args:
        port: HTTP port for the metrics server

    Returns:
        True if server started, False if already running
    """
    global _server_started

    with _server_lock:
        if _server_started:
            logger.debug(f"Metrics server already running on port {port}")
            return False

        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            _server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False


def is_metrics_server_running() -> bool:
    """Check if the metrics server is running."""
    return _server_started


__all__ = [
    # Core metrics
    "AI_MOVE_REQUESTS",
    "AI_MOVE_LATENCY",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "AI_INSTANCE_CACHE_SIZE",
    "PYTHON_INVARIANT_VIOLATIONS",
    # Game metrics
    "GAME_OUTCOMES",
    "GAMES_COMPLETED",
    "GAMES_MOVES_TOTAL",
    "GAME_DURATION_SECONDS",
    "WIN_RATE_BY_PLAYER",
    "DRAW_RATE",
    # Cluster metrics
    "CLUSTER_NODE_UP",
    "CLUSTER_NODE_COST_PER_HOUR",
    "CLUSTER_GPU_UTILIZATION",
    "CLUSTER_CPU_UTILIZATION",
    "CLUSTER_GPU_MEMORY_USED_BYTES",
    "CLUSTER_MEMORY_USED_BYTES",
    "GPU_HOURLY_RATES",
    # Training data metrics
    "TRAINING_SAMPLES_BY_PHASE",
    "TRAINING_SAMPLES_BY_MOVE_NUMBER",
    "TRAINING_DATA_RECENCY",
    "TRAINING_UNIQUE_POSITIONS",
    "TRAINING_POSITION_ENTROPY",
    # Orchestrator metrics
    "SELFPLAY_GAMES_TOTAL",
    "SELFPLAY_GAMES_PER_SECOND",
    "SELFPLAY_BATCH_DURATION",
    "SELFPLAY_ERRORS_TOTAL",
    "TRAINING_RUNS_TOTAL",
    "TRAINING_RUN_DURATION",
    "TRAINING_LOSS",
    "TRAINING_ACCURACY",
    "TRAINING_SAMPLES_PROCESSED",
    "EVALUATION_GAMES_TOTAL",
    "EVALUATION_ELO_DELTA",
    "MODEL_PROMOTIONS_TOTAL",
    "MODEL_PROMOTION_ELO_GAIN",
    "PIPELINE_STAGE_DURATION",
    "PIPELINE_ITERATIONS_TOTAL",
    "PIPELINE_ERRORS_TOTAL",
    "DATA_SYNC_DURATION",
    "DATA_SYNC_GAMES",
    "MODEL_SYNC_DURATION",
    # Helper functions
    "observe_ai_move_start",
    "record_game_outcome",
    "record_training_sample",
    "report_cluster_node",
    "record_selfplay_batch",
    "record_training_run",
    "record_evaluation",
    "record_model_promotion",
    "record_pipeline_stage",
    "record_data_sync",
    "record_model_sync",
    # Training logger
    "MetricsLogger",
    "MetricsBackend",
    "TensorBoardBackend",
    "WandBBackend",
    "ConsoleBackend",
    "JSONFileBackend",
    "create_training_logger",
    # Server
    "start_metrics_server",
    "is_metrics_server_running",
]
