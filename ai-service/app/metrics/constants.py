"""Metric constants for RingRift AI service.

This module provides standardized histogram bucket definitions to ensure
consistency across metrics collection.

Usage:
    from app.metrics.constants import (
        DURATION_BUCKETS_SECONDS,
        TRAINING_DURATION_BUCKETS,
        LATENCY_BUCKETS_SECONDS,
    )

    histogram = Histogram(
        'my_metric',
        'Description',
        buckets=DURATION_BUCKETS_SECONDS,
    )
"""

from __future__ import annotations

# =============================================================================
# Duration Buckets (seconds)
# =============================================================================

# Standard duration buckets for long-running operations (training epochs, selfplay)
# Covers 10 seconds to 1 hour
DURATION_BUCKETS_SECONDS: tuple[float, ...] = (
    10, 30, 60, 120, 300, 600, 1200, 1800, 3600
)

# Training-specific duration buckets (same as standard for now)
TRAINING_DURATION_BUCKETS: tuple[float, ...] = DURATION_BUCKETS_SECONDS

# Extended duration buckets for very long operations (up to 4 hours)
EXTENDED_DURATION_BUCKETS: tuple[float, ...] = (
    300, 600, 1200, 1800, 3600, 7200, 14400
)

# Short duration buckets for quick operations (5 seconds to 10 minutes)
SHORT_DURATION_BUCKETS: tuple[float, ...] = (
    5, 10, 30, 60, 120, 300, 600
)

# =============================================================================
# Latency Buckets (seconds)
# =============================================================================

# API and inference latency buckets (50ms to 10 seconds)
LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0
)

# Fine-grained latency buckets for fast operations (1ms to 1 second)
FINE_LATENCY_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0
)

# Decision-making latency (10ms to 10 seconds)
DECISION_LATENCY_BUCKETS: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)

# =============================================================================
# Count/Quantity Buckets
# =============================================================================

# Game move count buckets
MOVE_COUNT_BUCKETS: tuple[int, ...] = (
    10, 50, 100, 200, 500, 800, 1000, 1500, 2000
)

# Iteration/batch count buckets
ITERATION_COUNT_BUCKETS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 8, 10, 15, 20
)

# Node/simulation count buckets for MCTS
SIMULATION_COUNT_BUCKETS: tuple[int, ...] = (
    50, 100, 200, 400, 800, 1600, 3200, 6400
)

# =============================================================================
# Resource Buckets
# =============================================================================

# Memory usage buckets (MB)
MEMORY_MB_BUCKETS: tuple[int, ...] = (
    100, 256, 512, 1024, 2048, 4096, 8192, 16384
)

# Batch size buckets
BATCH_SIZE_BUCKETS: tuple[int, ...] = (
    1, 4, 8, 16, 32, 64, 128, 256, 512
)
