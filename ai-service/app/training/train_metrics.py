"""Training Metrics Module.

Centralizes all Prometheus metrics used by the training pipeline.
Extracted from train.py (December 2025).

Usage:
    from app.training.train_metrics import (
        TRAINING_EPOCHS,
        TRAINING_LOSS,
        TRAINING_SAMPLES,
        HAS_PROMETHEUS,
    )

    if HAS_PROMETHEUS and TRAINING_LOSS:
        TRAINING_LOSS.labels(config="hex8_2p", loss_type="policy").set(0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram

    HAS_PROMETHEUS = True

    # December 2025: Use centralized metric registry
    from app.metrics.constants import TRAINING_DURATION_BUCKETS
    from app.metrics.registry import safe_metric as _safe_metric

    # Core training metrics
    TRAINING_EPOCHS: Counter | None = _safe_metric(
        Counter,
        "ringrift_training_epochs_total",
        "Total training epochs completed",
        labelnames=["config"],
    )
    TRAINING_LOSS: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_training_loss",
        "Current training loss",
        labelnames=["config", "loss_type"],
    )
    TRAINING_SAMPLES: Counter | None = _safe_metric(
        Counter,
        "ringrift_training_samples_total",
        "Total samples processed",
        labelnames=["config"],
    )
    TRAINING_DURATION: Histogram | None = _safe_metric(
        Histogram,
        "ringrift_training_epoch_duration_seconds",
        "Training epoch duration",
        labelnames=["config"],
        buckets=TRAINING_DURATION_BUCKETS,
    )

    # Calibration metrics
    CALIBRATION_ECE: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_calibration_ece",
        "Expected Calibration Error",
        labelnames=["config"],
    )
    CALIBRATION_MCE: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_calibration_mce",
        "Maximum Calibration Error",
        labelnames=["config"],
    )

    # Batch metrics
    BATCH_SIZE: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_training_batch_size",
        "Current training batch size",
        labelnames=["config"],
    )

    # Fault tolerance metrics (2025-12)
    CIRCUIT_BREAKER_STATE: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        labelnames=["config", "operation"],
    )
    ANOMALY_DETECTIONS: Counter | None = _safe_metric(
        Counter,
        "ringrift_training_anomalies_total",
        "Total training anomalies detected",
        labelnames=["config", "type"],
    )
    GRADIENT_CLIP_NORM: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_gradient_clip_norm",
        "Current gradient clipping threshold",
        labelnames=["config"],
    )
    GRADIENT_NORM: Gauge | None = _safe_metric(
        Gauge,
        "ringrift_gradient_norm",
        "Recent gradient norm",
        labelnames=["config"],
    )

except ImportError:
    HAS_PROMETHEUS = False
    TRAINING_EPOCHS = None
    TRAINING_LOSS = None
    TRAINING_SAMPLES = None
    TRAINING_DURATION = None
    CALIBRATION_ECE = None
    CALIBRATION_MCE = None
    BATCH_SIZE = None
    CIRCUIT_BREAKER_STATE = None
    ANOMALY_DETECTIONS = None
    GRADIENT_CLIP_NORM = None
    GRADIENT_NORM = None

# Optional: Dashboard metrics collector for persistent storage (2025-12)
try:
    from app.monitoring.training_dashboard import MetricsCollector

    HAS_METRICS_COLLECTOR = True
except ImportError:
    HAS_METRICS_COLLECTOR = False
    MetricsCollector = None  # type: ignore

__all__ = [
    # Feature flags
    "HAS_PROMETHEUS",
    "HAS_METRICS_COLLECTOR",
    # Core metrics
    "TRAINING_EPOCHS",
    "TRAINING_LOSS",
    "TRAINING_SAMPLES",
    "TRAINING_DURATION",
    # Calibration
    "CALIBRATION_ECE",
    "CALIBRATION_MCE",
    # Batch
    "BATCH_SIZE",
    # Fault tolerance
    "CIRCUIT_BREAKER_STATE",
    "ANOMALY_DETECTIONS",
    "GRADIENT_CLIP_NORM",
    "GRADIENT_NORM",
    # Dashboard
    "MetricsCollector",
]
