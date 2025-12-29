"""Unit tests for background model prefetch during training.

December 29, 2025: Tests for the TRAINING_PROGRESS-triggered prefetch
feature in UnifiedDistributionDaemon.
"""

from unittest import TestCase, mock


class TestBackgroundPrefetch(TestCase):
    """Tests for background model prefetch during training."""

    def test_prefetch_enabled_by_default(self):
        """Test prefetch is enabled by default."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()
        self.assertTrue(daemon._prefetch_enabled)
        self.assertEqual(daemon._prefetch_threshold, 0.80)

    def test_prefetch_threshold_configurable(self):
        """Test prefetch threshold can be configured."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()
        daemon._prefetch_threshold = 0.90
        self.assertEqual(daemon._prefetch_threshold, 0.90)

    def test_prefetch_skipped_when_disabled(self):
        """Test no prefetch when disabled."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()
        daemon._prefetch_enabled = False

        # Simulate progress event
        event = {
            "epochs_completed": 90,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        # Should not enqueue when disabled
        self.assertEqual(len(daemon._pending_items), 0)
        self.assertEqual(len(daemon._prefetched_checkpoints), 0)

    def test_prefetch_skipped_below_threshold(self):
        """Test no prefetch when progress below threshold."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        # 50% progress, below 80% threshold
        event = {
            "epochs_completed": 50,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        self.assertEqual(len(daemon._pending_items), 0)
        self.assertEqual(len(daemon._prefetched_checkpoints), 0)

    def test_prefetch_triggered_above_threshold(self):
        """Test prefetch triggered when progress exceeds threshold."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        # 85% progress, above 80% threshold
        event = {
            "epochs_completed": 85,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        # Should enqueue the checkpoint
        self.assertEqual(len(daemon._pending_items), 1)
        self.assertIn("/path/to/checkpoint.pth", daemon._prefetched_checkpoints)

        # Verify item contents
        item = daemon._pending_items[0]
        self.assertEqual(item["path"], "/path/to/checkpoint.pth")
        self.assertEqual(item["config_key"], "hex8_2p")
        self.assertTrue(item["is_prefetch"])
        self.assertAlmostEqual(item["training_progress"], 0.85, places=2)

    def test_prefetch_not_duplicated(self):
        """Test same checkpoint not prefetched twice."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        event = {
            "epochs_completed": 85,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }

        # First call - should enqueue
        daemon._on_training_progress_for_prefetch(event)
        self.assertEqual(len(daemon._pending_items), 1)

        # Second call with same checkpoint - should skip
        daemon._on_training_progress_for_prefetch(event)
        self.assertEqual(len(daemon._pending_items), 1)  # Still 1

    def test_prefetch_different_checkpoints(self):
        """Test different checkpoints can be prefetched."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        # First checkpoint at 85%
        event1 = {
            "epochs_completed": 85,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint_85.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event1)

        # Second checkpoint at 90%
        event2 = {
            "epochs_completed": 90,
            "total_epochs": 100,
            "checkpoint_path": "/path/to/checkpoint_90.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event2)

        # Both should be enqueued
        self.assertEqual(len(daemon._pending_items), 2)
        self.assertEqual(len(daemon._prefetched_checkpoints), 2)

    def test_prefetch_skipped_without_checkpoint_path(self):
        """Test no prefetch when checkpoint path missing."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        event = {
            "epochs_completed": 90,
            "total_epochs": 100,
            # No checkpoint_path
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        self.assertEqual(len(daemon._pending_items), 0)

    def test_prefetch_skipped_without_total_epochs(self):
        """Test no prefetch when total_epochs is 0 or missing."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        event = {
            "epochs_completed": 90,
            "total_epochs": 0,  # Invalid
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        self.assertEqual(len(daemon._pending_items), 0)

    def test_prefetch_uses_best_checkpoint_path_fallback(self):
        """Test best_checkpoint_path used as fallback."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        event = {
            "epochs_completed": 90,
            "total_epochs": 100,
            "best_checkpoint_path": "/path/to/best_checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)

        self.assertEqual(len(daemon._pending_items), 1)
        self.assertEqual(daemon._pending_items[0]["path"], "/path/to/best_checkpoint.pth")

    def test_metrics_include_prefetch_info(self):
        """Test get_metrics includes prefetch information."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        metrics = daemon.get_metrics()
        self.assertIn("prefetch_enabled", metrics)
        self.assertIn("prefetch_threshold", metrics)
        self.assertIn("prefetched_checkpoints_count", metrics)

        self.assertTrue(metrics["prefetch_enabled"])
        self.assertEqual(metrics["prefetch_threshold"], 0.80)
        self.assertEqual(metrics["prefetched_checkpoints_count"], 0)

    def test_prefetch_handles_event_with_payload(self):
        """Test handling events wrapped in payload attribute."""
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()

        # Simulate event object with payload attribute
        class MockEvent:
            payload = {
                "epochs_completed": 85,
                "total_epochs": 100,
                "checkpoint_path": "/path/to/checkpoint.pth",
                "config_key": "hex8_2p",
            }

        daemon._on_training_progress_for_prefetch(MockEvent())

        self.assertEqual(len(daemon._pending_items), 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
