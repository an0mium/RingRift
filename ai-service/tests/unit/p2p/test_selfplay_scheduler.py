"""Unit tests for SelfplayScheduler (P2P selfplay configuration selection).

December 29, 2025: Initial test coverage for P2P managers.
"""

import time
from unittest import TestCase, mock
from scripts.p2p.managers.selfplay_scheduler import (
    DiversityMetrics,
    SelfplayScheduler,
)


class TestDiversityMetrics(TestCase):
    """Tests for DiversityMetrics dataclass."""

    def test_default_values(self):
        """Test default empty state."""
        metrics = DiversityMetrics()
        self.assertEqual(metrics.games_by_engine_mode, {})
        self.assertEqual(metrics.games_by_board_config, {})
        self.assertEqual(metrics.games_by_difficulty, {})
        self.assertEqual(metrics.asymmetric_games, 0)
        self.assertEqual(metrics.symmetric_games, 0)

    def test_to_dict_empty(self):
        """Test to_dict with empty metrics."""
        metrics = DiversityMetrics()
        d = metrics.to_dict()
        self.assertEqual(d["asymmetric_ratio"], 0.0)
        self.assertEqual(d["engine_mode_distribution"], {})
        self.assertIn("uptime_seconds", d)

    def test_to_dict_with_data(self):
        """Test to_dict with populated metrics."""
        metrics = DiversityMetrics(
            games_by_engine_mode={"gumbel-mcts": 50, "heuristic": 50},
            games_by_board_config={"hex8_2p": 60, "square8_2p": 40},
            asymmetric_games=30,
            symmetric_games=70,
        )
        d = metrics.to_dict()
        
        # Check asymmetric ratio
        self.assertAlmostEqual(d["asymmetric_ratio"], 0.3, places=5)
        
        # Check engine mode distribution
        self.assertAlmostEqual(d["engine_mode_distribution"]["gumbel-mcts"], 0.5)
        self.assertAlmostEqual(d["engine_mode_distribution"]["heuristic"], 0.5)

    def test_uptime_seconds(self):
        """Test uptime tracking."""
        before = time.time()
        metrics = DiversityMetrics()
        time.sleep(0.1)  # Small delay
        d = metrics.to_dict()
        after = time.time()
        
        # Uptime should be between 0 and the elapsed time
        self.assertGreaterEqual(d["uptime_seconds"], 0.1)
        self.assertLessEqual(d["uptime_seconds"], after - before + 0.1)


class TestSelfplaySchedulerConstants(TestCase):
    """Tests for SelfplayScheduler class constants."""

    def test_gpu_required_engine_modes(self):
        """Test GPU-required engine modes are defined correctly."""
        self.assertIn("gumbel-mcts", SelfplayScheduler.GPU_REQUIRED_ENGINE_MODES)
        self.assertIn("mcts", SelfplayScheduler.GPU_REQUIRED_ENGINE_MODES)
        self.assertIn("policy-only", SelfplayScheduler.GPU_REQUIRED_ENGINE_MODES)
        # Heuristic should NOT be in GPU-required
        self.assertNotIn("heuristic", SelfplayScheduler.GPU_REQUIRED_ENGINE_MODES)

    def test_cpu_compatible_engine_modes(self):
        """Test CPU-compatible engine modes are defined correctly."""
        self.assertIn("heuristic", SelfplayScheduler.CPU_COMPATIBLE_ENGINE_MODES)
        self.assertIn("random", SelfplayScheduler.CPU_COMPATIBLE_ENGINE_MODES)
        self.assertIn("brs", SelfplayScheduler.CPU_COMPATIBLE_ENGINE_MODES)
        # Gumbel-mcts should NOT be in CPU-compatible
        self.assertNotIn("gumbel-mcts", SelfplayScheduler.CPU_COMPATIBLE_ENGINE_MODES)

    def test_large_board_types(self):
        """Test large board types are identified."""
        self.assertIn("square19", SelfplayScheduler.LARGE_BOARD_TYPES)
        self.assertIn("hexagonal", SelfplayScheduler.LARGE_BOARD_TYPES)
        # Small boards should not be included
        self.assertNotIn("hex8", SelfplayScheduler.LARGE_BOARD_TYPES)
        self.assertNotIn("square8", SelfplayScheduler.LARGE_BOARD_TYPES)

    def test_engine_mix_coverage(self):
        """Test large board engine mix adds up to 100%."""
        # GPU variant
        gpu_total = sum(weight for _, weight, _, _ in SelfplayScheduler.LARGE_BOARD_ENGINE_MIX)
        self.assertEqual(gpu_total, 100)
        
        # CPU-only variant
        cpu_total = sum(weight for _, weight, _, _ in SelfplayScheduler.LARGE_BOARD_ENGINE_MIX_CPU)
        self.assertEqual(cpu_total, 100)

    def test_engine_mix_no_gpu_in_cpu_variant(self):
        """Test CPU variant has no GPU-required engines."""
        for engine, weight, gpu_required, args in SelfplayScheduler.LARGE_BOARD_ENGINE_MIX_CPU:
            self.assertFalse(
                gpu_required,
                f"CPU-only mix should not have GPU-required engine: {engine}"
            )


class TestSelfplaySchedulerMixinType(TestCase):
    """Tests for SelfplayScheduler mixin integration."""

    def test_mixin_type_defined(self):
        """Test MIXIN_TYPE is defined for base class compatibility."""
        self.assertEqual(SelfplayScheduler.MIXIN_TYPE, "selfplay_scheduler")


if __name__ == "__main__":
    import unittest
    unittest.main()
