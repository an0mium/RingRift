"""Test that data freshness check is mandatory by default.

This test verifies that the freshness check is enabled by default and fails
when data is stale (Phase 1.5 of the improvement plan).
"""

import unittest
from unittest.mock import MagicMock, patch
from app.coordination.training_freshness import FreshnessResult


class TestFreshnessCheckMandatory(unittest.TestCase):
    """Test mandatory freshness check behavior."""

    @patch('app.training.train.check_freshness_sync')
    @patch('app.training.train.HAS_FRESHNESS_CHECK', True)
    def test_default_behavior_checks_freshness(self, mock_check):
        """Test that freshness check runs by default."""
        from app.training.train import train_model
        from app.training.config import TrainConfig

        # Mock fresh data result
        mock_check.return_value = FreshnessResult(
            success=True,
            is_fresh=True,
            data_age_hours=0.5,
            games_available=1000,
        )

        config = TrainConfig()

        # Note: This will still fail due to missing data file, but we're testing
        # that the freshness check is called with default parameters
        try:
            train_model(
                config=config,
                data_path='/tmp/nonexistent.npz',
                save_path='/tmp/model.pth',
                # Default: skip_freshness_check=False (check is enabled)
                # Default: max_data_age_hours=1.0
                # Default: allow_stale_data=False (fail on stale)
            )
        except Exception:
            pass  # Expected to fail due to missing data

        # Verify freshness check was called
        mock_check.assert_called_once()
        call_kwargs = mock_check.call_args[1]
        self.assertEqual(call_kwargs['max_age_hours'], 1.0)

    @patch('app.training.train.check_freshness_sync')
    @patch('app.training.train.HAS_FRESHNESS_CHECK', True)
    def test_stale_data_fails_by_default(self, mock_check):
        """Test that stale data causes training to fail by default."""
        from app.training.train import train_model
        from app.training.config import TrainConfig

        # Mock stale data result
        mock_check.return_value = FreshnessResult(
            success=True,
            is_fresh=False,  # Data is stale
            data_age_hours=2.5,  # Older than 1.0 hour threshold
            games_available=1000,
        )

        config = TrainConfig()

        # Training should fail with ValueError when data is stale
        with self.assertRaises(ValueError) as cm:
            train_model(
                config=config,
                data_path='/tmp/nonexistent.npz',
                save_path='/tmp/model.pth',
                # Default: allow_stale_data=False (fail on stale)
            )

        # Verify error message mentions stale data
        error_msg = str(cm.exception)
        self.assertIn('TRAINING BLOCKED', error_msg)
        self.assertIn('STALE', error_msg)

    @patch('app.training.train.check_freshness_sync')
    @patch('app.training.train.HAS_FRESHNESS_CHECK', True)
    def test_allow_stale_data_flag_warns_instead_of_fails(self, mock_check):
        """Test that --allow-stale-data warns instead of failing."""
        from app.training.train import train_model
        from app.training.config import TrainConfig

        # Mock stale data result
        mock_check.return_value = FreshnessResult(
            success=True,
            is_fresh=False,  # Data is stale
            data_age_hours=2.5,
            games_available=1000,
        )

        config = TrainConfig()

        # With allow_stale_data=True, should warn but not fail
        try:
            train_model(
                config=config,
                data_path='/tmp/nonexistent.npz',
                save_path='/tmp/model.pth',
                allow_stale_data=True,  # Override: allow stale data
            )
        except ValueError as e:
            # Should NOT fail with stale data error
            if 'TRAINING BLOCKED' in str(e) and 'STALE' in str(e):
                self.fail('Should not fail on stale data when allow_stale_data=True')
        except Exception:
            pass  # Other errors are expected (missing data file)

    @patch('app.training.train.check_freshness_sync')
    @patch('app.training.train.HAS_FRESHNESS_CHECK', True)
    def test_skip_freshness_check_bypasses_check(self, mock_check):
        """Test that --skip-freshness-check bypasses the check entirely."""
        from app.training.train import train_model
        from app.training.config import TrainConfig

        config = TrainConfig()

        try:
            train_model(
                config=config,
                data_path='/tmp/nonexistent.npz',
                save_path='/tmp/model.pth',
                skip_freshness_check=True,  # Skip check entirely
            )
        except Exception:
            pass  # Expected to fail due to missing data

        # Verify freshness check was NOT called
        mock_check.assert_not_called()

    def test_cli_defaults(self):
        """Test that CLI argument defaults are correct."""
        from app.training.train_cli import parse_args

        args = parse_args([])

        # Verify defaults
        self.assertFalse(args.skip_freshness_check,
                        'Freshness check should be enabled by default')
        self.assertEqual(args.max_data_age_hours, 1.0,
                        'Default threshold should be 1.0 hours')
        self.assertFalse(args.allow_stale_data,
                        'Should fail on stale data by default')

    def test_freshness_result_structure(self):
        """Test FreshnessResult dataclass structure."""
        result = FreshnessResult(
            success=True,
            is_fresh=False,
            data_age_hours=2.5,
            games_available=1000,
        )

        self.assertTrue(result.success)
        self.assertFalse(result.is_fresh)
        self.assertEqual(result.data_age_hours, 2.5)
        self.assertEqual(result.games_available, 1000)
        self.assertFalse(result.sync_triggered)
        self.assertFalse(result.sync_completed)


if __name__ == '__main__':
    unittest.main()
