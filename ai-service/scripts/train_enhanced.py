#!/usr/bin/env python3
"""Enhanced training script with all performance improvements enabled.

This script runs training with optimized settings designed to maximize model strength:

Key Enhancements:
1. Higher learning rate (0.01 vs 0.003 default) with cosine annealing
2. Quality weighting - weights samples by MCTS visit counts
3. Hard example mining - focuses on difficult positions
4. Outcome-weighted policy - winner's moves get higher weight
5. Warmup epochs for stable training start

Expected Elo Improvement: +50-100 Elo over default training settings

Usage:
    python scripts/train_enhanced.py --board-type hex8 --num-players 2

    # With specific data path:
    python scripts/train_enhanced.py --board-type hex8 --num-players 2 \
        --data-path data/training/hex8_2p.npz

    # Quick test run:
    python scripts/train_enhanced.py --board-type hex8 --num-players 2 --epochs 5

December 2025: Created to address Elo ceiling observations.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Enhanced training parameters (vs defaults)
ENHANCED_PARAMS = {
    # Learning rate: 10x higher than default 0.001
    # AlphaZero used 0.01 with cosine decay
    'learning_rate': 0.01,

    # Warmup prevents instability at high LR
    'warmup_epochs': 3,

    # Cosine annealing for smooth LR decay
    'lr_scheduler': 'cosine',
    'lr_min': 1e-5,

    # Quality weighting (+5-15 Elo expected)
    # Weights samples by MCTS visit counts
    'enable_quality_weighting': True,
    'quality_weight_blend': 0.5,
    'quality_ranking_weight': 0.1,

    # Hard example mining (+10-20 Elo expected)
    # Prioritizes positions where model struggles
    'enable_hard_example_mining': True,
    'hard_example_top_k': 0.3,

    # Outcome-weighted policy (+5-10 Elo expected)
    # Winners' moves get higher weight
    'enable_outcome_weighted_policy': True,
    'outcome_weight_scale': 0.5,

    # Standard training settings
    'batch_size': 512,
    'epochs': 50,

    # Early stopping patience
    'early_stopping_patience': 10,

    # Mixed precision for speed
    'mixed_precision': True,
    'amp_dtype': 'bfloat16',
}


def find_best_data_path(board_type: str, num_players: int) -> str | None:
    """Find the best available training data for the config."""
    data_dir = Path('data/training')
    config_key = f"{board_type}_{num_players}p"

    # Try various naming patterns
    patterns = [
        f"{config_key}_consolidated.npz",
        f"{config_key}.npz",
        f"canonical_{config_key}.npz",
        f"{config_key}_selfplay*.npz",
    ]

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            # Return most recent
            best = max(matches, key=lambda p: p.stat().st_mtime)
            return str(best)

    return None


def build_train_command(args: argparse.Namespace) -> list[str]:
    """Build the training command with enhanced parameters."""
    cmd = [
        sys.executable, '-m', 'app.training.train',
        '--board-type', args.board_type,
        '--num-players', str(args.num_players),
    ]

    # Data path
    data_path = args.data_path
    if not data_path:
        data_path = find_best_data_path(args.board_type, args.num_players)
        if not data_path:
            logger.error(
                f"No training data found for {args.board_type}_{args.num_players}p. "
                "Please provide --data-path or run export first."
            )
            sys.exit(1)
    cmd.extend(['--data-path', data_path])

    # Core training params
    cmd.extend(['--learning-rate', str(args.learning_rate)])
    cmd.extend(['--batch-size', str(args.batch_size)])
    cmd.extend(['--epochs', str(args.epochs)])

    # LR scheduling
    cmd.extend(['--warmup-epochs', str(args.warmup_epochs)])
    cmd.extend(['--lr-scheduler', args.lr_scheduler])
    cmd.extend(['--lr-min', str(args.lr_min)])

    # Quality weighting
    if args.enable_quality_weighting:
        cmd.append('--enable-quality-weighting')
        cmd.extend(['--quality-weight-blend', str(args.quality_weight_blend)])
        cmd.extend(['--quality-ranking-weight', str(args.quality_ranking_weight)])

    # Hard example mining
    if args.enable_hard_example_mining:
        cmd.append('--enable-hard-example-mining')
        cmd.extend(['--hard-example-top-k', str(args.hard_example_top_k)])

    # Outcome-weighted policy
    if args.enable_outcome_weighted_policy:
        cmd.append('--enable-outcome-weighted-policy')
        cmd.extend(['--outcome-weight-scale', str(args.outcome_weight_scale)])

    # Early stopping
    cmd.extend(['--early-stopping-patience', str(args.early_stopping_patience)])

    # Mixed precision
    if args.mixed_precision:
        cmd.append('--mixed-precision')
        cmd.extend(['--amp-dtype', args.amp_dtype])

    # Save path
    if args.save_path:
        cmd.extend(['--save-path', args.save_path])
    else:
        config_key = f"{args.board_type}_{args.num_players}p"
        default_save = f"models/enhanced_{config_key}.pth"
        cmd.extend(['--save-path', default_save])

    # Transfer learning
    if args.init_weights:
        cmd.extend(['--init-weights', args.init_weights])

    return cmd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run enhanced training with all optimizations enabled',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/train_enhanced.py --board-type hex8 --num-players 2

    # With transfer learning from existing model
    python scripts/train_enhanced.py --board-type hex8 --num-players 2 \\
        --init-weights models/canonical_hex8_2p.pth

    # Quick test with 5 epochs
    python scripts/train_enhanced.py --board-type hex8 --num-players 2 --epochs 5
        """
    )

    # Required
    parser.add_argument(
        '--board-type', type=str, required=True,
        choices=['hex8', 'square8', 'square19', 'hexagonal'],
        help='Board type to train'
    )
    parser.add_argument(
        '--num-players', type=int, default=2,
        help='Number of players (default: 2)'
    )

    # Data
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to training data (auto-discovered if not provided)'
    )

    # Training params (with enhanced defaults)
    parser.add_argument(
        '--learning-rate', type=float,
        default=ENHANCED_PARAMS['learning_rate'],
        help=f"Learning rate (default: {ENHANCED_PARAMS['learning_rate']})"
    )
    parser.add_argument(
        '--batch-size', type=int,
        default=ENHANCED_PARAMS['batch_size'],
        help=f"Batch size (default: {ENHANCED_PARAMS['batch_size']})"
    )
    parser.add_argument(
        '--epochs', type=int,
        default=ENHANCED_PARAMS['epochs'],
        help=f"Training epochs (default: {ENHANCED_PARAMS['epochs']})"
    )
    parser.add_argument(
        '--warmup-epochs', type=int,
        default=ENHANCED_PARAMS['warmup_epochs'],
        help=f"Warmup epochs (default: {ENHANCED_PARAMS['warmup_epochs']})"
    )
    parser.add_argument(
        '--lr-scheduler', type=str,
        default=ENHANCED_PARAMS['lr_scheduler'],
        help=f"LR scheduler (default: {ENHANCED_PARAMS['lr_scheduler']})"
    )
    parser.add_argument(
        '--lr-min', type=float,
        default=ENHANCED_PARAMS['lr_min'],
        help=f"Minimum LR (default: {ENHANCED_PARAMS['lr_min']})"
    )

    # Enhancement toggles (enabled by default)
    parser.add_argument(
        '--enable-quality-weighting', action='store_true',
        default=ENHANCED_PARAMS['enable_quality_weighting'],
        help='Enable quality-weighted training'
    )
    parser.add_argument(
        '--no-quality-weighting', action='store_true',
        help='Disable quality weighting'
    )
    parser.add_argument(
        '--quality-weight-blend', type=float,
        default=ENHANCED_PARAMS['quality_weight_blend'],
        help='Quality weight blend factor'
    )
    parser.add_argument(
        '--quality-ranking-weight', type=float,
        default=ENHANCED_PARAMS['quality_ranking_weight'],
        help='Quality ranking weight'
    )

    parser.add_argument(
        '--enable-hard-example-mining', action='store_true',
        default=ENHANCED_PARAMS['enable_hard_example_mining'],
        help='Enable hard example mining'
    )
    parser.add_argument(
        '--no-hard-example-mining', action='store_true',
        help='Disable hard example mining'
    )
    parser.add_argument(
        '--hard-example-top-k', type=float,
        default=ENHANCED_PARAMS['hard_example_top_k'],
        help='Hard example fraction'
    )

    parser.add_argument(
        '--enable-outcome-weighted-policy', action='store_true',
        default=ENHANCED_PARAMS['enable_outcome_weighted_policy'],
        help='Enable outcome-weighted policy'
    )
    parser.add_argument(
        '--no-outcome-weighted-policy', action='store_true',
        help='Disable outcome weighting'
    )
    parser.add_argument(
        '--outcome-weight-scale', type=float,
        default=ENHANCED_PARAMS['outcome_weight_scale'],
        help='Outcome weight scale'
    )

    parser.add_argument(
        '--early-stopping-patience', type=int,
        default=ENHANCED_PARAMS['early_stopping_patience'],
        help='Early stopping patience'
    )

    # Mixed precision
    parser.add_argument(
        '--mixed-precision', action='store_true',
        default=ENHANCED_PARAMS['mixed_precision'],
        help='Enable mixed precision training'
    )
    parser.add_argument(
        '--no-mixed-precision', action='store_true',
        help='Disable mixed precision'
    )
    parser.add_argument(
        '--amp-dtype', type=str,
        default=ENHANCED_PARAMS['amp_dtype'],
        help='AMP dtype (bfloat16 or float16)'
    )

    # Output
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--init-weights', type=str, default=None,
        help='Path to initial weights (transfer learning)'
    )

    # Execution
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print command without executing'
    )

    args = parser.parse_args()

    # Handle --no-* flags
    if args.no_quality_weighting:
        args.enable_quality_weighting = False
    if args.no_hard_example_mining:
        args.enable_hard_example_mining = False
    if args.no_outcome_weighted_policy:
        args.enable_outcome_weighted_policy = False
    if args.no_mixed_precision:
        args.mixed_precision = False

    return args


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Enhanced Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Board type: {args.board_type}")
    logger.info(f"Num players: {args.num_players}")
    logger.info(f"Learning rate: {args.learning_rate} (default was 0.001)")
    logger.info(f"LR scheduler: {args.lr_scheduler}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Quality weighting: {args.enable_quality_weighting}")
    logger.info(f"Hard example mining: {args.enable_hard_example_mining}")
    logger.info(f"Outcome-weighted policy: {args.enable_outcome_weighted_policy}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info("=" * 60)

    # Build command
    cmd = build_train_command(args)

    logger.info(f"Training command: {' '.join(cmd)}")

    if args.dry_run:
        logger.info("Dry run - not executing")
        print("\nTo run training, execute:")
        print(' '.join(cmd))
        return 0

    # Execute training
    logger.info("Starting enhanced training...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        logger.info("Training completed successfully!")
    else:
        logger.error(f"Training failed with exit code {result.returncode}")

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
