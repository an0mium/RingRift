"""
DEPRECATED: This module is a legacy stub.

The working ReAnalyze implementation is in:
    scripts/reanalyze_replay_dataset.py

It is already integrated into the improvement loop via:
    scripts/run_improvement_loop.py --dataset-policy-target mcts_visits

The default policy target is 'mcts_visits' which uses MCTS visit-count
distributions as soft policy targets. Alternative: 'descent_softmax'.

Usage:
    # Via improvement loop (recommended):
    python scripts/run_improvement_loop.py --board square8 --players 2 \
        --dataset-policy-target mcts_visits

    # Direct reanalysis:
    python scripts/reanalyze_replay_dataset.py \
        --db data/games/selfplay.db \
        --board-type square8 \
        --num-players 2 \
        --output data/training/reanalyzed.npz \
        --policy-target mcts_visits

This stub is kept for backwards compatibility but does nothing useful.
"""

import warnings


def reanalyze_dataset(
    input_file="data/dataset.npz",
    output_file="data/dataset_reanalyzed.npz",
    model_path="ai-service/app/models/ringrift_best.pth",
):
    """DEPRECATED: Use scripts/reanalyze_replay_dataset.py instead."""
    warnings.warn(
        "reanalyze_dataset() is deprecated. "
        "Use scripts/reanalyze_replay_dataset.py or run_improvement_loop.py "
        "with --dataset-policy-target mcts_visits",
        DeprecationWarning,
        stacklevel=2,
    )
    # Suppress unused parameter warnings
    _ = input_file
    _ = output_file
    _ = model_path
    print(
        "DEPRECATED: This function does nothing. "
        "Use scripts/reanalyze_replay_dataset.py instead."
    )


if __name__ == "__main__":
    reanalyze_dataset()
