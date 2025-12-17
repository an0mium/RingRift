#!/usr/bin/env python3
"""Cross-board transfer learning experiment.

Tests whether pre-training on square8 helps hexagonal model training.
Freezes early convolutional layers and fine-tunes final layers.

Usage:
    python scripts/transfer_learning_experiment.py --source-model models/sq8_2p_best.pth --target-board hexagonal
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType
from app.ai.neural_net import get_policy_size_for_board, get_spatial_size_for_board

logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def load_source_model(model_path: str) -> Dict:
    """Load source model state dict."""
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    else:
        return checkpoint


def get_transferable_layers(state_dict: Dict) -> Dict:
    """Extract layers that can be transferred across board types.

    Early convolutional layers (feature extractors) are typically transferable
    as they learn general patterns. Final layers need reinitialization.
    """
    transferable = {}
    skip_patterns = ["policy", "value_head", "final", "output", "fc"]

    for key, value in state_dict.items():
        # Skip board-specific layers
        should_skip = any(pattern in key.lower() for pattern in skip_patterns)
        if not should_skip:
            transferable[key] = value

    return transferable


def create_transfer_model(
    source_path: str,
    target_board: str,
    freeze_backbone: bool = True,
) -> Dict:
    """Create a model for target board type using source weights.

    Args:
        source_path: Path to source model checkpoint
        target_board: Target board type (hexagonal, square19, etc.)
        freeze_backbone: Whether to freeze transferred layers

    Returns:
        Dict with transfer statistics and recommendations
    """
    source_state = load_source_model(source_path)
    transferable = get_transferable_layers(source_state)

    # Get board-specific dimensions
    board_type_map = {
        "hexagonal": BoardType.HEXAGONAL,
        "square19": BoardType.SQUARE19,
        "square8": BoardType.SQUARE8,
        "hex8": BoardType.HEX8,
    }
    target_board_type = board_type_map.get(target_board.lower())

    if target_board_type:
        target_policy_size = get_policy_size_for_board(target_board_type)
        target_spatial_size = get_spatial_size_for_board(target_board_type)
    else:
        target_policy_size = 0
        target_spatial_size = 0

    total_params = sum(p.numel() for p in source_state.values())
    transferred_params = sum(p.numel() for p in transferable.values())
    transfer_ratio = transferred_params / total_params if total_params > 0 else 0

    result = {
        "source_model": source_path,
        "target_board": target_board,
        "target_policy_size": target_policy_size,
        "target_spatial_size": target_spatial_size,
        "total_source_params": total_params,
        "transferred_params": transferred_params,
        "transfer_ratio": transfer_ratio,
        "transferable_layers": list(transferable.keys()),
        "num_transferable_layers": len(transferable),
        "freeze_backbone": freeze_backbone,
        "recommendation": "",
    }

    # Recommendation based on transfer ratio
    if transfer_ratio > 0.8:
        result["recommendation"] = "High transfer potential - most features are transferable"
    elif transfer_ratio > 0.5:
        result["recommendation"] = "Medium transfer potential - backbone can be reused"
    elif transfer_ratio > 0.2:
        result["recommendation"] = "Low transfer potential - only early layers transferable"
    else:
        result["recommendation"] = "Very low transfer potential - consider training from scratch"

    return result


def run_transfer_experiment(
    source_path: str,
    target_board: str,
    training_db: str,
    output_dir: str,
    epochs: int = 10,
    learning_rate: float = 0.0001,
) -> Dict:
    """Run a transfer learning experiment.

    This is a placeholder for the full experiment. In a real implementation,
    this would:
    1. Create target model architecture
    2. Initialize with transferred weights
    3. Fine-tune on target board data
    4. Evaluate and compare to baseline
    """
    analysis = create_transfer_model(source_path, target_board)

    logger.info(f"Transfer analysis:")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Target: {target_board}")
    logger.info(f"  Transfer ratio: {analysis['transfer_ratio']:.1%}")
    logger.info(f"  Transferable layers: {analysis['num_transferable_layers']}")
    logger.info(f"  Recommendation: {analysis['recommendation']}")

    # Note: Full training would be implemented here
    # For now, just return the analysis
    return {
        "analysis": analysis,
        "experiment_status": "analysis_only",
        "note": "Full training not implemented - analysis shows transfer potential",
    }


def main():
    parser = argparse.ArgumentParser(description="Transfer learning experiment")
    parser.add_argument("--source-model", type=str, required=True,
                       help="Path to source model checkpoint")
    parser.add_argument("--target-board", type=str, default="hexagonal",
                       choices=["hexagonal", "square19", "square8"],
                       help="Target board type")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze transfer potential, don't train")
    parser.add_argument("--training-db", type=str,
                       help="Path to training database")
    parser.add_argument("--output-dir", type=str, default="models/transfer",
                       help="Output directory for transferred model")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.analyze_only:
        result = create_transfer_model(args.source_model, args.target_board)
        print("\n=== Transfer Learning Analysis ===")
        print(f"Source model: {result['source_model']}")
        print(f"Target board: {result['target_board']}")
        print(f"Target policy size: {result['target_policy_size']}")
        print(f"Target spatial size: {result['target_spatial_size']}")
        print(f"Total source params: {result['total_source_params']:,}")
        print(f"Transferred params: {result['transferred_params']:,}")
        print(f"Transfer ratio: {result['transfer_ratio']:.1%}")
        print(f"Transferable layers: {result['num_transferable_layers']}")
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"\nTransferable layer names:")
        for layer in result['transferable_layers'][:10]:
            print(f"  - {layer}")
        if len(result['transferable_layers']) > 10:
            print(f"  ... and {len(result['transferable_layers']) - 10} more")
    else:
        result = run_transfer_experiment(
            source_path=args.source_model,
            target_board=args.target_board,
            training_db=args.training_db or "",
            output_dir=args.output_dir,
        )
        print(f"\nExperiment result: {result['experiment_status']}")
        print(f"Note: {result['note']}")


if __name__ == "__main__":
    main()
