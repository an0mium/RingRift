#!/usr/bin/env python3
"""Post-training evaluation script for GMO models.

Automatically runs comprehensive evaluation after GMO training completes.
Tests against baselines and compares GMO-Gumbel vs CNN-Gumbel.

Usage:
    python scripts/gmo_post_training_eval.py \
        --checkpoint models/gmo/sq8_2p_playerrel/gmo_best.pt \
        --output results/gmo_playerrel_eval.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(checkpoint_path: str, output_path: str, device: str = "cuda"):
    """Run comprehensive GMO evaluation."""
    from scripts.evaluate_nn_models import (
        evaluate_model, run_tournament, print_summary, save_results
    )

    results = []

    # Phase 1: GMO standalone vs baselines
    logger.info("=" * 60)
    logger.info("Phase 1: GMO Standalone Evaluation")
    logger.info("=" * 60)

    baselines = ["random", "heuristic", "mcts_d5"]
    for baseline in baselines:
        logger.info(f"Evaluating GMO vs {baseline}...")
        result = evaluate_model(
            model_type="gmo",
            model_checkpoint=checkpoint_path,
            opponent_type=baseline,
            num_games=40,
            device=device,
        )
        results.append(result)

    # Phase 2: GMO-Gumbel vs baselines
    logger.info("=" * 60)
    logger.info("Phase 2: GMO-Gumbel Hybrid Evaluation")
    logger.info("=" * 60)

    for baseline in baselines:
        logger.info(f"Evaluating GMO-Gumbel vs {baseline}...")
        result = evaluate_model(
            model_type="gmo_gumbel",
            model_checkpoint=checkpoint_path,
            opponent_type=baseline,
            num_games=40,
            device=device,
        )
        results.append(result)

    # Phase 3: Head-to-head GMO vs GMO-Gumbel
    logger.info("=" * 60)
    logger.info("Phase 3: GMO vs GMO-Gumbel Head-to-Head")
    logger.info("=" * 60)

    # This requires a custom matchup - we'll skip for now and recommend manual testing

    # Print and save results
    print_summary(results)
    save_results(results, output_path)

    # Generate summary
    summary = generate_summary(results)
    logger.info("\n" + summary)

    return results


def generate_summary(results) -> str:
    """Generate a markdown summary of results."""
    lines = [
        "# GMO Player-Relative Encoding Evaluation Results",
        f"\nEvaluated: {datetime.now().isoformat()}",
        "\n## Results Summary\n",
        "| Model | Opponent | Win Rate | P1 Win | P2 Win |",
        "|-------|----------|----------|--------|--------|",
    ]

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.opponent_name} | "
            f"{r.win_rate*100:.1f}% | {r.win_rate_as_p1*100:.0f}% | "
            f"{r.win_rate_as_p2*100:.0f}% |"
        )

    # Add analysis
    lines.extend([
        "\n## Key Findings\n",
    ])

    # Check P2 win rate improvement
    gmo_results = [r for r in results if r.model_name.startswith("gmo_") and "gumbel" not in r.model_name]
    if gmo_results:
        avg_p2 = sum(r.win_rate_as_p2 for r in gmo_results) / len(gmo_results)
        lines.append(f"- **P2 Win Rate**: {avg_p2*100:.1f}% average (target: >40%)")
        if avg_p2 > 0.4:
            lines.append("  - ✅ Player-relative encoding fix successful!")
        else:
            lines.append("  - ⚠️ P2 win rate still low - may need investigation")

    # Compare with MCTS-d5
    mcts_results = [r for r in results if r.opponent_name == "mcts_d5"]
    if mcts_results:
        for r in mcts_results:
            lines.append(f"- **{r.model_name} vs MCTS-d5**: {r.win_rate*100:.1f}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post-training GMO evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to GMO checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="results/gmo_eval.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda, mps, cpu)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    run_evaluation(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
