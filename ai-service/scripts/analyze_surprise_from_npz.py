#!/usr/bin/env python3
"""
Surprise Metric Analysis from NPZ Training Data

Analyzes policy entropy and surprise distribution from training data to assess
Titans/MIRAS applicability for RingRift.

Key metrics:
- Policy entropy distribution (low entropy = confident, high = uncertain)
- Implied surprise from policy targets
- Variance in policy predictions
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_policy_entropy(policy_probs: np.ndarray) -> float:
    """Compute entropy of a policy distribution."""
    # Avoid log(0)
    probs = np.clip(policy_probs, 1e-10, 1.0)
    # Normalize if needed
    probs = probs / probs.sum()
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def analyze_npz(npz_path: str, sample_size: int = 10000) -> Dict:
    """
    Analyze policy entropy and surprise from NPZ training data.

    Returns:
        Dictionary of analysis results
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    print(f"Keys in NPZ: {list(data.keys())}")

    # Get policy data - could be 'policy', 'policy_target', 'policies', etc.
    policy_key = None
    for key in ['policy', 'policy_target', 'policies', 'policy_targets']:
        if key in data:
            policy_key = key
            break

    if policy_key is None:
        # Check for sparse policy format
        if 'policy_indices' in data:
            print("Found sparse policy format (policy_indices)")
            policy_indices = data['policy_indices']
            total_samples = len(policy_indices)

            # For sparse format, the "chosen action" is deterministic
            # Surprise analysis isn't directly applicable
            # But we can analyze the distribution of chosen actions

            # Sample
            indices = np.random.choice(total_samples, min(sample_size, total_samples), replace=False)
            sampled_indices = policy_indices[indices]

            # Analyze action distribution
            unique_actions, counts = np.unique(sampled_indices.flatten(), return_counts=True)

            results = {
                "npz_path": npz_path,
                "total_samples": int(total_samples),
                "samples_analyzed": len(indices),
                "policy_format": "sparse",
                "unique_actions": int(len(unique_actions)),
                "action_distribution": {
                    "mean_count": float(np.mean(counts)),
                    "std_count": float(np.std(counts)),
                    "max_count": int(np.max(counts)),
                    "min_count": int(np.min(counts)),
                },
                "titans_applicability": {
                    "policy_format_note": "Sparse policy format - each sample has single target action",
                    "entropy_analysis": "Not applicable for sparse format",
                    "recommendation": "Need soft policy targets for full surprise analysis. Consider enabling soft targets in selfplay.",
                }
            }
            return results

        return {"error": f"No policy data found. Keys: {list(data.keys())}"}

    policies = data[policy_key]
    total_samples = len(policies)

    print(f"Total samples: {total_samples}")
    print(f"Policy shape: {policies.shape}")

    # Sample if too many
    if total_samples > sample_size:
        indices = np.random.choice(total_samples, sample_size, replace=False)
        policies = policies[indices]
    else:
        indices = np.arange(total_samples)

    print(f"Analyzing {len(policies)} samples...")

    # Compute entropy for each sample
    entropies = []
    max_probs = []
    effective_actions = []  # Number of actions with prob > threshold

    for i, policy in enumerate(policies):
        # Flatten if needed
        if len(policy.shape) > 1:
            policy = policy.flatten()

        # Skip all-zero policies
        if policy.sum() == 0:
            continue

        # Normalize
        policy = policy / policy.sum()

        # Entropy
        entropy = compute_policy_entropy(policy)
        entropies.append(entropy)

        # Max probability (confidence)
        max_probs.append(np.max(policy))

        # Effective actions (prob > 0.01)
        effective_actions.append(np.sum(policy > 0.01))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(policies)}...")

    entropies = np.array(entropies)
    max_probs = np.array(max_probs)
    effective_actions = np.array(effective_actions)

    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]

    # Implied surprise = entropy (expected surprise)
    # Higher entropy = more uncertain = more potential for surprising moves

    results = {
        "npz_path": npz_path,
        "total_samples": int(total_samples),
        "samples_analyzed": len(entropies),
        "policy_format": "dense",
        "policy_shape": list(policies.shape),
        "entropy_stats": {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
            "percentiles": {str(p): float(np.percentile(entropies, p)) for p in percentiles},
        },
        "confidence_stats": {
            "mean_max_prob": float(np.mean(max_probs)),
            "std_max_prob": float(np.std(max_probs)),
            "high_confidence_pct": float(100 * np.mean(max_probs > 0.5)),
            "low_confidence_pct": float(100 * np.mean(max_probs < 0.1)),
        },
        "effective_actions_stats": {
            "mean": float(np.mean(effective_actions)),
            "std": float(np.std(effective_actions)),
            "max": float(np.max(effective_actions)),
        },
        "titans_applicability": {
            "mean_entropy": float(np.mean(entropies)),
            "high_entropy_pct": float(100 * np.mean(entropies > 2.0)),  # High uncertainty
            "low_confidence_pct": float(100 * np.mean(max_probs < 0.3)),
            "memory_benefit_score": 0.0,
            "recommendation": "",
        }
    }

    # Compute benefit score
    # Higher entropy and lower confidence = more benefit from memory
    high_entropy_pct = results["titans_applicability"]["high_entropy_pct"]
    low_confidence_pct = results["titans_applicability"]["low_confidence_pct"]
    mean_entropy = results["titans_applicability"]["mean_entropy"]

    benefit_score = (
        10 * min(mean_entropy, 3.0) +  # Base from entropy (max 30)
        0.5 * high_entropy_pct +        # High entropy positions (max 50)
        0.2 * low_confidence_pct         # Low confidence positions (max 20)
    )
    benefit_score = min(100, benefit_score)

    results["titans_applicability"]["memory_benefit_score"] = benefit_score

    if benefit_score > 50:
        results["titans_applicability"]["recommendation"] = (
            "HIGH: Many positions have uncertain policies. "
            "Titans memory could help by remembering opponent patterns and adapting."
        )
    elif benefit_score > 25:
        results["titans_applicability"]["recommendation"] = (
            "MEDIUM: Some positions show uncertainty. "
            "Titans could help with opponent modeling in complex positions."
        )
    else:
        results["titans_applicability"]["recommendation"] = (
            "LOW: Policies are generally confident. "
            "Current CNN approach may be sufficient, but memory could still help with adaptation."
        )

    return results


def print_results(results: Dict):
    """Pretty print analysis results."""
    print("\n" + "=" * 60)
    print("SURPRISE/ENTROPY ANALYSIS FOR TITANS APPLICABILITY")
    print("=" * 60)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\nNPZ Path: {results['npz_path']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Samples analyzed: {results['samples_analyzed']}")
    print(f"Policy format: {results['policy_format']}")

    if results['policy_format'] == 'sparse':
        print("\n--- Sparse Policy Analysis ---")
        print(f"  Unique actions: {results['unique_actions']}")
        ad = results['action_distribution']
        print(f"  Action distribution: mean={ad['mean_count']:.1f}, max={ad['max_count']}")
        print("\n--- TITANS APPLICABILITY ---")
        ta = results['titans_applicability']
        print(f"  {ta['policy_format_note']}")
        print(f"  {ta['recommendation']}")
        print("=" * 60)
        return

    print(f"Policy shape: {results['policy_shape']}")

    print("\n--- Entropy Statistics ---")
    stats = results["entropy_stats"]
    print(f"  Mean entropy: {stats['mean']:.3f}")
    print(f"  Std deviation: {stats['std']:.3f}")
    print(f"  Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
    print("\n  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p}th: {v:.3f}")

    print("\n--- Confidence Statistics ---")
    conf = results["confidence_stats"]
    print(f"  Mean max probability: {conf['mean_max_prob']:.3f}")
    print(f"  High confidence (>50%): {conf['high_confidence_pct']:.1f}%")
    print(f"  Low confidence (<10%): {conf['low_confidence_pct']:.1f}%")

    print("\n--- Effective Actions ---")
    ea = results["effective_actions_stats"]
    print(f"  Mean actions with prob > 1%: {ea['mean']:.1f}")
    print(f"  Max effective actions: {ea['max']:.0f}")

    print("\n--- TITANS APPLICABILITY ASSESSMENT ---")
    ta = results["titans_applicability"]
    print(f"  Mean entropy: {ta['mean_entropy']:.3f}")
    print(f"  High entropy positions (>2.0): {ta['high_entropy_pct']:.1f}%")
    print(f"  Low confidence positions (<30%): {ta['low_confidence_pct']:.1f}%")
    print(f"  Memory benefit score: {ta['memory_benefit_score']:.1f}/100")
    print(f"\n  RECOMMENDATION: {ta['recommendation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze surprise/entropy from NPZ training data")
    parser.add_argument("--npz", type=str, default="data/training/hex8_2p_4k.npz",
                        help="Path to NPZ file")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of samples to analyze")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON")

    args = parser.parse_args()

    if not Path(args.npz).exists():
        print(f"Error: NPZ file not found: {args.npz}")
        sys.exit(1)

    results = analyze_npz(args.npz, args.sample_size)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
