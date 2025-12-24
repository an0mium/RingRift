#!/usr/bin/env python3
"""Create synthetic soft targets from hard targets for comparison testing.

This script takes hard (one-hot) policy targets and converts them to soft
targets by adding Dirichlet noise, simulating what MCTS visit distributions
might look like.
"""

import argparse
import numpy as np
from pathlib import Path


def convert_hard_to_soft(
    input_path: str,
    output_path: str,
    noise_alpha: float = 0.3,
    noise_fraction: float = 0.5,
    num_neighbors: int = 5,
) -> None:
    """Convert hard policy targets to synthetic soft targets.

    Args:
        input_path: Path to input NPZ with hard targets
        output_path: Path to output NPZ with soft targets
        noise_alpha: Dirichlet alpha parameter (lower = more peaked)
        noise_fraction: Fraction of probability to distribute as noise
        num_neighbors: Number of additional actions to add probability to
    """
    print(f"Loading {input_path}...")
    data = np.load(input_path, allow_pickle=True)

    # Get policy data
    policy_indices = data["policy_indices"]
    policy_values = data["policy_values"]

    n_samples = len(policy_indices)
    print(f"  Samples: {n_samples}")

    # Calculate max policy size from data
    max_policy_size = policy_indices.shape[1] if policy_indices.ndim > 1 else 1

    # Check if already soft
    avg_actions = np.mean([np.sum(pv > 0) for pv in policy_values])
    if avg_actions > 1.5:
        print(f"  Already soft (avg={avg_actions:.1f} actions/sample)")
        # Just copy
        np.savez_compressed(output_path, **{k: data[k] for k in data.files})
        return

    print(f"  Converting hard targets to soft (noise_alpha={noise_alpha})...")

    # Get all unique policy indices to sample neighbors from
    all_indices = np.unique(policy_indices[policy_indices > 0])
    if len(all_indices) == 0:
        all_indices = np.arange(100)  # Fallback

    # Create new soft targets
    new_indices = []
    new_values = []

    for i in range(n_samples):
        orig_idx = policy_indices[i]
        orig_val = policy_values[i]

        # Find the selected action (one-hot)
        selected_idx = orig_idx[orig_val > 0]
        if len(selected_idx) == 0:
            # No action, keep as-is
            new_indices.append(orig_idx)
            new_values.append(orig_val)
            continue

        main_action = selected_idx[0]

        # Sample neighbor actions (different from main)
        candidates = all_indices[all_indices != main_action]
        if len(candidates) < num_neighbors:
            neighbors = candidates
        else:
            neighbors = np.random.choice(candidates, size=num_neighbors, replace=False)

        # Create soft distribution
        all_actions = np.concatenate([[main_action], neighbors])
        n_actions = len(all_actions)

        # Use Dirichlet noise weighted toward main action
        weights = np.zeros(n_actions)
        weights[0] = 1.0 - noise_fraction  # Main action gets most probability
        weights[1:] = noise_fraction / (n_actions - 1)  # Rest distributed

        # Add Dirichlet noise
        noise = np.random.dirichlet([noise_alpha] * n_actions)
        probs = (1 - noise_fraction) * weights + noise_fraction * noise
        probs = probs / probs.sum()  # Normalize

        # Store
        soft_idx = np.zeros(max(max_policy_size, n_actions), dtype=np.int32)
        soft_val = np.zeros(max(max_policy_size, n_actions), dtype=np.float32)
        soft_idx[:n_actions] = all_actions
        soft_val[:n_actions] = probs

        new_indices.append(soft_idx)
        new_values.append(soft_val)

    # Pad to same shape
    max_len = max(len(x) for x in new_indices)
    padded_indices = np.zeros((n_samples, max_len), dtype=np.int32)
    padded_values = np.zeros((n_samples, max_len), dtype=np.float32)

    for i, (idx, val) in enumerate(zip(new_indices, new_values)):
        padded_indices[i, :len(idx)] = idx
        padded_values[i, :len(val)] = val

    # Save with same metadata but updated policies
    save_dict = {}
    for key in data.files:
        if key == "policy_indices":
            save_dict[key] = padded_indices
        elif key == "policy_values":
            save_dict[key] = padded_values
        elif key == "source":
            save_dict[key] = np.asarray("synthetic_soft")
        else:
            save_dict[key] = data[key]

    np.savez_compressed(output_path, **save_dict)

    # Stats
    avg_actions = np.mean([np.sum(pv > 0) for pv in padded_values])
    entropies = []
    for pv in padded_values:
        pv_valid = pv[pv > 0]
        if len(pv_valid) > 0:
            pv_norm = pv_valid / pv_valid.sum()
            entropy = -np.sum(pv_norm * np.log(pv_norm + 1e-10))
            entropies.append(entropy)

    print(f"  Saved: {padded_indices.shape}")
    print(f"  Avg actions/sample: {avg_actions:.1f}")
    print(f"  Avg entropy: {np.mean(entropies):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic soft targets")
    parser.add_argument("input", help="Input NPZ with hard targets")
    parser.add_argument("output", help="Output NPZ with soft targets")
    parser.add_argument("--noise-alpha", type=float, default=0.3, help="Dirichlet alpha")
    parser.add_argument("--noise-fraction", type=float, default=0.25, help="Noise fraction")
    parser.add_argument("--num-neighbors", type=int, default=4, help="Number of neighbor actions")

    args = parser.parse_args()
    convert_hard_to_soft(
        args.input,
        args.output,
        noise_alpha=args.noise_alpha,
        noise_fraction=args.noise_fraction,
        num_neighbors=args.num_neighbors,
    )


if __name__ == "__main__":
    main()
