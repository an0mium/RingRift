#!/usr/bin/env python3
"""Fast GPU MCTS soft target generation for comparison experiment."""

from app.training.gpu_mcts_selfplay import GPUMCTSSelfplayConfig, GPUMCTSSelfplayRunner
import numpy as np
import torch
import sys
import time

print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
sys.stdout.flush()

config = GPUMCTSSelfplayConfig(
    board_type="hex8",
    num_players=2,
    simulation_budget=16,  # Fast: 16 instead of 64
    batch_size=8,
    encoder_version="v3",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

runner = GPUMCTSSelfplayRunner(config)
all_games = []
start = time.time()

for batch in range(10):  # 10 batches x 8 games = 80 games
    batch_start = time.time()
    games = runner.run_batch(num_games=8)
    all_games.extend(games)
    elapsed = time.time() - batch_start
    samples = sum(len(g.samples) for g in games)
    print(f"Batch {batch+1}/10: {len(games)} games, {samples} samples ({elapsed:.1f}s)")
    sys.stdout.flush()

total_time = time.time() - start
print(f"\nTotal: {len(all_games)} games in {total_time:.1f}s ({len(all_games)/total_time*60:.1f} games/min)")

completed = [g for g in all_games if g.termination_reason == "normal"]
all_samples = []
for g in completed:
    all_samples.extend(g.samples)

print(f"Completed: {len(completed)}/{len(all_games)} games, {len(all_samples)} samples")

if all_samples:
    features = np.stack([s.features for s in all_samples])
    globals_arr = np.stack([s.globals for s in all_samples])
    values = np.array([s.value for s in all_samples])

    max_actions = max(len(s.policy_indices) for s in all_samples)
    policy_indices = np.zeros((len(all_samples), max_actions), dtype=np.int32)
    policy_values = np.zeros((len(all_samples), max_actions), dtype=np.float32)

    for i, s in enumerate(all_samples):
        n = len(s.policy_indices)
        policy_indices[i, :n] = s.policy_indices
        policy_values[i, :n] = s.policy_values

    np.savez_compressed("data/training/gpu_mcts_soft_hex8_2p.npz",
        features=features, globals=globals_arr, values=values,
        policy_indices=policy_indices, policy_values=policy_values,
        board_type="hex8", encoder_version="v3", source="gpu_mcts_soft"
    )
    print(f"Saved: {features.shape}")
else:
    print("ERROR: No samples collected!")

print("Done!")
