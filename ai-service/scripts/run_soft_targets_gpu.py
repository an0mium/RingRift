#!/usr/bin/env python3
"""Generate soft target data using GPU MCTS for comparison experiment."""

from app.training.gpu_mcts_selfplay import GPUMCTSSelfplayConfig, GPUMCTSSelfplayRunner
import numpy as np
import time
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}")
sys.stdout.flush()

config = GPUMCTSSelfplayConfig(
    board_type="hex8",
    num_players=2,
    simulation_budget=8,   # Fast: 8 sims per move
    batch_size=16,         # 16 parallel games
    encoder_version="v3",
    device=device
)

runner = GPUMCTSSelfplayRunner(config)
all_games = []
start = time.time()

# Run batches until we have ~3000 samples
target_samples = 3000
total_samples = 0

batch = 0
while total_samples < target_samples:
    batch += 1
    batch_start = time.time()
    games = runner.run_batch(num_games=16)
    all_games.extend(games)

    batch_samples = sum(len(g.samples) for g in games)
    total_samples += batch_samples
    elapsed = time.time() - batch_start
    print(f"Batch {batch}: {batch_samples} samples in {elapsed:.1f}s (total: {total_samples})")
    sys.stdout.flush()

total_time = time.time() - start
print(f"\nTotal: {len(all_games)} games, {total_samples} samples in {total_time:.1f}s")

# Collect all samples
completed = [g for g in all_games if g.termination_reason == "normal"]
all_samples = []
for g in completed:
    all_samples.extend(g.samples)

print(f"Completed games: {len(completed)}/{len(all_games)}")
print(f"Total samples: {len(all_samples)}")

if not all_samples:
    print("ERROR: No samples collected!")
    sys.exit(1)

# Export to NPZ
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

np.savez_compressed("data/training/soft_targets_hex8_2p.npz",
    features=features,
    globals=globals_arr,
    values=values,
    policy_indices=policy_indices,
    policy_values=policy_values,
    board_type="hex8",
    encoder_version="v3",
    source="gpu_mcts_soft"
)

print(f"\nSaved: {features.shape}")
print(f"Avg actions/sample: {np.mean([np.sum(pv > 0) for pv in policy_values]):.1f}")

# Calculate entropy
entropies = []
for pv in policy_values:
    pv_valid = pv[pv > 0]
    if len(pv_valid) > 0:
        pv_norm = pv_valid / pv_valid.sum()
        entropy = -np.sum(pv_norm * np.log(pv_norm + 1e-10))
        entropies.append(entropy)
print(f"Avg policy entropy: {np.mean(entropies):.3f}")
print("Done!")
