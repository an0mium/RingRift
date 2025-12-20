#!/usr/bin/env python
"""Diagnose EBMO model behavior.

Checks:
1. Energy value distributions
2. Whether optimization is working
3. Move selection quality
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import EBMONetwork, EBMOConfig, ActionFeatureExtractor
from app.ai.factory import AIFactory
from app.models.core import AIType, AIConfig, BoardType
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state


def diagnose_model(model_path: str):
    """Run diagnostics on EBMO model."""
    print(f"\n{'='*60}")
    print(f"EBMO Model Diagnostics: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print("Checkpoint keys:", list(checkpoint.keys()))

    if 'config' in checkpoint:
        config_data = checkpoint['config']
        print(f"Model config: {config_data}")
        # Handle dict vs EBMOConfig object
        if isinstance(config_data, dict):
            config = EBMOConfig(**{k: v for k, v in config_data.items() if hasattr(EBMOConfig, k) or k in EBMOConfig.__dataclass_fields__})
        else:
            config = config_data
    else:
        config = EBMOConfig()

    model = EBMONetwork(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Check for NaN/Inf in weights
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  WARNING: NaN in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  WARNING: Inf in {name}")
            has_inf = True

    if not has_nan and not has_inf:
        print("  All weights are finite (no NaN/Inf)")

    # Check weight statistics
    print("\nWeight statistics:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}, "
                  f"min={param.min().item():.4f}, max={param.max().item():.4f}")

    # Create a test game state
    print("\n" + "="*60)
    print("Testing on sample game state")
    print("="*60 + "\n")

    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    # Create EBMO AI
    ebmo = EBMO_AI(
        player_number=1,
        config=AIConfig(difficulty=5),
        model_path=model_path
    )

    # Get valid moves
    valid_moves = ebmo.get_valid_moves(state)
    print(f"Valid moves at start: {len(valid_moves)}")

    # Encode state
    with torch.no_grad():
        state_embed = ebmo.network.encode_state_from_game(state, 1, ebmo.device)
        print(f"\nState embedding shape: {state_embed.shape}")
        print(f"State embedding stats: mean={state_embed.mean():.4f}, std={state_embed.std():.4f}")

        # Encode all legal moves
        legal_embeddings = ebmo._encode_legal_moves(valid_moves)
        print(f"\nMove embeddings shape: {legal_embeddings.shape}")
        print(f"Move embedding stats: mean={legal_embeddings.mean():.4f}, std={legal_embeddings.std():.4f}")

        # Compute energies for all moves
        state_batch = state_embed.unsqueeze(0).expand(len(valid_moves), -1)
        energies = ebmo.network.compute_energy(state_batch, legal_embeddings)

        print(f"\nEnergy distribution for {len(valid_moves)} moves:")
        print(f"  min: {energies.min().item():.4f}")
        print(f"  max: {energies.max().item():.4f}")
        print(f"  mean: {energies.mean().item():.4f}")
        print(f"  std: {energies.std().item():.4f}")
        print(f"  range: {(energies.max() - energies.min()).item():.4f}")

        # Check if energies are all the same (degenerate)
        if energies.std().item() < 0.01:
            print("\n  WARNING: Energy values are nearly identical!")
            print("  This means the model can't distinguish between moves!")

        # Show top and bottom moves by energy
        sorted_indices = energies.argsort()
        print("\nTop 5 lowest energy (should be best) moves:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i].item()
            move = valid_moves[idx]
            print(f"  {i+1}. {move.type.value} energy={energies[idx].item():.4f}")

        print("\nTop 5 highest energy (should be worst) moves:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[-(i+1)].item()
            move = valid_moves[idx]
            print(f"  {i+1}. {move.type.value} energy={energies[idx].item():.4f}")

    # Test optimization on one move
    print("\n" + "="*60)
    print("Testing gradient descent optimization")
    print("="*60 + "\n")

    init_move = valid_moves[0]
    feature_extractor = ActionFeatureExtractor(8)
    init_features = feature_extractor.extract_tensor([init_move], ebmo.device)

    with torch.no_grad():
        action_embed = ebmo.network.encode_action(init_features).squeeze(0)
        init_energy = ebmo.network.compute_energy(
            state_embed.unsqueeze(0),
            action_embed.unsqueeze(0)
        ).item()

    print(f"Initial move: {init_move.type.value}")
    print(f"Initial energy: {init_energy:.4f}")

    # Run optimization
    action_embed = action_embed.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([action_embed], lr=0.1)

    energies_during_opt = []
    for step in range(100):
        optimizer.zero_grad()
        energy = ebmo.network.compute_energy(
            state_embed.unsqueeze(0),
            action_embed.unsqueeze(0)
        )
        energies_during_opt.append(energy.item())
        energy.backward()
        optimizer.step()

    print(f"\nOptimization over 100 steps:")
    print(f"  Start energy: {energies_during_opt[0]:.4f}")
    print(f"  End energy: {energies_during_opt[-1]:.4f}")
    print(f"  Change: {energies_during_opt[-1] - energies_during_opt[0]:.4f}")

    if abs(energies_during_opt[-1] - energies_during_opt[0]) < 0.01:
        print("\n  WARNING: Optimization didn't change energy much!")
        print("  Gradients may be too small or model may be degenerate.")

    # Check gradients
    with torch.enable_grad():
        action_test = legal_embeddings[0].clone().requires_grad_(True)
        energy = ebmo.network.compute_energy(
            state_embed.unsqueeze(0),
            action_test.unsqueeze(0)
        )
        energy.backward()
        grad_norm = action_test.grad.norm().item()
        print(f"\nGradient norm at a legal move: {grad_norm:.6f}")

        if grad_norm < 1e-6:
            print("  WARNING: Gradients are vanishing!")

    # Play a few moves and see what happens
    print("\n" + "="*60)
    print("Testing actual move selection")
    print("="*60 + "\n")

    # Compare with random and heuristic
    random_ai = AIFactory.create(AIType.RANDOM, 2, AIConfig(difficulty=1))
    heuristic_ai = AIFactory.create(AIType.HEURISTIC, 2, AIConfig(difficulty=5))

    # Play 3 moves with EBMO and show what it picks
    test_state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    for move_num in range(6):
        current_player = test_state.current_player
        if current_player == 1:
            ai = ebmo
            ai_name = "EBMO"
        else:
            ai = heuristic_ai
            ai_name = "Heuristic"

        move = ai.select_move(test_state)
        if move is None:
            break

        print(f"Move {move_num + 1} (Player {current_player} - {ai_name}): {move.type.value}")

        test_state = engine.apply_move(test_state, move)

        if test_state.winner is not None:
            print(f"Game ended: Player {test_state.winner} wins")
            break

    print("\n" + "="*60)
    print("Diagnosis complete")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ebmo/ebmo_improved_best.pt")
    args = parser.parse_args()

    diagnose_model(args.model)
