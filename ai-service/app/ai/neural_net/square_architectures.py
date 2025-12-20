"""Square board CNN architectures for RingRift AI.

This module contains the CNN architecture classes for square boards (8x8 and 19x19):
- RingRiftCNN_v2: High-capacity SE architecture (96GB systems)
- RingRiftCNN_v2_Lite: Memory-efficient SE architecture (48GB systems)
- RingRiftCNN_v3: Spatial policy heads with SE backbone
- RingRiftCNN_v3_Lite: Memory-efficient spatial policy heads
- RingRiftCNN_v4: NAS-discovered attention architecture

Migrated from _neural_net_legacy.py as part of Phase 2 modularization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .blocks import SEResidualBlock
from .constants import (
    POLICY_SIZE,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
)


class RingRiftCNN_v2(nn.Module):
    """
    High-capacity CNN for 19x19 square boards (96GB memory target).

    This architecture is designed for maximum playing strength on systems
    with sufficient memory (96GB+) to run two instances simultaneously
    for comparison matches with MCTS search overhead.

    Key improvements over RingRiftCNN_MPS:
    - 12 SE residual blocks with Squeeze-and-Excitation for global patterns
    - 192 filters for richer representations
    - 14 base input channels capturing stack/cap/territory mechanics
    - 20 global features for multi-player state tracking
    - Multi-player value head (outputs per-player win probability)
    - 384-dim policy intermediate for better move discrimination

    Input Feature Channels (14 base × 4 frames = 56 total):
        1-4: Per-player stack presence (binary, one per player)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized 0-1)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12-14: Territory ownership channels

    Global Features (20):
        1-4: Rings in hand (per player)
        5-8: Eliminated rings (per player)
        9-12: Territory count (per player)
        13-16: Line count (per player)
        17: Current player indicator
        18: Game phase (early/mid/late)
        19: Total rings in play
        20: LPS threat indicator

    Memory profile (FP32):
    - Model weights: ~150 MB
    - Per-model with activations: ~350 MB
    - Two models + MCTS: ~18 GB total

    Architecture Version:
        v2.0.0 - High-capacity SE architecture for 96GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        history_length: int = 3,
        policy_size: int | None = None,
        policy_intermediate: int = 384,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        # Input channels = base_channels * (history_length + 1)
        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks for global pattern recognition
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (outputs per-player win probability)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head with larger intermediate
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))  # [-1, 1] per player

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference.

        Args:
            feature: Board features [C, H, W]
            globals_vec: Global features [G]
            player_idx: Which player's value to return (default 0)

        Returns:
            Tuple of (value for player, policy logits)
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


class RingRiftCNN_v2_Lite(nn.Module):
    """
    Memory-efficient CNN for 19x19 square boards (48GB memory target).

    This architecture is designed for systems with limited memory (48GB)
    while maintaining reasonable playing strength. Suitable for running
    two instances simultaneously for comparison matches.

    Key trade-offs vs RingRiftCNN_v2:
    - 6 SE residual blocks (vs 12) - faster but shallower
    - 96 filters (vs 192) - smaller representations
    - 192-dim policy intermediate (vs 384)
    - 12 base input channels (vs 14) - reduced history
    - 3 history frames (vs 4) - reduced temporal context

    Input Feature Channels (12 base × 3 frames = 36 total):
        1-4: Per-player stack presence (binary)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12: Current player territory

    Global Features (20):
        Same as RingRiftCNN_v2 for compatibility

    Memory profile (FP32):
    - Model weights: ~60 MB
    - Per-model with activations: ~130 MB
    - Two models + MCTS: ~8 GB total

    Architecture Version:
        v2.0.0-lite - Memory-efficient SE architecture for 48GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0-lite"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 12,
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        history_length: int = 2,
        policy_size: int | None = None,
        policy_intermediate: int = 192,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


# V3 and V4 architectures will be added here as the migration continues
# For now, they remain in _neural_net_legacy.py and are re-exported via __init__.py
