#!/usr/bin/env python3
"""Graph Neural Network Policy/Value Network for RingRift.

This module implements a GNN-based architecture that naturally handles:
- Hexagonal and square board geometries without masking hacks
- Territory connectivity through message passing
- Variable board sizes (train small, play large)

Architecture:
- Input: Graph from board_to_graph() or board_to_graph_hex()
- Encoder: GraphSAGE or GAT layers for message passing
- Global pooling: Attention-weighted graph readout
- Heads: Policy (per-node + global) and Value

Based on research showing GNNs achieve +413 Elo in territory control games
and better generalization than CNNs for board game AI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric availability
try:
    from torch_geometric.nn import (
        SAGEConv,
        GATConv,
        global_mean_pool,
        global_add_pool,
    )
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning(
        "PyTorch Geometric not installed. Install with: "
        "pip install torch-geometric torch-scatter torch-sparse"
    )


class GraphAttentionPooling(nn.Module):
    """Attention-based graph pooling for global readout."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Compute attention-weighted global pooling.

        Args:
            x: Node features (num_nodes, hidden_dim)
            batch: Batch assignment (num_nodes,) or None for single graph

        Returns:
            Graph-level features (batch_size, hidden_dim)
        """
        # Compute attention scores
        attn_scores = self.attention(x)  # (num_nodes, 1)

        if batch is None:
            # Single graph
            attn_weights = F.softmax(attn_scores, dim=0)
            return (x * attn_weights).sum(dim=0, keepdim=True)

        # Batched graphs - softmax within each graph
        from torch_geometric.utils import softmax
        attn_weights = softmax(attn_scores.squeeze(-1), batch)

        # Weighted sum per graph
        weighted = x * attn_weights.unsqueeze(-1)
        return global_add_pool(weighted, batch)


class GNNPolicyNet(nn.Module):
    """Graph Neural Network for policy and value prediction.

    This architecture is designed for territory control games:
    - Message passing captures connectivity patterns
    - Attention pooling focuses on strategically important nodes
    - Per-node policy enables position-aware action prediction

    Args:
        node_feature_dim: Input node feature dimension (default: 32)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of GNN layers (default: 6)
        num_heads: Attention heads for GAT layers (default: 4)
        dropout: Dropout probability (default: 0.1)
        conv_type: 'sage' or 'gat' (default: 'sage')
        action_space_size: Total action space size (default: 6158 for square8)
        num_players: Number of players for value head (default: 4)
    """

    def __init__(
        self,
        node_feature_dim: int = 32,
        edge_attr_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_type: str = "sage",
        action_space_size: int = 6158,
        num_players: int = 4,
        global_feature_dim: int = 20,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric required for GNN. "
                "Install with: pip install torch-geometric"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_space_size = action_space_size
        self.conv_type = conv_type

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Edge feature encoder (optional)
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_dim // 4)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if conv_type == "gat":
                # Graph Attention Network
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            else:
                # GraphSAGE (default - more efficient)
                conv = SAGEConv(hidden_dim, hidden_dim)

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Global feature encoder (game phase, rings, etc.)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # Graph pooling
        self.graph_pool = GraphAttentionPooling(hidden_dim)

        # Policy head
        # Combines per-node features with global context
        self.policy_node_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.policy_global_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_players),
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        globals_: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Node features (num_nodes, node_feature_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_attr_dim) or None
            batch: Batch assignment (num_nodes,) or None for single graph
            globals_: Global features (batch_size, global_feature_dim) or None

        Returns:
            Tuple of:
            - policy_logits: (batch_size, action_space_size)
            - value: (batch_size, num_players)
        """
        # Encode nodes
        h = self.node_encoder(x)

        # Message passing layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            h = h + h_new

        # Global graph representation
        graph_repr = self.graph_pool(h, batch)  # (batch_size, hidden_dim)

        # Add global features if provided
        if globals_ is not None:
            global_enc = self.global_encoder(globals_)
            graph_repr = torch.cat([graph_repr, global_enc], dim=-1)
        else:
            # Pad to expected size
            graph_repr = torch.cat([
                graph_repr,
                torch.zeros_like(graph_repr)
            ], dim=-1)

        # Policy: combine global context with node-aware projection
        policy_logits = self.policy_head(
            self.policy_global_proj(graph_repr)
        )

        # Value: predict per-player values
        value = self.value_head(graph_repr)
        value = torch.tanh(value)  # Bound to [-1, 1]

        return policy_logits, value

    def forward_from_data(
        self,
        data: "Data",
        globals_: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass from PyG Data object.

        Convenience method for single graph inference.
        """
        return self.forward(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, 'edge_attr', None),
            batch=getattr(data, 'batch', None),
            globals_=globals_,
        )


class GNNPolicyNetLite(nn.Module):
    """Lightweight GNN for fast inference.

    Reduced version for deployment:
    - Fewer layers (4 vs 6)
    - Smaller hidden dim (64 vs 128)
    - No edge attributes
    - Simple mean pooling

    Can be trained via distillation from full GNNPolicyNet.
    """

    def __init__(
        self,
        node_feature_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 4,
        action_space_size: int = 6158,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        h = F.relu(self.node_encoder(x))

        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h  # Residual

        # Simple mean pooling
        if batch is None:
            graph_repr = h.mean(dim=0, keepdim=True)
        else:
            graph_repr = global_mean_pool(h, batch)

        policy = self.policy_head(graph_repr)
        value = torch.tanh(self.value_head(graph_repr))

        return policy, value.squeeze(-1)


def create_gnn_policy(
    board_type: str = "square8",
    variant: str = "full",
    **kwargs,
) -> nn.Module:
    """Factory function to create GNN policy network.

    Args:
        board_type: 'square8', 'square19', 'hex8', or 'hexagonal'
        variant: 'full' or 'lite'
        **kwargs: Additional arguments for the network

    Returns:
        Configured GNN policy network
    """
    # Board-specific configurations
    configs = {
        "square8": {"action_space_size": 6158, "node_feature_dim": 32},
        "square19": {"action_space_size": 67000, "node_feature_dim": 32},
        "hex8": {"action_space_size": 3000, "node_feature_dim": 32},
        "hexagonal": {"action_space_size": 25000, "node_feature_dim": 32},
    }

    config = configs.get(board_type, configs["square8"])
    config.update(kwargs)

    if variant == "lite":
        return GNNPolicyNetLite(**config)
    return GNNPolicyNet(**config)


__all__ = [
    "GNNPolicyNet",
    "GNNPolicyNetLite",
    "GraphAttentionPooling",
    "create_gnn_policy",
    "HAS_PYG",
]
