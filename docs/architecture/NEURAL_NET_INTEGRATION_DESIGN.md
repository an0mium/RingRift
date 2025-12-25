# Neural Network Integration Design for Minimax and MCTS

## Overview

This document outlines a research-backed approach to integrating neural network evaluation into the existing Minimax and MCTS AI engines at their higher difficulty levels, while preserving the existing difficulty ladder structure.

## Difficulty Ladder

The difficulty ladder maps Minimax to D3–D4, Descent to D5–D6, MCTS to D7–D8
(heuristic then neural), and Gumbel MCTS to D9–D10.

| Difficulty | AI Type     | Neural | Implementation                            |
| ---------- | ----------- | ------ | ----------------------------------------- |
| 1          | Random      | No     | Pure random move selection                |
| 2          | Heuristic   | No     | Simple heuristic evaluation               |
| 3          | Minimax     | No     | Alpha-beta + PVS + hand-crafted heuristic |
| 4          | Minimax     | Yes    | **NNUE-style neural evaluation**          |
| 5          | Descent     | Yes    | UBFM/Descent search with neural guidance  |
| 6          | Descent     | Yes    | Stronger Descent configuration            |
| 7          | MCTS        | No     | PUCT + heuristic rollouts                 |
| 8          | MCTS        | Yes    | **Neural value/policy guidance**          |
| 9          | Gumbel MCTS | Yes    | Gumbel MCTS + neural policy/value         |
| 10         | Gumbel MCTS | Yes    | Strongest Gumbel MCTS configuration       |

### Key Design Decisions

- **Minimax slots (D3-4)**: D3 uses pure heuristic evaluation, D4 adds NNUE neural evaluation
- **Descent slots (D5-6)**: Always use neural policy+value (AlphaZero-style)
- **MCTS slots (D7-8)**: D7 uses heuristic rollouts only, D8 adds neural value/policy
- **Gumbel MCTS slots (D9-10)**: Always use neural policy+value

---

## Part 1: Minimax with Neural Evaluation (Difficulty 4)

### Approach: NNUE-Inspired Architecture

**Reference**: Efficiently Updatable Neural Networks (NNUE), used in Stockfish NNUE since 2020.

#### Key Design Principles

1. **Incremental Updates**: Unlike full neural network inference, NNUE maintains an accumulator that can be updated incrementally when pieces move, avoiding full re-computation.

2. **Small Network**: NNUE uses relatively small networks (typically 2-3 layers, ~10M parameters) that run on CPU without GPU requirement.

3. **Feature Engineering**: Input features are sparse and position-aware, enabling efficient incremental updates.

#### Proposed RingRift NNUE Architecture

```python
class RingRiftNNUE(nn.Module):
    """NNUE-style evaluation network for RingRift minimax search.

    Architecture inspired by Stockfish NNUE with RingRift-specific features.
    Designed for efficient incremental updates during make/unmake search.
    """

    def __init__(self, board_type: BoardType):
        super().__init__()

        # Feature dimensions per board type
        self.feature_dims = {
            BoardType.SQUARE8: 8 * 8 * 12,    # 768 features
            BoardType.SQUARE19: 19 * 19 * 12, # 4332 features
            BoardType.HEXAGONAL: 217 * 12,    # 2604 features (hex grid)
        }

        input_dim = self.feature_dims[board_type]

        # Half-King-Piece-Square style accumulator
        # Separate accumulator per player perspective
        self.accumulator = nn.Linear(input_dim, 256, bias=True)

        # ClippedReLU activation (0-127 range like Stockfish NNUE)
        # Small hidden layers for fast inference
        self.hidden1 = nn.Linear(512, 32)  # Concatenated accumulators
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, input_dim] sparse input
        acc = torch.clamp(self.accumulator(features), 0, 1)

        # Concatenate perspective accumulators
        x = torch.cat([acc, acc.flip(-1)], dim=-1)  # Simplified

        x = torch.clamp(self.hidden1(x), 0, 1)
        x = torch.clamp(self.hidden2(x), 0, 1)
        return torch.tanh(self.output(x))  # Score in [-1, 1]
```

#### Input Features (12 planes per position)

1. **Ring presence** (4 planes): One-hot for each player's ring at position
2. **Stack presence** (4 planes): One-hot for each player's stack at position
3. **Territory ownership** (4 planes): One-hot for each player's claimed territory

#### Integration into MinimaxAI

```python
class MinimaxAI(HeuristicAI):
    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)

        # Enable NNUE at difficulty >= 4 when neural net is available
        self.use_nnue = (
            config.difficulty >= 4 and
            getattr(config, 'use_neural_net', True) and
            self._load_nnue_model()
        )

    def _evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate using NNUE when available, fallback to heuristic."""
        if state.is_game_over():
            return self._terminal_score(state)

        if self.use_nnue and self.nnue_model is not None:
            return self._evaluate_nnue(state)

        # Fallback to hand-crafted heuristic
        immutable = state.to_immutable()
        return self.evaluate_position(immutable)

    def _evaluate_nnue(self, state: MutableGameState) -> float:
        """NNUE evaluation with incremental update support."""
        features = self._extract_nnue_features(state)
        with torch.no_grad():
            score = self.nnue_model(features).item()

        # Scale from [-1, 1] to centipawn-like score
        return score * 10000.0
```

#### Training Strategy

1. **Supervised Learning from Self-Play**:
   - Use existing self-play games from SQLite databases
   - Label positions with game outcomes (win/loss/draw)
   - Train to predict game result from position

2. **Distillation from Gumbel MCTS**:
   - Use Gumbel MCTS (D9-10) evaluations as teacher
   - Train NNUE to match Gumbel value predictions
   - More sample-efficient than outcome-based training

3. **Incremental Weight Updates**:
   - Track which positions caused largest evaluation errors
   - Prioritize training on problematic positions

---

## Part 2: MCTS with Neural Value Network (Difficulty 8+)

### Approach: Neural Leaf Evaluation + Policy Prior

**Reference**: AlphaZero (Silver et al., 2018), MuZero (Schrittwieser et al., 2020)

#### Key Design Principles

1. **Value Network for Leaf Evaluation**: Replace heuristic rollouts with neural value prediction at search frontier.

2. **Policy Network for Move Ordering**: Use neural policy priors to guide UCT exploration, dramatically improving search efficiency.

3. **Hybrid with Temperature**: Blend neural and heuristic evaluations based on network confidence.

#### Proposed Neural MCTS Architecture

The existing MCTS infrastructure already has `NeuralNetAI` integration points. The enhancement focuses on:

1. **Difficulty-Gated Neural Usage**:

   ```python
   # In MCTSAI.__init__
   self.use_neural_evaluation = (
       config.difficulty >= 6 and
       getattr(config, 'use_neural_net', True)
   )
   ```

2. **PUCT with Neural Policy Prior**:

   ```python
   def puct_value(child: MCTSNode) -> float:
       # Q-value from MCTS statistics
       q_value = child.wins / max(child.visits, 1)

       # Neural policy prior P(s, a)
       if self.use_neural_evaluation and child.move in self.policy_cache:
           prior = self.policy_cache[child.move]
       else:
           prior = 1.0 / len(parent.children)  # Uniform fallback

       # PUCT formula (AlphaZero style)
       c_puct = 1.5  # Exploration constant
       u_value = c_puct * prior * math.sqrt(parent.visits) / (1 + child.visits)

       return q_value + u_value
   ```

3. **Neural Leaf Evaluation**:

   ```python
   def _evaluate_leaf(self, state: GameState) -> float:
       """Evaluate leaf node using neural value network."""
       if not self.use_neural_evaluation or self.neural_net is None:
           # Fallback: heuristic rollout
           return self._heuristic_rollout(state)

       # Neural value prediction
       value = self.neural_net.evaluate_position(state)

       # Optional: blend with heuristic for robustness at lower neural tiers
       if self.config.difficulty == 6:
           # D6: 70% neural, 30% heuristic blend
           heuristic_value = self.evaluate_position(state) / 10000.0
           value = 0.7 * value + 0.3 * heuristic_value

       return value
   ```

#### Network Architecture (Shared with Descent)

The existing `NeuralNetAI` architecture can be reused:

```python
# From neural_net.py - shared policy/value network
class RingRiftNet(nn.Module):
    def __init__(self, in_channels: int, board_size: int):
        # Residual tower (6-10 blocks)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(8)
        ])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, policy_size)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
```

#### Training Strategy

1. **Self-Play with MCTS Targets**:
   - Generate games using D8 MCTS
   - Extract visit count distributions as policy targets
   - Use game outcomes as value targets

2. **Progressive Training**:
   - Start with heuristic-only MCTS (current D8)
   - Generate training data
   - Train neural network
   - Replace heuristic with neural evaluation
   - Repeat with stronger network

---

## Part 3: Implementation Plan

### Phase 1: NNUE for Minimax (D4)

**Files to Modify**:

- `ai-service/app/ai/minimax_ai.py` - Add NNUE evaluation path
- `ai-service/app/ai/nnue.py` (new) - NNUE model and feature extraction
- `ai-service/app/config/ladder_config.py` - D4 tier to use neural minimax (already configured)

**Implementation Steps**:

1. Create `nnue.py` with model architecture and feature extraction
2. Add `_evaluate_nnue()` method to MinimaxAI
3. Gate neural usage on `config.difficulty >= 4 and config.use_neural_net`
4. Create training script using existing self-play databases
5. Add model checkpoint loading infrastructure
6. LadderTierConfig for D4 already has `use_neural_net=True`

### Phase 2: Neural MCTS (D8+)

**Files to Modify**:

- `ai-service/app/ai/mcts_ai.py` - Add neural leaf evaluation
- `ai-service/app/config/ladder_config.py` - D8+ tiers already have neural config

**Implementation Steps**:

1. Add `use_neural_evaluation` flag gated on difficulty >= 8
2. Modify `_evaluate_leaf()` to use neural value when available
3. Modify PUCT calculation to use neural policy priors
4. Create training pipeline for MCTS policy targets
5. LadderTierConfig for D8-10 already has `use_neural_net=True`

### Phase 3: Training Pipeline

**New Files**:

- `ai-service/scripts/train_nnue.py` - NNUE training from self-play
- `ai-service/scripts/train_mcts_policy.py` - Policy/value training
- `ai-service/app/training/nnue_dataset.py` - NNUE training dataset

**Training Data Sources**:

1. Existing self-play databases (SQLite)
2. Game outcomes for value targets
3. MCTS visit distributions for policy targets
4. Descent AI evaluations for distillation

---

## Part 4: Research References

### NNUE (Minimax Enhancement)

1. **Stockfish NNUE** (2020): Efficiently Updatable Neural Networks
   - Paper: "NNUE-Derived Evaluation in Shogi" (Nasu, 2018)
   - Key insight: Incremental accumulator updates enable neural eval in alpha-beta

2. **Leela Chess Zero LC0** (2018): Neural network chess engine
   - Uses full network evaluation but with efficient batching

### AlphaZero-style MCTS

3. **AlphaZero** (Silver et al., 2018): Mastering Chess and Shogi
   - PUCT formula: `Q + c_puct * P * sqrt(N) / (1 + n)`
   - Policy network guides exploration
   - Value network replaces rollouts

4. **MuZero** (Schrittwieser et al., 2020): Learning without game rules
   - Learned dynamics model for planning
   - Applicable when rules are complex

5. **KataGo** (Wu, 2020): State-of-the-art Go AI
   - Auxiliary policy targets improve training
   - Ownership estimation heads

### Simple AlphaZero Variants

6. **"A Simple Alpha(Go) Zero Tutorial"** (arXiv:2008.01188):
   - Simplified implementation reference
   - Descent AI already uses this approach

---

## Part 5: Configuration Updates

### AIConfig Extensions

```python
@dataclass
class AIConfig:
    difficulty: int = Field(ge=1, le=10)

    # Existing fields...
    use_neural_net: Optional[bool] = Field(
        None,
        description=(
            "When explicitly set to False, disables neural-network-backed "
            "evaluation. When True or None, neural nets remain enabled "
            "subject to model availability and difficulty thresholds."
        )
    )

    # NEW: Fine-grained neural control
    nnue_model_path: Optional[str] = Field(
        None,
        description="Path to NNUE model checkpoint for Minimax D4+"
    )

    neural_blend_ratio: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Blend ratio between neural and heuristic eval (1.0 = pure neural)"
    )
```

### LadderTierConfig Updates

```python
# D4 – mid minimax with NNUE on square8, 2-player.
(4, BoardType.SQUARE8, 2): LadderTierConfig(
    difficulty=4,
    board_type=BoardType.SQUARE8,
    num_players=2,
    ai_type=AIType.MINIMAX,
    model_id="nnue_square8_2p",
    heuristic_profile_id="heuristic_v1_sq8_2p",
    randomness=0.08,
    think_time_ms=2800,
    use_neural_net=True,  # NEW
    notes="Mid square8 2p tier with NNUE neural evaluation.",
),

# D8 – strong MCTS with neural guidance on square8, 2-player.
(8, BoardType.SQUARE8, 2): LadderTierConfig(
    difficulty=8,
    board_type=BoardType.SQUARE8,
    num_players=2,
    ai_type=AIType.MCTS,
    model_id="ringrift_best_sq8_2p",
    heuristic_profile_id="heuristic_v1_sq8_2p",
    randomness=0.0,
    think_time_ms=9600,
    use_neural_net=True,  # NEW
    notes="Strong square8 2p tier with neural guidance.",
),
```

---

## Summary

| Component   | Difficulty      | Approach                     | Key Benefit                                |
| ----------- | --------------- | ---------------------------- | ------------------------------------------ |
| Minimax     | D3 (non-neural) | Pure heuristic evaluation    | Fast, predictable baseline                 |
| Minimax     | D4 (neural)     | NNUE-style neural eval       | Fast CPU inference, incremental updates    |
| Descent     | D5-6 (neural)   | Neural guidance              | Strong search with moderate compute        |
| MCTS        | D7 (heuristic)  | Heuristic rollouts           | Tree search without neural dependency      |
| MCTS        | D8 (neural)     | Neural value + policy priors | Better search guidance, no rollouts needed |
| Gumbel MCTS | D9-10           | Neural + Gumbel selection    | Strongest search, highest compute          |

This design targets a clear progression from heuristic search to neural-guided search.
Refer to `ai-service/app/config/ladder_config.py` for the current ladder mapping.

- Graceful degradation when neural nets are unavailable
- Smooth difficulty curve with neural capabilities at appropriate tiers
