# Gradient Move Optimization (GMO) Algorithm

A novel game-playing algorithm that uses gradient ascent in move embedding space
combined with entropy-guided exploration to find optimal moves.

## Key Innovation

Traditional neural network game AI uses forward-pass policy sampling or tree search:

- Policy Network: state -> policy_net -> move_probs -> sample
- MCTS: tree search -> visit counts
- Minimax: enumerate moves -> evaluate leaves -> propagate

GMO innovation: optimize in a continuous move embedding space using gradients,
then project back to legal moves:

state -> embed -> gradient_ascent(value + uncertainty) -> optimized_embed -> nearest_legal_move

## Architecture

```
GameState -> StateEncoder -> state_embed (128-dim)
LegalMoves -> MoveEncoder -> move_embeds (128-dim each)

[state_embed] + [move_embed_i]
    -> GMOValueNetWithUncertainty
    -> (value, log_var)

Gradient dObjective/dMoveEmbed + MC Dropout uncertainty
    -> optimized move embedding
    -> nearest legal move
    -> selected move
```

## Information-Theoretic Objective

```
Objective = E[Value] + beta * sqrt(Variance) + gamma * Novelty
```

Where:

- E[Value]: Expected game outcome (exploitation)
- sqrt(Variance): Epistemic uncertainty via MC Dropout (exploration)
- Novelty: Distance from previously explored embeddings (diversity)

The beta and gamma coefficients control the exploration/exploitation tradeoff.

## Core Components

### 1. StateEncoder

- Input: GameState (board features)
- Output: 128-dimensional state embedding
- Architecture: MLP (768 -> 256 -> 128)
- Features: 12 planes x 64 positions = 768 features

### 2. MoveEncoder

- Input: Move object
- Output: 128-dimensional move embedding
- Components:
  - move_type embedding (8 types)
  - from_position embedding (64 positions)
  - to_position embedding (64 positions)
  - placement_count linear (scalar)
- Projection: concat -> 128-dim

### 3. GMOValueNetWithUncertainty

- Input: concat(state_embed, move_embed) = 256-dim
- Hidden: 256 -> 256 with ReLU + Dropout(0.1)
- Outputs:
  - value: tanh scalar in [-1, 1]
  - log_var: uncertainty log-variance

### 4. NoveltyTracker

- Ring buffer of explored embeddings (size 1000)
- Computes min distance to memory for novelty score
- Encourages exploration of diverse moves

## Algorithm

```python
def select_move(state, legal_moves):
    # 1. Encode state
    state_embed = state_encoder(state)

    # 2. Encode all legal moves
    move_embeds = [move_encoder(m) for m in legal_moves]

    # 3. Initial evaluation with uncertainty
    candidates = []
    for idx, me in enumerate(move_embeds):
        mean, entropy, var = estimate_uncertainty(state_embed, me, value_net)
        novelty = novelty_tracker.compute_novelty(me)
        score = mean + beta * sqrt(var) + gamma * novelty
        candidates.append((idx, score, me))

    # 4. Optimize top-k candidates
    top_k = sorted(candidates, key=score, reverse=True)[:5]

    best_move, best_score = None, -inf
    for idx, _, initial_embed in top_k:
        # Gradient ascent in embedding space
        optimized = optimize_move_with_entropy(
            state_embed, initial_embed, value_net,
            steps=10, lr=0.1, beta=0.3
        )

        # Project to nearest legal move
        similarities = [cosine_sim(optimized, me) for me in move_embeds]
        nearest_idx = argmax(similarities)

        # Evaluate projected move
        final_value = value_net(state_embed, move_embeds[nearest_idx])
        if final_value > best_score:
            best_score = final_value
            best_move = legal_moves[nearest_idx]

    # Update novelty memory
    novelty_tracker.add(move_embeds[best_move])

    return best_move
```

## Configuration

```python
@dataclass
class GMOConfig:
    state_dim: int = 128        # State embedding dimension
    move_dim: int = 128         # Move embedding dimension
    hidden_dim: int = 256       # Value network hidden size
    top_k: int = 5              # Candidates to optimize
    optim_steps: int = 10       # Gradient steps per candidate
    lr: float = 0.1             # Optimization learning rate
    beta: float = 0.3           # Uncertainty exploration coefficient
    gamma: float = 0.1          # Novelty exploration coefficient
    exploration_temp: float = 1.0
    dropout_rate: float = 0.1
    mc_samples: int = 10        # MC Dropout samples for uncertainty
    novelty_memory_size: int = 1000
    device: str = "cpu"
```

## Files

| File                        | Description                      |
| --------------------------- | -------------------------------- |
| `app/ai/gmo_ai.py`          | Main implementation (~400 lines) |
| `app/training/train_gmo.py` | Training script (~350 lines)     |
| `tests/test_gmo_ai.py`      | Unit tests                       |

## Usage

### Create GMO AI

```python
from app.ai.factory import AIFactory
from app.models import AIConfig, AIType

# Via factory
ai = AIFactory.create(AIType.GMO, player_number=1, config=AIConfig(difficulty=6))

# Via tournament factory
ai = AIFactory.create_for_tournament("gmo", player_number=1)
```

### Training

```bash
python -m app.training.train_gmo \
    --data-path data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl \
    --output-dir models/gmo \
    --epochs 50 \
    --batch-size 64 \
    --device mps
```

## Why This Is Novel

1. Gradient-based move search: no existing algorithm uses gradient ascent in
   move embedding space at inference time.
2. Continuous move optimization: treats discrete move selection as continuous
   optimization plus projection.
3. Different from policy gradient: policy gradient optimizes network parameters
   during training; GMO optimizes move embeddings at inference time.
4. Different from MCTS: no tree search or rollouts, pure gradient-based
   optimization.
5. Information-theoretic exploration: uses entropy and uncertainty to guide
   search, balancing exploitation with exploration.

## Theoretical Motivation

- The value landscape over moves is often smooth (similar moves have similar values)
- Gradient ascent can find local maxima in this landscape efficiently
- The embedding space provides a continuous relaxation of discrete move space
- Multiple gradient steps can discover non-obvious good moves that a single
  forward pass might miss
