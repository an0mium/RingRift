# Gradient Move Optimization (GMO) Algorithm

**A Novel Game-Playing Algorithm for RingRift**

GMO is a new approach to game AI that uses gradient ascent in continuous move embedding space, guided by information-theoretic exploration principles (uncertainty and novelty), to find optimal moves.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Innovation](#key-innovation)
3. [Theoretical Foundation](#theoretical-foundation)
4. [Architecture](#architecture)
5. [Algorithm Details](#algorithm-details)
6. [Information-Theoretic Components](#information-theoretic-components)
7. [Configuration Reference](#configuration-reference)
8. [Training](#training)
9. [Usage](#usage)
10. [Comparison to Other Approaches](#comparison-to-other-approaches)
11. [File Reference](#file-reference)
12. [Future Improvements](#future-improvements)

---

## Overview

GMO (Gradient Move Optimization) is a neural network-based game AI that differs fundamentally from traditional approaches:

| Traditional AI                        | GMO                                                     |
| ------------------------------------- | ------------------------------------------------------- |
| Enumerate moves -> evaluate -> select | Encode moves -> optimize embeddings -> project to legal |
| Discrete action space                 | Continuous embedding space                              |
| Forward pass only                     | Gradient ascent at inference                            |
| Random exploration                    | Information-theoretic exploration                       |

**Core Insight**: Instead of treating move selection as classification (pick highest probability) or tree search (explore game tree), GMO treats it as continuous optimization (gradient ascent in embedding space).

---

## Key Innovation

### Traditional Neural Network Game AI

```
Policy Network:  state -> neural_net -> softmax -> move_probabilities -> sample
Value + MCTS:    state -> tree_search -> visit_counts -> move_selection
Minimax:         state -> enumerate_moves -> evaluate_leaves -> alpha-beta
```

### GMO Innovation

```
state -> encode -> gradient_ascent(value + uncertainty + novelty) -> optimized_embedding -> nearest_legal_move
```

The key differences:

1. **Continuous Optimization at Inference**: Unlike policy networks that make a single forward pass, GMO runs multiple gradient steps to refine the move selection.

2. **Embedding Space Search**: Moves are embedded in a continuous vector space where similar moves have similar embeddings. This allows gradient-based navigation.

3. **Projection Back to Discrete**: After optimization in continuous space, the result is projected back to the nearest legal move via cosine similarity.

4. **No Tree Search**: Unlike MCTS or Minimax, GMO does not build or traverse a game tree. All computation happens in embedding space.

---

## Theoretical Foundation

### Why Continuous Optimization Works for Discrete Games

1. **Smooth Value Landscape**: The value function over move embeddings is learned to be smooth - similar moves tend to have similar values. This smoothness enables gradient-based optimization.

2. **Continuous Relaxation**: The discrete set of legal moves can be viewed as points in a continuous embedding space. Optimizing in this space finds directions of improvement.

3. **Local Refinement**: Starting from a good candidate move, gradient ascent can discover nearby moves that are even better - moves that a single forward pass might rank lower.

### Information-Theoretic Exploration

GMO balances three objectives:

```
Objective = E[Value] + beta*sqrt(Var) + gamma*Novelty
```

- **E[Value]**: Exploitation - moves predicted to lead to winning
- **sqrt(Var)**: Epistemic uncertainty - moves where the model is uncertain (worth exploring)
- **Novelty**: State-space diversity - moves different from previously played ones

This is inspired by:

- **UCB (Upper Confidence Bound)** from multi-armed bandits: value + exploration bonus
- **Information gain** from Bayesian optimization: explore uncertain regions
- **Novelty search** from evolutionary algorithms: encourage diversity

---

## Architecture

### System Diagram

```
+-----------------------------------------------------------------------------+
|                           GMO Architecture                                   |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +----------------+         +----------------+                              |
|  |   GameState    |         |  Legal Moves   |                              |
|  |   (RingRift)   |         |   [m1..mN]     |                              |
|  +-------+--------+         +-------+--------+                              |
|          |                          |                                        |
|          v                          v                                        |
|  +----------------+         +----------------+                              |
|  |  StateEncoder  |         |  MoveEncoder   |                              |
|  |   768->256->128  |         |  112->128->128   |                              |
|  +-------+--------+         +-------+--------+                              |
|          |                          |                                        |
|          v                          v                                        |
|     state_embed              move_embeds                                     |
|      (128-dim)              (N x 128-dim)                                   |
|          |                          |                                        |
|          +----------+---------------+                                        |
|                     |                                                        |
|                     v                                                        |
|  +---------------------------------------------------------------------+   |
|  |                    Phase 1: Initial Ranking                          |   |
|  |                                                                      |   |
|  |  For each move m_i:                                                 |   |
|  |    1. MC Dropout -> mean_value, variance                             |   |
|  |    2. NoveltyTracker -> novelty_score                                |   |
|  |    3. score = value + beta*sqrt(var) + gamma*novelty                     |   |
|  |                                                                      |   |
|  |  Select top-K candidates by score                                   |   |
|  +---------------------------------------------------------------------+   |
|                     |                                                        |
|                     v                                                        |
|  +---------------------------------------------------------------------+   |
|  |                   Phase 2: Gradient Optimization                     |   |
|  |                                                                      |   |
|  |  For each top-K candidate:                                          |   |
|  |    1. Initialize: m_opt = m_candidate.clone()                       |   |
|  |    2. For step in 1..10:                                            |   |
|  |         objective = value(s, m_opt) + beta*sqrt(var(s, m_opt))            |   |
|  |         m_opt += lr * grad_m objective                                 |   |
|  |    3. Project: find nearest legal move by cosine similarity         |   |
|  |    4. Evaluate final move                                           |   |
|  +---------------------------------------------------------------------+   |
|                     |                                                        |
|                     v                                                        |
|              +-------------+                                                 |
|              |  Best Move  |                                                 |
|              |  (highest   |                                                 |
|              |   score)    |                                                 |
|              +-------------+                                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Component Details

#### 1. StateEncoder (`gmo_ai.py:179-253`)

Encodes a RingRift GameState into a 128-dimensional embedding.

**Features (12 planes x 64 positions = 768 features):**

- Planes 0-3: Ring presence per player (which player owns rings at each position)
- Planes 4-7: Stack control per player (which player controls stacks)
- Planes 8-11: Territory ownership per player

**Architecture:**

```
Input: 768 features (flattened board representation)
  v
Linear(768 -> 256) + ReLU
  v
Linear(256 -> 128)
  v
Output: 128-dim state embedding
```

#### 2. MoveEncoder (`gmo_ai.py:95-172`)

Encodes a Move object into a 128-dimensional embedding.

**Components:**

- `type_embed`: 8 move types -> 32-dim (PLACE_RING, MOVE_STACK, OVERTAKING_CAPTURE, etc.)
- `from_embed`: 65 positions -> 32-dim (64 board positions + None)
- `to_embed`: 65 positions -> 32-dim
- `placement_embed`: 4 values -> 16-dim (0, 1, 2, 3 rings)

**Architecture:**

```
Move -> [type_idx, from_idx, to_idx, placement_count]
  v
[type_embed(32) || from_embed(32) || to_embed(32) || placement_embed(16)]
  v (concatenate = 112 dim)
Linear(112 -> 128) + ReLU
  v
Linear(128 -> 128)
  v
Output: 128-dim move embedding
```

#### 3. GMOValueNetWithUncertainty (`gmo_ai.py:260-326`)

Joint network that predicts both value and uncertainty for a (state, move) pair.

**Architecture:**

```
Input: concat(state_embed, move_embed) = 256-dim
  v
Linear(256 -> 256) + ReLU + Dropout(0.1)
  v
Linear(256 -> 256) + ReLU + Dropout(0.1)
  v
     +--------------+--------------+
     v              v
value_head    uncertainty_head
Linear(256->1)    Linear(256->1)
     v              v
   tanh           raw
     v              v
value in [-1,1]   log_variance
```

**Two Types of Uncertainty:**

1. **Epistemic (model uncertainty)**: Estimated via MC Dropout - run 10 forward passes with dropout enabled, measure variance of predictions
2. **Aleatoric (data uncertainty)**: Learned log_variance output - network learns when outcomes are inherently unpredictable

#### 4. NoveltyTracker (`gmo_ai.py:333-381`)

Tracks previously explored move embeddings to encourage diversity.

**Implementation:**

- Ring buffer storing last 1000 move embeddings
- Novelty = minimum L2 distance to any stored embedding
- Reset at start of each game

```python
novelty(m) = min_{m' in memory} ||embed(m) - embed(m')||_2
```

High novelty = move is far from anything we've tried before -> worth exploring.

---

## Algorithm Details

### Main Algorithm (`select_move` at `gmo_ai.py:597-712`)

```python
def select_move(game_state):
    # 1. Get all legal moves
    legal_moves = get_valid_moves(game_state)

    # 2. Encode state and moves
    state_embed = state_encoder.encode(game_state)  # 128-dim
    move_embeds = [move_encoder.encode(m) for m in legal_moves]  # N x 128

    # 3. Phase 1: Initial ranking with UCB-style scores
    candidates = []
    for i, move_embed in enumerate(move_embeds):
        # MC Dropout: 10 forward passes with dropout enabled
        mean_value, entropy, variance = estimate_uncertainty(state_embed, move_embed)

        # Distance to nearest previously explored move
        novelty = novelty_tracker.compute_novelty(move_embed)

        # UCB-style score
        score = mean_value + beta * sqrt(variance) + gamma * novelty
        candidates.append((i, score, move_embed))

    # Select top 5 candidates
    top_k = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    # 4. Phase 2: Gradient optimization for each top candidate
    best_move, best_score = None, -inf

    for idx, _, initial_embed in top_k:
        # Run gradient ascent in embedding space
        optimized_embed = gradient_optimize(
            state_embed,
            initial_embed,
            steps=10,
            lr=0.1
        )

        # Project back to nearest legal move
        similarities = [cosine_sim(optimized_embed, me) for me in move_embeds]
        nearest_idx = argmax(similarities)

        # Final evaluation of the projected move
        final_score = evaluate(state_embed, move_embeds[nearest_idx])

        if final_score > best_score:
            best_score = final_score
            best_move = legal_moves[nearest_idx]

    # 5. Update novelty tracker with selected move
    novelty_tracker.add(embed(best_move))

    return best_move
```

### Gradient Optimization (`optimize_move_with_entropy` at `gmo_ai.py:429-472`)

```python
def optimize_move_with_entropy(state_embed, initial_move_embed, config):
    # Clone embedding and enable gradients
    move_embed = initial_move_embed.clone().requires_grad_(True)
    optimizer = Adam([move_embed], lr=0.1)

    for step in range(10):
        optimizer.zero_grad()

        # Estimate value and uncertainty via MC Dropout
        mean_value, entropy, variance = estimate_uncertainty(
            state_embed, move_embed, value_net, n_samples=10
        )

        # Anneal exploration over optimization steps
        # Start with high exploration, reduce to pure exploitation
        progress = step / 9  # 0.0 to 1.0
        exploration_weight = beta * (1 - progress) * temperature

        # Objective: maximize value + exploration bonus
        objective = mean_value + exploration_weight * sqrt(variance)

        # Gradient ascent (minimize negative objective)
        loss = -objective
        loss.backward()
        optimizer.step()

    return move_embed.detach()
```

### MC Dropout Uncertainty Estimation (`estimate_uncertainty` at `gmo_ai.py:388-426`)

```python
def estimate_uncertainty(state_embed, move_embed, value_net, n_samples=10):
    # Enable dropout during inference
    value_net.train()

    # Collect n_samples predictions with different dropout masks
    values = []
    for _ in range(n_samples):
        value, log_var = value_net(state_embed, move_embed)
        values.append(value)

    values = torch.stack(values)

    # Statistics
    mean_value = values.mean()
    variance = values.var() + 1e-8  # Epistemic uncertainty

    # Gaussian entropy: H = 0.5 * log(2pie*sigma^2)
    entropy = 0.5 * log(2pi * e * variance)

    return mean_value, entropy, variance
```

### Projection to Legal Move (`project_to_legal_move` at `gmo_ai.py:475-505`)

```python
def project_to_legal_move(optimized_embed, move_embeds, legal_moves, temperature=0):
    # Normalize embeddings
    opt_norm = normalize(optimized_embed)
    moves_norm = normalize(move_embeds)

    # Cosine similarities
    similarities = dot(opt_norm, moves_norm.T)

    if temperature > 0:
        # Soft selection: sample from softmax
        probs = softmax(similarities / temperature)
        idx = sample(probs)
    else:
        # Hard selection: argmax
        idx = argmax(similarities)

    return legal_moves[idx], idx
```

### Exploration Temperature Adaptation (`get_exploration_temperature` at `gmo_ai.py:508-522`)

```python
def get_exploration_temperature(game_state, base_temp=1.0):
    move_count = len(game_state.move_history)
    rings_on_board = sum(stack.height for stack in game_state.board.stacks)

    if move_count < 10:
        return base_temp * 1.5   # Opening: explore more
    elif rings_on_board > 20:
        return base_temp * 0.5   # Complex position: exploit more
    else:
        return base_temp         # Midgame: balanced
```

---

## Information-Theoretic Components

### 1. Monte Carlo Dropout for Epistemic Uncertainty

**Concept**: Use dropout at inference time to approximate Bayesian uncertainty.

**Implementation**:

- Keep dropout enabled (`model.train()`) during forward passes
- Run 10 forward passes with different dropout masks
- Variance of predictions = epistemic uncertainty

**Interpretation**:

- High variance -> model has seen few similar positions -> explore
- Low variance -> model is confident -> exploit

**Reference**: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"

### 2. Learned Aleatoric Uncertainty

**Concept**: Let the network output its own uncertainty estimate.

**Implementation**:

- ValueNet outputs both `value` and `log_variance`
- Loss function: `precision * MSE + log_variance`
  - `precision = exp(-log_variance)`

**Training behavior**:

- If prediction is wrong: increase variance to reduce loss
- If prediction is accurate: decrease variance
- Network learns which positions are inherently unpredictable

**Reference**: Nix & Weigend (1994) "Estimating the mean and variance of the target probability distribution"

### 3. Novelty Bonus

**Concept**: Encourage exploring diverse moves, not just high-value moves.

**Implementation**:

- Maintain ring buffer of last 1000 move embeddings played
- Novelty = L2 distance to nearest neighbor in buffer
- Add gamma\*novelty to the objective

**Effect**:

- Prevents repetitive play patterns
- Encourages trying new strategic ideas
- Helps discover non-obvious good moves

**Reference**: Lehman & Stanley (2011) "Abandoning Objectives: Evolution Through the Search for Novelty Alone"

### 4. UCB-Style Exploration

**Concept**: Balance exploitation (high expected value) with exploration (high uncertainty).

**Formula**:

```
score = E[value] + beta*sqrt(variance) + gamma*novelty
```

**Analogy to UCB1**:

```
UCB1: x_bar + c*sqrt(ln(n)/n_i)
GMO:  E[V] + beta*sqrt(Var) + gamma*novelty
```

Both add an exploration bonus proportional to uncertainty.

---

## Configuration Reference

```python
@dataclass
class GMOConfig:
    # === Embedding Dimensions ===
    state_dim: int = 128     # State embedding size
    move_dim: int = 128      # Move embedding size
    hidden_dim: int = 256    # Value network hidden layer size

    # === Optimization Parameters ===
    top_k: int = 5           # Number of candidates to optimize
    optim_steps: int = 10    # Gradient steps per candidate
    lr: float = 0.1          # Learning rate for move optimization

    # === Exploration Parameters ===
    beta: float = 0.3        # Uncertainty exploration coefficient
                             # Higher = more exploration of uncertain moves
    gamma: float = 0.1       # Novelty exploration coefficient
                             # Higher = more diverse move selection
    exploration_temp: float = 1.0  # Base exploration temperature

    # === MC Dropout Parameters ===
    dropout_rate: float = 0.1  # Dropout probability in value network
    mc_samples: int = 10       # Number of dropout samples for uncertainty

    # === Novelty Tracking ===
    novelty_memory_size: int = 1000  # Ring buffer size

    # === Device ===
    device: str = "cpu"  # "cpu", "cuda", or "mps"
```

### Parameter Tuning Guide

| Parameter     | Low Value Effect            | High Value Effect           | Recommended Range |
| ------------- | --------------------------- | --------------------------- | ----------------- |
| `beta`        | Exploit known good moves    | Explore uncertain moves     | 0.1 - 0.5         |
| `gamma`       | Allow repetitive patterns   | Force diverse play          | 0.05 - 0.2        |
| `top_k`       | Faster, may miss good moves | Thorough, slower            | 3 - 10            |
| `optim_steps` | Quick, less refined         | Better optimization, slower | 5 - 20            |
| `lr`          | Slow convergence            | May overshoot               | 0.05 - 0.2        |
| `mc_samples`  | Noisy uncertainty           | Accurate but slow           | 5 - 20            |

---

## Training

### Data Format

GMO trains on game records in JSONL format:

```json
{
  "initial_state": {
    /* GameState object */
  },
  "moves": [
    /* array of Move objects */
  ],
  "winner": 1,
  "board_type": "square8"
}
```

### Training Procedure

1. **Load game records** from JSONL file
2. **Extract (state, move, outcome) tuples**:
   - State features from initial position
   - Move embeddings for each move played
   - Outcome: +1 if player won, -1 if lost (with temporal discounting)
3. **Train networks** to predict outcomes with uncertainty
4. **Evaluate periodically** by playing vs Random AI

### Loss Function

Negative log-likelihood with learned uncertainty:

```
L = exp(-log_var) * (pred - target)^2 + log_var
```

This loss:

- Penalizes wrong predictions (MSE term)
- Penalizes overconfident wrong predictions (precision weighting)
- Penalizes excessive uncertainty (log_var regularization)

### Training Command

```bash
python -m app.training.train_gmo \
    --data-path data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl \
    --output-dir models/gmo \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --eval-interval 5 \
    --device mps \
    --verbose
```

### Training Arguments

| Argument          | Default                                              | Description                 |
| ----------------- | ---------------------------------------------------- | --------------------------- |
| `--data-path`     | `data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl` | Training data               |
| `--output-dir`    | `models/gmo`                                         | Checkpoint output directory |
| `--epochs`        | 50                                                   | Number of training epochs   |
| `--batch-size`    | 64                                                   | Batch size                  |
| `--lr`            | 0.001                                                | Learning rate               |
| `--max-samples`   | None                                                 | Limit training samples      |
| `--eval-interval` | 5                                                    | Epochs between evaluations  |
| `--device`        | cpu                                                  | Device (cpu/cuda/mps)       |
| `--verbose`       | False                                                | Enable debug logging        |

### Checkpoints

Training saves two checkpoints:

- `gmo_best.pt`: Best validation loss during training
- `gmo_final.pt`: Final model after all epochs

Checkpoint contents:

```python
{
    "state_encoder": state_encoder.state_dict(),
    "move_encoder": move_encoder.state_dict(),
    "value_net": value_net.state_dict(),
    "gmo_config": GMOConfig(...),
    "epoch": int,
    "val_loss": float,
}
```

---

## Usage

### Creating a GMO AI

```python
from app.ai.factory import AIFactory
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.models import AIConfig, AIType
from pathlib import Path

# Method 1: Via factory (uses defaults)
ai = AIFactory.create(AIType.GMO, player_number=1, config=AIConfig(difficulty=6))

# Method 2: Via tournament factory
ai = AIFactory.create_for_tournament("gmo", player_number=1)

# Method 3: Direct instantiation with custom config
gmo_config = GMOConfig(
    beta=0.4,      # More exploration
    gamma=0.15,    # More novelty seeking
    top_k=7,       # More candidates
    device="mps",  # Use Apple Silicon GPU
)
ai = GMOAI(player_number=1, config=AIConfig(difficulty=6), gmo_config=gmo_config)

# Load trained weights
ai.load_checkpoint(Path("models/gmo/gmo_best.pt"))
```

### Playing a Game

```python
from app.game_engine import GameEngine

engine = GameEngine()
state = create_initial_state()  # Your game state initialization

while state.winner is None:
    current_player = state.current_player

    if current_player == 1:
        move = gmo_ai.select_move(state)
    else:
        move = opponent_ai.select_move(state)

    state = engine.apply_move(state, move)

print(f"Winner: Player {state.winner}")
```

### Evaluating GMO vs Other AIs

```python
from app.training.train_gmo import evaluate_vs_random

# Load trained model
gmo_ai = GMOAI(1, AIConfig(difficulty=6))
gmo_ai.load_checkpoint(Path("models/gmo/gmo_best.pt"))

# Run evaluation
results = evaluate_vs_random(gmo_ai, num_games=100)
print(f"Win rate vs Random: {results['win_rate']:.1%}")
print(f"Average game length: {results['avg_game_length']:.1f} moves")
```

---

## Comparison to Other Approaches

| Approach           | Move Selection        | Exploration     | Uncertainty          | Compute at Inference   |
| ------------------ | --------------------- | --------------- | -------------------- | ---------------------- |
| **Random AI**      | Uniform random        | N/A             | None                 | O(1)                   |
| **Heuristic AI**   | Hand-crafted rules    | Rule-based      | None                 | O(N) per move          |
| **Policy Network** | Softmax sampling      | Entropy bonus   | None                 | O(1) forward pass      |
| **MCTS**           | Visit counts          | UCT formula     | Visit-based          | O(simulations)         |
| **AlphaZero**      | MCTS + Policy prior   | Dirichlet noise | Visit-based          | O(simulations)         |
| **GMO**            | Gradient optimization | Info-theoretic  | MC Dropout + learned | O(K x steps x samples) |

### Advantages of GMO

1. **No Tree Search**: Avoids exponential blowup of game tree
2. **Continuous Refinement**: Can discover moves a single forward pass might miss
3. **Principled Exploration**: Information-theoretic rather than random noise
4. **Uncertainty Quantification**: Knows when it doesn't know
5. **Fast at Low Settings**: With top_k=1, steps=1, it's just a forward pass

### Disadvantages of GMO

1. **Projection Loss**: Optimized embedding may not correspond to any legal move perfectly
2. **Local Optima**: Gradient ascent may not find global optimum
3. **Training Data Quality**: Learns from quality of training games
4. **Compute Cost**: Multiple forward passes per move selection

---

## File Reference

| File                        | Lines     | Description             |
| --------------------------- | --------- | ----------------------- |
| `app/ai/gmo_ai.py`          | ~400      | Main implementation     |
| `app/training/train_gmo.py` | ~350      | Training script         |
| `tests/test_gmo_ai.py`      | ~620      | Unit tests (28 tests)   |
| `docs/GMO_ALGORITHM.md`     | This file | Documentation           |
| `models/gmo/gmo_best.pt`    | ~1.5MB    | Best trained checkpoint |
| `models/gmo/gmo_final.pt`   | ~1.5MB    | Final checkpoint        |

### Key Functions

| Function                     | Location           | Description                             |
| ---------------------------- | ------------------ | --------------------------------------- |
| `GMOAI.select_move`          | `gmo_ai.py:597`    | Main move selection algorithm           |
| `estimate_uncertainty`       | `gmo_ai.py:388`    | MC Dropout uncertainty estimation       |
| `optimize_move_with_entropy` | `gmo_ai.py:429`    | Gradient ascent in embedding space      |
| `project_to_legal_move`      | `gmo_ai.py:475`    | Project embedding to nearest legal move |
| `nll_loss_with_uncertainty`  | `gmo_ai.py:746`    | Training loss function                  |
| `train_gmo`                  | `train_gmo.py:404` | Main training function                  |

---

## Future Improvements

### Short-term

1. **Hyperparameter sweep**: Systematic tuning of beta, gamma, top_k, optim_steps
2. **Self-play training**: Generate GMO vs GMO games for continued learning
3. **Ensemble uncertainty**: Use multiple value networks instead of dropout

### Medium-term

1. **State-conditioned exploration**: Learn beta and gamma as functions of game state
2. **Learned move embeddings**: Fine-tune move encoder end-to-end
3. **Multi-step lookahead**: Optimize sequences of moves, not just single moves

### Long-term

1. **Combine with search**: Use GMO to propose moves for MCTS consideration
2. **Meta-learning**: Adapt exploration parameters based on opponent model
3. **Hierarchical optimization**: Optimize high-level strategy embeddings first

---

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML.

2. Nix, D. A., & Weigend, A. S. (1994). Estimating the mean and variance of the target probability distribution. ICNN.

3. Lehman, J., & Stanley, K. O. (2011). Abandoning Objectives: Evolution Through the Search for Novelty Alone. Evolutionary Computation.

4. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning.

5. Silver, D., et al. (2017). Mastering the Game of Go without Human Knowledge. Nature.

---

_Last updated: December 2024_
_Author: Claude (Anthropic)_
_RingRift AI Service v1.0_
