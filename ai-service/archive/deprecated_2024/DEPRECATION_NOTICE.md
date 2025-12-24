# Deprecated AI Approaches - December 2024

## Summary

The following AI approaches have been deprecated based on comprehensive evaluation.
They are archived here for reference but should NOT be used for new development.

## Deprecated: GMO (Gumbel MuZero Optimization)

**Files:**

- `app/ai/gmo_ai.py`
- `app/ai/gmo_v2.py`
- `app/ai/gmo_shared.py`
- `app/ai/gmo_mcts_hybrid.py`
- `app/ai/gmo_gumbel_hybrid.py`
- `app/ai/gmo_policy_provider.py`
- `scripts/gmo_*.py`

**Why Deprecated:**

- 500x temperature scaling indicates fundamentally broken uncertainty calibration
- Fragile hyperparameters (optim_steps=5, top_k=5, beta=0.5)
- Multiple training variants (diverse, online, selfplay) with no clear winner
- ~500 forward passes per move vs 1 for policy-only

**Performance:**

- vs Random: 100% win rate
- vs Heuristic: 80% win rate
- Training stability: Fragile

## Deprecated: EBMO (Energy-Based Move Optimization)

**Files:**

- `app/ai/ebmo_ai.py`
- `app/ai/ebmo_network.py`
- `app/ai/ebmo_online_learner.py`
- `app/training/ebmo_trainer.py`
- `app/training/ebmo_dataset.py`

**Why Deprecated:**

- 35% win rate vs Heuristic AI (worse than random!)
- Trained on MCTS labels (teaches "what MCTS would do" not "what wins")
- ~2000 forward passes per move (100 steps × 8 restarts × 2.5 passes)
- Gradient descent escapes legal move manifold

**Performance:**

- vs Random: 70% win rate
- vs Heuristic: 35% win rate (CRITICAL FAILURE)
- Inference cost: ~2000 FP per move

## Recommended Alternatives

1. **GNN-based architecture** (see `app/ai/neural_net/gnn_policy.py`)
2. **Entropy-guided MCTS** (see `app/ai/entropy_mcts.py`)
3. **Policy distillation** for fast inference
4. **MARL** for multi-player games

## Migration Path

If you have code using GMO or EBMO:

```python
# OLD (deprecated)
from app.ai.gmo_ai import GMOAI
ai = GMOAI(player_number=1, config=config)

# NEW (recommended)
from app.ai.gnn_policy_ai import GNNPolicyAI
ai = GNNPolicyAI(player_number=1, config=config)
```

## Archive Location

Original files preserved in:

- `archive/deprecated_2024/gmo/`
- `archive/deprecated_2024/ebmo/`

---

_Deprecated: December 2024_
_Decision by: AI Architecture Review_
