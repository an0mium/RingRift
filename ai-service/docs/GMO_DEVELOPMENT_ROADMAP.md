# GMO Development Roadmap

## Current Status (December 23, 2025)

### Completed

1. **GMO + Gumbel MCTS Integration** (`app/ai/gmo_gumbel_hybrid.py`)
   - Created `GumbelMCTSGMOAI` class combining GMO value network with Gumbel search
   - Added `AIType.GMO_GUMBEL` to factory and difficulty profiles
   - Integrated into gauntlet tournament system

2. **Player-Relative Encoding Fix** (`app/ai/gmo_ai.py`)
   - Fixed StateEncoder to use player-relative feature planes
   - Plane 0 = current player, Plane 1 = opponent (instead of absolute P1/P2)
   - Updated `train_gmo.py` to replay games with correct perspective

3. **Data Augmentation Fix** (`app/training/train_gmo.py`)
   - Fixed API call: `transform_board()` instead of `apply_transform()`

### In Progress

- **GMO Retraining** on GH200 cluster
  - Training data: 447,473 samples from 3,888 square8 2p games
  - Configuration: 100 epochs, batch 64, lr 0.0005, D4 augmentation
  - Monitoring script running with auto-evaluation on completion

---

## Evaluation Results (Pre-fix)

| Model  | vs Random | vs Heuristic | vs MCTS-d5 | P2 Win Rate |
| ------ | --------- | ------------ | ---------- | ----------- |
| GMO    | 90%       | 75%          | **85%**    | 80%         |
| IG-GMO | 75%       | 55%          | -          | -           |
| EBMO   | 70%       | 35%          | -          | -           |

**Key Finding**: GMO standalone achieves 85% vs MCTS-d5 without any search augmentation.

---

## Recommended Next Steps

### Phase 1: Validation (Immediate)

After current training completes:

1. **Verify P2 Win Rate Improvement**
   - Target: P2 win rate >40% (was 0% before fix due to encoding bug)
   - Run: `python scripts/gmo_post_training_eval.py --checkpoint models/gmo/sq8_2p_playerrel/gmo_best.pt`

2. **Compare GMO vs GMO-Gumbel**
   - Baseline comparison at equal "intelligence budget"
   - GMO standalone vs GMO-Gumbel (budget=150)

3. **Fair CNN Comparison**
   - CNN standalone vs MCTS-d5
   - CNN-Gumbel vs MCTS-d5 (same budget as GMO-Gumbel)

### Phase 2: Extended Model Training (This Week)

4. **Train GMO v2 with Player-Relative Encoding**

   ```bash
   python -m app.training.train_gmo_v2 \
       --data-path data/gmo_training/combined_sq8_2p.jsonl \
       --output-dir models/gmo_v2/sq8_2p_playerrel \
       --epochs 80 --device cuda
   ```

5. **Train IG-GMO with Player-Relative Encoding**

   ```bash
   python -m app.training.train_ig_gmo \
       --data-path data/gmo_training/combined_sq8_2p.jsonl \
       --output-dir models/ig_gmo/sq8_2p_playerrel \
       --epochs 60 --device cuda
   ```

6. **EBMO Full Dataset Training**
   - Fix max_samples limit in `ebmo_dataset.py`
   - Train on full 289K samples

### Phase 3: Comprehensive Tournament (Next Week)

7. **Full Gauntlet Tournament**

   ```bash
   python -m app.tournament.composite_gauntlet \
       --models \
           models/gmo/sq8_2p_playerrel/gmo_best.pt \
           models/gmo_v2/sq8_2p_playerrel/gmo_v2_best.pt \
           models/ig_gmo/sq8_2p_playerrel/ig_gmo_best.pt \
       --algorithms gumbel_mcts,mcts,descent,gmo_gumbel \
       --baselines random,heuristic,mcts_d5,mcts_d7 \
       --games-per-pair 30
   ```

8. **Elo Rating Calculation**
   - Establish Elo ratings for all (NN, Algorithm) combinations
   - Target: GMO-Gumbel achieving 1800+ Elo

### Phase 4: Production Integration (If Successful)

9. **Add to Difficulty Ladder**
   - GMO at D15 (experimental)
   - GMO-Gumbel at D21 (experimental)

10. **Automated Training Pipeline**
    - Add GMO to continuous training loop
    - Enable online learning for GMO during gameplay

11. **Cross-Board Generalization**
    - Train GMO variants for hexagonal boards
    - Test transfer learning from square8 to hex8

---

## Architecture Comparison

| Model       | Params | Encoder          | Value Head     | Search Compatible     |
| ----------- | ------ | ---------------- | -------------- | --------------------- |
| GMO         | ~400K  | MLP (128d)       | Uncertainty    | Gumbel, MCTS          |
| GMO v2      | ~600K  | Attention (256d) | Ensemble       | Gumbel, MCTS          |
| IG-GMO      | ~500K  | GNN (4-head)     | MI-regularized | Gumbel, MCTS          |
| EBMO        | ~25M   | Contrastive      | Energy-based   | Direct                |
| CNN (v2-v4) | ~2M    | Residual Conv    | Policy+Value   | Gumbel, MCTS, Descent |

---

## Key Metrics to Track

1. **Standalone Win Rate vs MCTS-d5**: Target >80%
2. **P2 Win Rate**: Target >40% (tests perspective invariance)
3. **Training Convergence**: val_loss < 0.5 without divergence
4. **Elo Rating**: Target >1800 (with search), >1500 (standalone)
5. **Games per Second**: Measure inference speed

---

## Files Modified/Created

### New Files

- `app/ai/gmo_gumbel_hybrid.py` - GMO + Gumbel MCTS hybrid
- `scripts/gmo_post_training_eval.py` - Post-training evaluation
- `scripts/monitor_gmo_training.sh` - Training monitor

### Modified Files

- `app/ai/gmo_ai.py` - Player-relative encoding
- `app/training/train_gmo.py` - Data augmentation fix, game replay
- `app/models/core.py` - Added AIType.GMO_GUMBEL
- `app/ai/factory.py` - Factory support for GMO_GUMBEL
- `app/training/composite_participant.py` - GMO_GUMBEL config
- `app/tournament/composite_gauntlet.py` - Phase 2 algorithms

---

## Commands Reference

### Monitor Training

```bash
ssh ubuntu@192.222.57.162 "cd ~/ringrift/ai-service && tail -50 logs/gmo_train_playerrel_v2*.log"
```

### Check Training Process

```bash
ssh ubuntu@192.222.57.162 "ps aux | grep train_gmo | grep -v grep"
```

### View Monitor Output

```bash
ssh ubuntu@192.222.57.162 "cd ~/ringrift/ai-service && tail -50 logs/gmo_monitor*.log"
```

### Manual Evaluation

```bash
ssh ubuntu@192.222.57.162 "cd ~/ringrift/ai-service && python scripts/gmo_post_training_eval.py --checkpoint models/gmo/sq8_2p_playerrel/gmo_best.pt --device cuda"
```
