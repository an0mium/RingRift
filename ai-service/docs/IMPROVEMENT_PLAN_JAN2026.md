# RingRift Improvement Plan - January 2026

**Created**: January 10, 2026
**Status**: Active

---

## Executive Summary

RingRift is a mature project with world-class infrastructure but needs focused effort on AI strength. The frontend is production-ready (A-), but only 1 of 12 AI models is competitive.

| Component      | Grade       | Status                         |
| -------------- | ----------- | ------------------------------ |
| Frontend       | A- (90/100) | Production-ready               |
| Infrastructure | A (95/100)  | World-class                    |
| AI Strength    | C+ (65/100) | Only 1/12 models competitive   |
| Overall        | B+          | Ready for users, AI needs work |

---

## Current State

### AI Model Elo Ratings

| Config           | Current Elo | Games    | Target Elo | Gap          |
| ---------------- | ----------- | -------- | ---------- | ------------ |
| **square8_2p**   | **1674** ✓  | 169      | 1900       | -226         |
| hex8_2p          | 1500        | 243      | 1750       | -250         |
| square19_2p      | 1500        | 53       | 1800       | -300         |
| hex8_3p          | 1500        | 132      | 1700       | -200         |
| square8_3p       | 1500        | 147      | 1700       | -200         |
| square19_3p      | 1500        | 223      | 1700       | -200         |
| hexagonal_2p     | 1500        | 74       | 1700       | -200         |
| hexagonal_3p     | 1500        | 217      | 1700       | -200         |
| hex8_4p          | 1500        | 127      | 1650       | -150         |
| square8_4p       | 1500        | 41       | 1650       | -150         |
| square19_4p      | 1500        | 69       | 1650       | -150         |
| **hexagonal_4p** | 1500        | **5** ❌ | 1600       | **CRITICAL** |

### Game Data Distribution

Total: ~118,000 games across all configurations

```
hexagonal_4p:   15,999 games (largest)
square8_4p:     31,486 games
hex8_2p:        16,441 games
square19_2p:    19,548 games
square8_2p:      7,569 games
hex8_3p:         7,245 games
hex8_4p:         3,951 games
hexagonal_2p:    4,424 games
hexagonal_3p:    1,936 games
square8_3p:      3,794 games
square19_3p:     3,444 games
square19_4p:     1,766 games
```

### Frontend Status

- **Framework**: React 19 + TypeScript + Vite
- **Features**: Complete game loop, multiplayer, AI, accessibility
- **Polish**: Professional with animations, colorblind modes, mobile support
- **Gaps**: No tutorial mode, no game analysis features

---

## Improvement Priorities

### Phase 1: Critical AI Gaps (Week 1)

**Goal**: Get all 12 configs to baseline competitiveness

| Task | Config       | Games Needed | Priority |
| ---- | ------------ | ------------ | -------- |
| 1    | hexagonal_4p | 500+         | CRITICAL |
| 2    | square8_4p   | 300+         | HIGH     |
| 3    | square19_4p  | 300+         | HIGH     |
| 4    | hex8_3p      | 200+         | MEDIUM   |

**Actions**:

- [x] Start master_loop.py for continuous training
- [ ] Verify selfplay allocation prioritizes undertrained configs
- [ ] Monitor Elo progression daily

### Phase 2: Push Elo Higher (Weeks 2-4)

**Goal**: Get 2p models to 1750+ Elo, 3p/4p to 1650+

| Config      | Current | Target | Strategy                |
| ----------- | ------- | ------ | ----------------------- |
| square8_2p  | 1674    | 1900   | More games, longer MCTS |
| hex8_2p     | 1500    | 1750   | Priority allocation     |
| square19_2p | 1500    | 1800   | Large board focus       |

**Actions**:

- [ ] Increase Gumbel budget for 2p configs (200 → 400)
- [ ] Evaluate v5-heavy architecture for large boards
- [ ] Run gauntlet evaluations weekly

### Phase 3: Product Polish (Weeks 3-6)

**Goal**: Prepare for public launch

| Task                  | Impact            | Effort |
| --------------------- | ----------------- | ------ |
| Add tutorial mode     | High (onboarding) | Medium |
| Add difficulty labels | High (UX)         | Low    |
| Deployment pipeline   | Critical          | Medium |
| Load testing          | Required          | Low    |

**Actions**:

- [ ] Map Elo ranges to difficulty levels (Easy/Medium/Hard/Expert)
- [ ] Create interactive tutorial for first-time players
- [ ] Set up production deployment (Vercel/Railway/etc.)
- [ ] Run load tests with 100+ concurrent users

---

## Success Metrics

### AI Quality

| Metric                 | Current | Week 4 Target | Launch Target |
| ---------------------- | ------- | ------------- | ------------- |
| Configs above 1600 Elo | 1/12    | 6/12          | 10/12         |
| Average Elo (2p)       | 1558    | 1700          | 1800          |
| Average Elo (3p/4p)    | 1500    | 1600          | 1650          |
| Weakest config Elo     | 1500    | 1550          | 1600          |

### Product Quality

| Metric              | Current | Launch Target |
| ------------------- | ------- | ------------- |
| Lighthouse score    | Unknown | 90+           |
| First paint         | Unknown | <2s           |
| Time to interactive | Unknown | <4s           |
| Mobile usability    | Good    | Excellent     |

---

## Technical Tasks

### Immediate (This Week)

1. **Verify cluster training**

   ```bash
   curl http://localhost:8770/status | python3 -c 'import sys,json; print(json.load(sys.stdin).get("alive_peers"))'
   ```

2. **Check selfplay allocation**

   ```bash
   tail -f /tmp/master_loop.log | grep -i "selfplay\|dispatch\|allocation"
   ```

3. **Monitor Elo progression**
   ```bash
   sqlite3 data/unified_elo.db "SELECT participant_id, board_type, num_players, rating FROM elo_ratings WHERE rating > 1500 ORDER BY rating DESC"
   ```

### Short-term (This Month)

1. **Evaluate v5-heavy models**
   - Train v5-heavy for hex8_2p and square8_2p
   - Compare Elo after 1000 games each
   - Decision: adopt v5-heavy if +50 Elo improvement

2. **Increase training throughput**
   - Ensure all Lambda GH200 nodes are training
   - Target: 10,000 games/day across all configs

3. **Frontend deployment**
   - Set up CI/CD pipeline
   - Deploy to staging environment
   - Run E2E tests in CI

---

## Risk Factors

| Risk                   | Likelihood | Impact | Mitigation                    |
| ---------------------- | ---------- | ------ | ----------------------------- |
| Cluster node failures  | Medium     | Low    | Auto-recovery daemons         |
| Elo plateau            | Medium     | Medium | Architecture changes          |
| Frontend bugs          | Low        | Medium | Comprehensive tests           |
| Model distribution lag | Medium     | Low    | Pre-distribution before games |

---

## Resource Allocation

### Cluster Compute

| Node Type    | Count | Usage               |
| ------------ | ----- | ------------------- |
| Lambda GH200 | 11    | Training (primary)  |
| Nebius H100  | 2     | Training + Selfplay |
| Nebius L40S  | 1     | Selfplay            |
| Hetzner CPU  | 3     | P2P voting only     |
| Vast.ai      | 14    | Selfplay            |

### Human Time

| Task        | Hours/Week |
| ----------- | ---------- |
| Monitoring  | 2-3h       |
| Debugging   | 2-4h       |
| Development | 8-16h      |
| Total       | 12-23h     |

---

## Next Review

- **Daily**: Check Elo progression, cluster health
- **Weekly**: Review config allocation, adjust priorities
- **Monthly**: Full assessment, plan update

---

## Commands Reference

```bash
# Start cluster automation
PYTHONPATH=. python3 scripts/master_loop.py

# Check cluster status
curl -s http://localhost:8770/status | python3 -m json.tool

# View Elo ratings
sqlite3 data/unified_elo.db "SELECT * FROM elo_ratings ORDER BY rating DESC LIMIT 20"

# Check game counts
curl -s http://localhost:8770/game_counts | python3 -m json.tool

# Dispatch specific selfplay
curl -X POST http://LEADER:8770/dispatch_selfplay -d '{"config_key": "hexagonal_4p", "num_games": 100}'

# Monitor training
tail -f /tmp/master_loop.log | grep -i "training\|epoch\|loss"
```

---

_This plan will be updated weekly based on progress._
