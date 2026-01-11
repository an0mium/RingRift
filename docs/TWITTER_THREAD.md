# Twitter/X Thread: Recreating AlphaZero as a Non-Programmer

## Thread

**1/**
Me 6 weeks ago: "I'll just recreate AlphaZero. How hard could training a neural network be?"

(Narrator: it was hard)

Me today: running a 41-node GPU cluster with 132 daemons, 11 of which exist to recover from the failures of the other 121.

The AI trained itself to play _worse_ than my backup heuristic.

I still don't know Python.

---

**2/**
Let me explain.

In 2016, DeepMind used hundreds of engineers and thousands of TPUs to create AlphaZero.

8 years later, I tried to recreate it. I'm not a programmer. I used Claude Code.

Here's what the democratization of AI research actually looks like:

---

**3/**
The goal wasn't just "make a game with AI."

It was: can one non-technical person recreate the entire AlphaZero training pipeline from scratch?

Self-play → neural network training → evaluation → model promotion → repeat.

No existing implementations. No shortcuts. Just vibes.

---

**4/**
I also designed a new game instead of using Chess/Go.

Not just for fun—I wanted to create something where humans might stay competitive with AI.

Chess and Go? Engines crushed humans years ago. RingRift has:

- Million+ choices per turn (vs Chess's ~35)
- 3-4 player coalition dynamics AI can't easily model
- Three interacting victory conditions

The game is ~65% novel compared to existing abstract games. No existing game combines all its mechanics.

---

**5/**
What I ended up building (a study in misplaced priorities):

| Metric         | Value    |
| -------------- | -------- |
| Python modules | 341      |
| Test coverage  | 99.5%    |
| Daemon types   | 132      |
| Event types    | 292      |
| Lines of code  | ~250,000 |
| Active users   | 0        |

The only number that matters is the last one.

---

**6/**
The cluster:

- 11 Lambda GH200 nodes (96GB each)
- 14 Vast.ai nodes (RTX 5090, 4090, 3090)
- 6 RunPod nodes (H100, A100)
- Plus Nebius, Hetzner, 2 local Macs

~1.5TB of GPU memory, coordinated by a P2P mesh I definitely didn't understand when I built it.

I still don't.

---

**7/**
The P2P orchestrator alone is 26,000 lines of code.

It has:

- Leader election
- Gossip protocol
- Circuit breakers with 4-tier escalation
- 22 background loops
- 11 recovery daemons

It runs for 48+ hours without human intervention.

The AI still got dumber.

---

**8/**
Here's a real transcript:

Me: "Training completed but Elo dropped from 1188 to 1141. Why is it getting worse?"

Claude: "Let me check..."

_2 hours later_

Claude: "The NPZ sync daemon was refreshing its local file list but not actually pushing files to GPU nodes. Training used 2-week-old data."

---

**9/**
So the self-improvement loop had been making the AI worse for two weeks.

Current best model: 1141 Elo
Simple heuristic fallback: 1200 Elo

The AI successfully trained itself to be worse than the backup plan.

AlphaZero this is not.

---

**10/**
There's also an anomaly detector designed to halt training when something goes wrong.

What it caught: Normal loss values of 65-67 (flagged as "10 consecutive anomalies")
What it missed: Training data being 2 weeks old

We disabled it.

---

**11/**
Other highlights:

- $140/month EC2 instance I couldn't SSH into (wrong key, never configured, just found it)
- Production site down for a week (forgot to set ALLOWED_ORIGINS)
- Sandbox AI defaulting to 50% random moves (users complained it was "too easy")

---

**12/**
The workflow that emerged with Claude Code:

1. Describe problem in plain English
2. Claude reads 50 files, proposes fix
3. I say "that's close but..."
4. Claude writes tests automatically
5. Tests pass
6. Something else breaks 2 weeks later

---

**13/**
What surprised me:

- 99.5% test coverage happened naturally
- The code quality is genuinely good
- Claude maintains consistent style across 341 modules
- I built systems I don't understand
- The systems don't always do what I think they do

---

**14/**
But here's the thing:

8 years ago, this required DeepMind.
Today, it required one person and 6 weeks.

That's a phase change.

The question isn't "can you build this?" anymore.

It's "should you?" and "does it actually work?"

---

**15/**
Lessons learned:

- AI-assisted dev is collaborative, not automatic
- You can build things you don't understand (not always good)
- Tests verify code runs, not that it's correct
- "48h autonomous operation" and "48h of improvement" are different
- Ship in week 2, not week 6

---

**16/**
What's next:

- Fix the training loop so AI improves instead of regresses
- Actually get some users (novel concept)
- Right-size infrastructure (41 nodes for 0 users is... a choice)
- Maybe open-source the training infra
- Probably discover more things that were broken all along

---

**17/**
Full blog post with technical details: [link]

Built with @AnthropicAI's Claude Code
Game: RingRift

DMs open if you want to chat about:

- Democratizing AI research
- Building the wrong thing really well
- Paying $140/month for servers you can't access

/end

---

## Alt Hooks

### Hook A: The Narrator Voice (recommended - used above)

"Me 6 weeks ago: 'I'll just recreate AlphaZero. How hard could training a neural network be?'

(Narrator: it was hard)

Me today: running a 41-node GPU cluster with 132 daemons. The AI trained itself to play _worse_ than my backup heuristic.

I still don't know Python."

### Hook B: The Failure Lead

"I built a 41-node GPU cluster to train a neural network.

The neural network trained itself to play worse than my simple fallback heuristic.

This is a story about democratizing AI research—and about building the wrong thing really well.

What 6 weeks with Claude Code taught me:"

### Hook C: The Numbers With Commentary

"341 modules. 99.5% test coverage. 132 daemons. 41-node GPU cluster.

Built by someone who can't write a for loop.

Users: 0

Here's what 6 weeks of 'vibe coding' actually looks like:"

### Hook D: The Honest One

"I recreated AlphaZero for a game with zero users.

Then the AI started getting worse instead of better.

This is a cautionary tale about building impressive infrastructure before asking 'should I?'

What 6 weeks with Claude Code taught me:"

### Hook E: The Technical Irony

"I built 11 recovery daemons to recover from the failures of 121 other daemons.

Then discovered the main training loop had been broken for 2 weeks and none of them noticed.

What 6 weeks of AI-assisted development actually looks like:"

### Hook F: The Game Design Angle

"I designed a board game specifically so humans could stay competitive with AI.

Then I built AlphaZero to train on it.

The AI trained itself to play worse than my backup heuristic.

Maybe humans will stay competitive after all?

What 6 weeks recreating DeepMind's pipeline looks like:"

### Hook G: The Research Thesis

"Chess and Go? AI crushed humans years ago.

So I designed a new game with:

- Million+ choices per turn
- 3-4 player coalition dynamics
- Three interacting victory paths

Then I recreated AlphaZero to train on it.

One non-programmer. 6 weeks. Claude Code.

What happened:"

---

## Suggested Images/Screenshots

1. Terminal showing cluster status with all the daemon counts
2. Elo chart showing the regression (1188 → 1141)
3. The anomaly detector log flagging normal values as "10 consecutive anomalies"
4. AWS Cost Explorer showing mysterious charges
5. The game itself (so people know there's actually a product)
6. CLAUDE.md file showing context management
7. Code snippet showing the NPZ sync daemon that wasn't syncing

---

## Timing Suggestions

- Post during US morning (9-11am ET) or lunch (12-1pm ET)
- Tuesday-Thursday typically best engagement
- Consider posting architecture doc to HN same day for cross-traffic
- The self-deprecating angle should play well—lean into the failures
