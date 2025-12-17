# RingRift Rules Quick Reference Diagrams

> **Purpose:** Visual quick-reference diagrams for the most commonly misunderstood rules.
> These diagrams are designed to be embedded in the game UI, tutorial, and documentation.

---

## 1. Marker Interactions

Understanding what happens when your stack interacts with markers during movement.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MARKER INTERACTIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PASSING OVER OPPONENT'S MARKER → Flip to your color         │
│                                                                  │
│     [You] ───────→ (Their marker) ───────→ [Destination]        │
│                          ↓                                       │
│                    (Your marker)                                 │
│                                                                  │
│     Result: Their marker becomes your marker                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2. PASSING OVER YOUR OWN MARKER → Collapse to territory        │
│                                                                  │
│     [You] ───────→ (Your marker) ───────→ [Destination]         │
│                          ↓                                       │
│                   [Territory ■]                                  │
│                                                                  │
│     Result: Marker collapses into permanent territory space      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  3. LANDING ON ANY MARKER → Marker removed + lose top ring      │
│                                                                  │
│     [Stack: 3 rings] ───────→ lands on (any marker)             │
│                                      ↓                           │
│     [Stack: 2 rings] + marker removed + 1 ring eliminated        │
│                                                                  │
│     Result: Marker destroyed, you pay with your top ring         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Passing OVER markers is beneficial; LANDING on them costs you a ring.

---

## 2. Line Processing Options

When you form a line of 4+ markers in a row, you process it for territory.

```
┌─────────────────────────────────────────────────────────────────┐
│              LINE PROCESSING (4+ markers in a row)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXACT LENGTH (exactly 4 markers on 8×8, or required length):   │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│     ○ ─ ○ ─ ○ ─ ○      (4 markers in a line)                   │
│           ↓                                                      │
│     ■ ─ ■ ─ ■ ─ ■      (All become territory)                  │
│                                                                  │
│     Cost: Eliminate ONE RING from any stack you control          │
│                                                                  │
│     NO CHOICE - this happens automatically                       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OVERLENGTH (more than required, e.g., 5+ markers):             │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│     ○ ─ ○ ─ ○ ─ ○ ─ ○      (5 markers in a line)               │
│                                                                  │
│     ┌────────────────────────┬────────────────────────┐         │
│     │       OPTION 1         │       OPTION 2         │         │
│     │    (Max Territory)     │    (Save Your Ring)    │         │
│     ├────────────────────────┼────────────────────────┤         │
│     │                        │                        │         │
│     │  ■ ─ ■ ─ ■ ─ ■ ─ ■    │  ■ ─ ■ ─ ■ ─ ■ ─ ○    │         │
│     │                        │                        │         │
│     │  Collapse ALL markers  │  Collapse only 4      │         │
│     │  (5 territory spaces)  │  (4 territory spaces) │         │
│     │                        │                        │         │
│     │  Cost: 1 ring          │  Cost: NONE           │         │
│     │                        │                        │         │
│     └────────────────────────┴────────────────────────┘         │
│                                                                  │
│     CHOICE: More territory vs. keep your ring                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Overlength lines give you a choice - trade rings for territory or play conservatively.

---

## 3. Territory Disconnection

When a region becomes isolated, it can be processed for territory.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TERRITORY DISCONNECTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A region is DISCONNECTED when it's physically cut off from     │
│  the rest of the board by some combination of:                   │
│                                                                  │
│    • Board edges                                                 │
│    • Already-collapsed territory spaces                          │
│    • Markers of a SINGLE border color                           │
│                                                                  │
│  Example: Blue creates a disconnected region                     │
│  ─────────────────────────────────────────────                  │
│                                                                  │
│     ┌───┬───┬───┬───┬───┐                                       │
│     │ ■ │ ■ │ ■ │ ■ │   │   ■ = Collapsed territory (barrier)  │
│     ├───┼───┼───┼───┼───┤   ● = Blue markers (border)          │
│     │ ● │ · │ · │ ■ │   │   · = Empty cells (the region)       │
│     ├───┼───┼───┼───┼───┤   ▲ = Red stack (inside region)      │
│     │ ● │ · │ ▲ │ ● │   │                                       │
│     ├───┼───┼───┼───┼───┤   The 4 empty cells + Red stack      │
│     │   │ ● │ ● │ ● │   │   form a disconnected region         │
│     └───┴───┴───┴───┴───┘   bordered only by Blue              │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHO CAN PROCESS IT?                                            │
│  ─────────────────────                                          │
│                                                                  │
│  The player whose markers BORDER the region (Blue in example)    │
│  can choose to process it.                                       │
│                                                                  │
│  PROCESSING COST:                                               │
│  ────────────────                                               │
│                                                                  │
│  • All stacks/markers INSIDE the region are eliminated           │
│  • The spaces become territory for the processing player         │
│  • You must pay: eliminate your ENTIRE CAP from a stack          │
│    you control OUTSIDE the region                                │
│                                                                  │
│  IMPORTANT: You need at least one stack outside to pay!          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Territory disconnection is powerful but costs your entire cap from another stack.

---

## 4. Elimination Contexts Summary

Different situations require different elimination costs.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ELIMINATION CONTEXTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context              │ What You Lose    │ From Which Stack     │
│  ─────────────────────┼──────────────────┼─────────────────────│
│                       │                  │                      │
│  LINE PROCESSING      │ ONE ring         │ Any stack you        │
│  (forming a line)     │ from top         │ control              │
│                       │                  │                      │
│  ─────────────────────┼──────────────────┼─────────────────────│
│                       │                  │                      │
│  TERRITORY PROCESSING │ ENTIRE cap       │ Any stack you        │
│  (disconnected region)│ (all your top    │ control OUTSIDE      │
│                       │  rings)          │ the region           │
│                       │                  │                      │
│  ─────────────────────┼──────────────────┼─────────────────────│
│                       │                  │                      │
│  FORCED ELIMINATION   │ ENTIRE cap       │ Any stack you        │
│  (no other moves)     │                  │ control              │
│                       │                  │                      │
│  ─────────────────────┼──────────────────┼─────────────────────│
│                       │                  │                      │
│  MARKER LANDING       │ ONE ring         │ The stack that       │
│  (landing on marker)  │ from top         │ landed               │
│                       │                  │                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Line processing costs just one ring; territory processing costs your whole cap.

---

## 5. Victory Conditions At-a-Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     VICTORY CONDITIONS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ RING ELIMINATION│  │TERRITORY CONTROL│  │LAST PLAYER     │ │
│  │                 │  │                 │  │STANDING        │ │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤ │
│  │                 │  │                 │  │                 │ │
│  │ Eliminate rings │  │ Control at least│  │ Be the only    │ │
│  │ equal to the    │  │ your fair share │  │ player who can │ │
│  │ victory         │  │ of the board    │  │ still make     │ │
│  │ threshold       │  │ (1/N) AND more  │  │ meaningful     │ │
│  │                 │  │ than all others │  │ moves          │ │
│  │ (varies by      │  │ combined        │  │                 │ │
│  │  board/players) │  │                 │  │                 │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  First player to achieve ANY ONE of these wins immediately!      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Notes

These diagrams can be:

1. Embedded in the tutorial mode
2. Shown as tooltips in the game UI
3. Linked from the teaching overlay
4. Printed as a quick reference card

For the interactive game UI, consider converting these to SVG or React components
for better presentation and potential animation.
