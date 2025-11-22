# RingRift Documentation Index

**Start Here** for a guide to the project's documentation and structure.

## 🚀 Quick Links

- **Setup & Run:** [QUICKSTART.md](../QUICKSTART.md) - How to install and run the project.
- **Current Status:** [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md) - What works, what doesn't, and verified code status.
- **Roadmap:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md) - Future plans and milestones.

## 📖 Rules & Design

- **Complete Rules:** [ringrift_complete_rules.md](../ringrift_complete_rules.md) - The authoritative rulebook.
- **Compact Rules:** [ringrift_compact_rules.md](../ringrift_compact_rules.md) - Implementation-focused summary.
- **Known Issues:** [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Current bugs and gaps.

## 🏗️ Project Structure

RingRift is a monorepo-style project with three main components:

1.  **Backend (`src/server/`)**: Node.js/Express/TypeScript. Handles game state, WebSockets, and API.
    - **Engine:** `src/server/game/` (GameEngine, RuleEngine).
2.  **Frontend (`src/client/`)**: React/TypeScript/Vite.
    - **Sandbox:** `src/client/sandbox/` (Client-side engine for testing/prototyping).
3.  **AI Service (`ai-service/`)**: Python/FastAPI.
    - Provides AI moves and heuristics.

## 🧪 Testing

- **Guide:** [tests/README.md](../tests/README.md)
- **Parity:** We maintain parity between the Backend Engine and the Client Sandbox Engine. See `tests/unit/Backend_vs_Sandbox.*` for parity tests.

## 🤝 Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Guidelines for contributing code.
