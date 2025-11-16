# RingRift - Multiplayer Strategy Game

⚠️ **PROJECT STATUS: CORE LOGIC ~75% COMPLETE – BACKEND PLAY AND AI TURNS WORK, UI & TESTING STILL EARLY** ⚠️

> **Important:** Core game mechanics are largely implemented (~75%), and there is now a **playable backend game flow**: the server’s `GameEngine` drives rules, WebSocket-backed games use it as the source of truth, the React client renders boards and submits moves, and AI opponents can make moves via the Python AI service. In addition, a **client-local sandbox engine** (`ClientSandboxEngine`) powers the `/sandbox` route with strong rules parity and dedicated Jest suites for movement, captures, lines, territory, and victory checks. However, the UI is still minimal, end-to-end UX is rough, and test coverage is low. See [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) for a verified breakdown.

A web-based multiplayer implementation of the RingRift strategy game supporting 2-4 players with flexible human/AI combinations across multiple board configurations.

## 📋 Current Status

**Last Updated:** November 14, 2025  
**Verification:** Code-verified assessment  
**Overall Progress:** 58% Complete (strong foundation, critical gaps remain)

### ✅ What's Working (75% of Core Logic)
- ✅ Project infrastructure (Docker, database, Redis, WebSocket)
- ✅ TypeScript type system and architecture (100%)
- ✅ Comprehensive game rules documentation
- ✅ Server and client scaffolding
- ✅ **Marker system** - Placement, flipping, collapsing fully functional
- ✅ **Movement validation** - Distance rules, path checking working
- ✅ **Basic captures** - Single captures work correctly
- ✅ **Line detection** - All board types (8x8, 19x19, hexagonal)
- ✅ **Territory disconnection** - Detection and processing implemented
- ✅ **Phase transitions** - Correct game flow through all phases
- ✅ **Player state tracking** - Ring counts, eliminations, territory
- ✅ **Hexagonal board support** - Full 331-space board validated
- ✅ **Client-local sandbox engine** - `/sandbox` uses `ClientSandboxEngine` plus `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxLinesEngine.ts`, `sandboxTerritoryEngine.ts`, and `sandboxVictory.ts` to run full games in the browser (movement, captures, lines, territory, and ring/territory victories) with dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts`.

### ⚠️ Critical Gaps (Blocks Production-Quality Play)
- ⚠️ **Player choice system is implemented but not yet deeply battle-tested** – Shared types and `PlayerInteractionManager` exist and GameEngine now uses them for line order, line reward, ring elimination, region order, and capture direction. `WebSocketInteractionHandler`, `GameContext`, and `ChoiceDialog` wire these choices to human clients for backend-driven games, and `AIInteractionHandler` answers choices for AI players via local heuristics. What’s missing is broad scenario coverage (all FAQ/rules examples), polished UX around errors/timeouts, and – optionally – AI-service–backed choice decisions.
- ⚠️ **Chain captures enforced engine-side; more edge-case tests still needed** – GameEngine maintains internal chain-capture state and uses `CaptureDirectionChoice` via `PlayerInteractionManager` to drive mandatory continuation when multiple follow-up captures exist. Core behaviour is covered by focused unit/integration tests, but additional rule/FAQ scenarios (e.g. complex 180° and cyclic patterns) and full UI/AI flows still need to be exercised.
- ⚠️ **UI is functional but minimal** – Board rendering, a local sandbox, and backend game mode exist for 8x8, 19x19, and hex boards. Backend games now support “click source, click highlighted destination” moves and server-driven choices, and AI opponents can take turns. However, the HUD, polish, and game lifecycle UX are still early.
- ❌ **Limited testing** – Dedicated Jest suites now cover the client-local sandbox engine (movement, captures, lines, territory, victory) and several backend engine/interaction paths, but overall coverage is still low and there is no comprehensive scenario suite derived from the rules/FAQ.
- ⚠️ **AI service integration is move- and choice-focused but still evolving** – The Python AI microservice is integrated into the turn loop via `AIEngine`/`AIServiceClient` and `WebSocketServer.maybePerformAITurn`, so AI players can select moves in backend games. The service is also used for several PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) behind `globalAIEngine`/`AIInteractionHandler`, with remaining choices currently answered via local heuristics. Higher-difficulty tactical behaviour still depends on future work.

### 🎯 What This Means
**Can Do (today):**
- Create games via the HTTP API and from the React lobby (including AI opponent configuration).
- Play backend-driven games end-to-end using the React client (BoardView + GamePage) with click-to-move and server-validated moves.
- Have AI opponents take turns in backend games via the Python AI service, using the unified `AIProfile` / `aiOpponents` pipeline.
- Process lines and territory disconnection, forced elimination, and hex boards through the shared GameEngine.
- Track full game state (phases, players, rings, territory, timers) and broadcast updates over WebSockets.
- Run full, rules-complete games in the `/sandbox` route using the client-local `ClientSandboxEngine` with simple random-choice AI for all PlayerChoices, reusing the same BoardView/ChoiceDialog/VictoryModal patterns as backend games.

**Cannot Do (yet):**
- Rely on tests for full rule coverage (scenario/edge-case tests and coverage are still incomplete).
- Guarantee every chain capture and PlayerChoice edge case from the rules/FAQ is battle-tested and bug-free.
- Enjoy a fully polished UX (HUD, timers, post-game flows, and lobby/matchmaking are still basic).
- Use the AI service for PlayerChoice decisions (choices are answered via local heuristics only).
- Play production-grade multiplayer with lobbies, matchmaking, reconnection, and spectators.

### 📊 Component Status
| Component | Status | Completion |
|-----------|--------|-----------|
| Type System | ✅ Complete | 100% |
| Board Manager | ✅ Complete | 90% |
| Game Engine | ⚠️ Partial | 75% |
| Rule Engine | ⚠️ Partial | 60% |
| Frontend UI | ⚠️ Basic board & choice UI + client-local sandbox engine | 30% |
| AI Integration | ⚠️ Moves + some choices service-backed | 60% |
| Testing | ⚠️ Growing but incomplete | 10% |
| Multiplayer | ⚠️ Basic backend play, no full lobby yet | 30% |

**For complete assessment, see [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)**  
**For detailed issues, see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md)**  
**For roadmap, see [TODO.md](./TODO.md)**

---

## 🎯 Overview

RingRift is a sophisticated turn-based strategy game featuring:
- **Multiple Board Types**: 8x8 square, 19x19 square, and hexagonal layouts
- **Flexible Player Support**: 2-4 players with human/AI combinations
- **Real-time Multiplayer**: WebSocket-based live gameplay
- **Spectator Mode**: Watch games in progress
- **Rating System**: ELO-based player rankings
- **Time Controls**: Configurable game timing
- **Cross-platform**: Web-based for universal accessibility

## 🏗️ Architecture

### Technology Stack

#### Backend
- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js with comprehensive middleware
- **Database**: PostgreSQL with Prisma ORM
- **Real-time**: Socket.IO for WebSocket communication
- **Caching**: Redis for session management and game state
- **Authentication**: JWT-based with bcrypt password hashing
- **Validation**: Zod schemas for type-safe data validation
- **Logging**: Winston for structured logging
- **Security**: Helmet, CORS, rate limiting

#### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Routing**: React Router for SPA navigation
- **State Management**: React Query for server state, Context API for client state
- **Styling**: Tailwind CSS for utility-first styling
- **WebSocket**: Socket.IO client for real-time communication
- **HTTP Client**: Axios with interceptors for API communication

#### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for development
- **Database**: PostgreSQL with connection pooling
- **Caching**: Redis for high-performance data access
- **Environment**: Environment-based configuration

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │    │   Express API   │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ • Game UI       │◄──►│ • REST Routes   │◄──►│ • Game Data     │
│ • Real-time     │    │ • WebSocket     │    │ • User Profiles │
│ • State Mgmt    │    │ • Game Engine   │    │ • Match History │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │      Redis      │              │
         └──────────────►│                 │◄─────────────┘
                        │ • Sessions      │
                        │ • Game Cache    │
                        │ • Rate Limiting │
                        └─────────────────┘
```

## ⚠️ Development Notice

**This application is not yet production-ready.** The current codebase now includes a working backend game loop and a minimal but functional React client for playing backend-driven games, but the overall UX, multiplayer flows, and test coverage are still incomplete. In its current state, the project is best suited for engine/AI development, rules validation, and early playtesting rather than public release.

The codebase currently provides:
- Infrastructure setup and configuration (Docker, database, Redis, WebSockets, logging, authentication)
- Fully typed shared game state and rules data structures
- A largely implemented GameEngine + BoardManager + RuleEngine for all board types
- A Python AI service wired into backend AI turns via `AIEngine` / `AIServiceClient`
- A minimal React client (LobbyPage + GamePage + BoardView + ChoiceDialog) that can:
  - Create backend games (including AI opponents) via the HTTP API and lobby UI
  - Connect to backend games over WebSockets and play via click-to-move
  - Surface server-driven PlayerChoices for humans (e.g. line rewards, ring elimination)
- A growing but still limited Jest test suite around core rules, interaction flows, AI turns, and territory disconnection

**To contribute or continue development, please review:**
1. [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) - Factual, code-verified analysis of the current state
2. [ARCHITECTURE_ASSESSMENT.md](./ARCHITECTURE_ASSESSMENT.md) - Architecture and refactoring axes (supersedes older codebase evaluation docs)
3. [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Specific bugs, missing features, and prioritization
4. [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) - Phased implementation plan and milestones
5. [CONTRIBUTING.md](./CONTRIBUTING.md) - Development priorities and guidelines

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm 9+
- Docker and Docker Compose
- PostgreSQL 14+ (or use Docker)
- Redis 6+ (or use Docker)

### Development Setup

1. **Clone and Install**
```bash
git clone <repository-url>
cd ringrift
npm install
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Database Setup**
```bash
# Start services with Docker
docker-compose up -d postgres redis

# Setup database
npm run db:migrate
npm run db:generate
```

4. **Start Development**
```bash
# Start both frontend and backend
npm run dev

# Or start individually
npm run dev:server  # Backend on :5000
npm run dev:client  # Frontend on :3000
```

### Production Deployment

```bash
# Build application
npm run build

# Start with Docker
docker-compose up -d

# Or manual deployment
npm start
```

## 🎮 Game Features

### Core Gameplay
- **Ring Placement**: Strategic positioning of rings on the board
- **Movement Phase**: Tactical ring repositioning
- **Row Formation**: Create rows of markers to remove opponent rings
- **Victory Conditions**: Remove required number of opponent rings

### Board Configurations
- **8x8 Square**: Compact tactical gameplay
- **19x19 Square**: Extended strategic depth
- **Hexagonal**: Unique geometric challenges

### Multiplayer Features *(planned/partially implemented)*
- **Real-time Synchronization**: Instant move updates
- **Spectator Mode**: Watch games with live commentary
- **Chat System**: In-game communication
- **Reconnection**: Seamless game resumption
- **Time Controls**: Blitz, rapid, and classical formats

### AI Integration *(planned/partially implemented)*
- **Difficulty Levels**: 1-10 skill ratings
- **Smart Opponents**: Strategic decision making
- **Mixed Games**: Human-AI combinations
- **Learning Algorithms**: Adaptive gameplay

## 🔧 API Documentation

### Authentication Endpoints
```
POST /api/auth/register    # User registration
POST /api/auth/login       # User authentication
GET  /api/auth/profile     # Get user profile
PUT  /api/auth/profile     # Update user profile
```

### Game Management
```
GET    /api/games          # List games
POST   /api/games          # Create new game
GET    /api/games/:id      # Get game details
POST   /api/games/:id/join # Join game
POST   /api/games/:id/leave # Leave game
POST   /api/games/:id/moves # Make move
```

### User Operations
```
GET /api/users             # List users
GET /api/users/:id         # Get user details
GET /api/users/leaderboard # Rating leaderboard
```

### WebSocket Events
```
join_game      # Join game room
leave_game     # Leave game room
player_move    # Send move
chat_message   # Send chat
game_update    # Receive game state
player_joined  # Player joined notification
player_left    # Player left notification
```

## 🛡️ Security Features

### Authentication & Authorization
- JWT token-based authentication
- Secure password hashing with bcrypt
- Role-based access control
- Session management with Redis

### API Security
- Rate limiting per endpoint
- CORS configuration
- Helmet security headers
- Input validation with Zod schemas
- SQL injection prevention with Prisma

### Game Security
- Move validation on server
- Anti-cheat mechanisms
- Secure WebSocket connections
- Game state integrity checks

## 📊 Performance Optimizations

### Backend Optimizations
- Database connection pooling
- Redis caching for frequent queries
- Efficient game state serialization
- Optimized database indexes
- Background job processing

### Frontend Optimizations
- Code splitting with React.lazy
- Memoization for expensive calculations
- Virtual scrolling for large lists
- Optimistic UI updates
- Service worker for offline capability

### Real-time Performance
- WebSocket connection pooling
- Efficient event broadcasting
- Delta updates for game state
- Compression for large payloads
- Heartbeat monitoring

## 🧪 Testing Strategy

> Note: This section describes the target testing setup. As of now, only basic unit tests exist; see CURRENT_STATE_ASSESSMENT.md for up-to-date coverage details.

### Backend Testing
```bash
npm test                   # Run all tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
```

### Frontend Testing
```bash
npm run test:client       # Client tests
npm run test:e2e          # End-to-end tests
```

### Test Coverage
- Unit tests for game logic
- Integration tests for API endpoints
- WebSocket connection testing
- UI component testing
- End-to-end gameplay scenarios

## 📈 Monitoring & Analytics *(future/partially implemented)*

### Application Monitoring
- Structured logging with Winston
- Error tracking and alerting
- Performance metrics collection
- Database query monitoring
- WebSocket connection analytics

### Game Analytics
- Player behavior tracking
- Game duration statistics
- Move pattern analysis
- Rating system metrics
- User engagement data

## 🔄 Development Workflow

### Code Quality
- TypeScript for type safety
- ESLint for code standards
- Prettier for formatting
- Husky for git hooks
- Conventional commits

### CI/CD Pipeline
- Automated testing on PR
- Code quality checks
- Security vulnerability scanning
- Automated deployment
- Database migration handling

## 📚 Additional Resources

### Game Rules
- Complete rule documentation in `ringrift_complete_rules.md`
- Interactive tutorial system
- Strategy guides and tips
- Video demonstrations

### Development Guides
- Architecture decision records
- API integration examples
- WebSocket implementation guide
- Database schema documentation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [Wiki](link-to-wiki)
- Issues: [GitHub Issues](link-to-issues)
- Discussions: [GitHub Discussions](link-to-discussions)
- Email: support@ringrift.com

---

Built with ❤️ by the RingRift Team
