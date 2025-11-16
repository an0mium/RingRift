# RingRift Testing Guide

## Overview

This directory contains the comprehensive testing framework for RingRift. The testing infrastructure uses **Jest** with **TypeScript** support via ts-jest.

## Directory Structure

```
tests/
├── README.md                 # This file - testing documentation
├── setup.ts                  # Jest setup - runs AFTER test framework
├── setup-env.ts              # Jest env setup (dotenv, timers, etc.)
├── test-environment.js       # Custom Jest environment (fixes localStorage)
├── utils/
│   └── fixtures.ts          # Test utilities and fixture creators
└── unit/
    ├── BoardManager.*.test.ts                 # Board geometry, lines, territory
    ├── GameEngine.*.test.ts                   # Core rules, chain capture, choices
    ├── ClientSandboxEngine.*.test.ts          # Client-local sandbox engine: movement, captures, lines, territory, victory
    ├── AIEngine.*.test.ts                     # AI service client + heuristics
    ├── WebSocket*.test.ts                     # WebSocket & PlayerInteractionManager flows
    └── ...                                    # Additional focused rule/interaction suites
```

## Running Tests

### Basic Commands

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run tests with coverage in watch mode
npm run test:coverage:watch

# Run tests for CI/CD (optimized)
npm run test:ci

# Run only unit tests
npm run test:unit

# Run only integration tests
npm run test:integration

# Run tests with verbose output
npm run test:verbose

# Run only the client-local sandbox engine suites
npm test -- ClientSandboxEngine

# Run only GameEngine territory/region tests
npm test -- GameEngine.territoryDisconnection
```

## Test Configuration

### Jest Configuration (`jest.config.js`)

- **Test Environment**: Custom Node environment with localStorage mock
- **Coverage Target**: 80% for branches, functions, lines, statements
- **Test Match Patterns**: `**/*.test.ts`, `**/*.spec.ts`
- **Coverage Directory**: `coverage/` (gitignored)
- **Timeout**: 10 seconds per test

### TypeScript Support

Tests are written in TypeScript and compiled via ts-jest. No separate compilation step needed.

### Path Aliases

The following path aliases are configured for imports:

- `@/` → `src/`
- `@shared/` → `src/shared/`
- `@server/` → `src/server/`
- `@client/` → `src/client/`

## Test Utilities (`tests/utils/fixtures.ts`)

### Board Creation

```typescript
import { createTestBoard } from '../utils/fixtures';

// Create square 8x8 board
const board = createTestBoard('square8');

// Create square 19x19 board
const board = createTestBoard('square19');

// Create hexagonal board
const board = createTestBoard('hexagonal');
```

### Player Creation

```typescript
import { createTestPlayer } from '../utils/fixtures';

// Create default player
const player = createTestPlayer(1);

// Create player with overrides
const player = createTestPlayer(2, { 
  ringsInHand: 10,
  eliminatedRings: 5 
});
```

### Game State Creation

```typescript
import { createTestGameState } from '../utils/fixtures';

// Create default game state
const gameState = createTestGameState();

// Create with custom board type
const gameState = createTestGameState({ boardType: 'hexagonal' });
```

### Board Manipulation

```typescript
import { addStack, addMarker, addCollapsedSpace, pos } from '../utils/fixtures';

// Add a stack
addStack(board, pos(3, 3), playerNumber, height);

// Add a marker
addMarker(board, pos(2, 2), playerNumber);

// Add collapsed space
addCollapsedSpace(board, pos(5, 5), playerNumber);

// Create a line of markers
createMarkerLine(board, pos(0, 0), { dx: 1, dy: 0 }, length, player);
```

### Position Helpers

```typescript
import { pos, posStr } from '../utils/fixtures';

// Create square board position
const position = pos(3, 3);

// Create hexagonal position
const position = pos(0, 0, 0);

// Convert to string
const key = posStr(3, 3); // "3,3"
const hexKey = posStr(0, 0, 0); // "0,0,0"
```

### Assertions

```typescript
import { 
  assertPositionHasStack,
  assertPositionHasMarker,
  assertPositionCollapsed 
} from '../utils/fixtures';

// Assert stack exists with optional player check
assertPositionHasStack(board, pos(3, 3), expectedPlayer);

// Assert marker exists
assertPositionHasMarker(board, pos(2, 2), expectedPlayer);

// Assert space is collapsed
assertPositionCollapsed(board, pos(5, 5), expectedPlayer);
```

### Constants

```typescript
import { 
  BOARD_CONFIGS, 
  SQUARE_POSITIONS, 
  HEX_POSITIONS, 
  GAME_PHASES 
} from '../utils/fixtures';

// Board configurations
const config = BOARD_CONFIGS.square8;
// { type: 'square8', size: 8, ringsPerPlayer: 18, minLineLength: 4, ... }

// Common positions
const center = SQUARE_POSITIONS.center8;
const hexCenter = HEX_POSITIONS.center;

// All game phases
GAME_PHASES.forEach(phase => { /* ... */ });
```

## Writing Tests

### Basic Test Structure

```typescript
import { createTestBoard, addStack, pos } from '../utils/fixtures';

describe('Feature Name', () => {
  let board: ReturnType<typeof createTestBoard>;

  beforeEach(() => {
    board = createTestBoard('square8');
  });

  describe('Specific Functionality', () => {
    it('should do something specific', () => {
      // Arrange
      addStack(board, pos(3, 3), 1);
      
      // Act
      const result = someFunction(board);
      
      // Assert
      expect(result).toBe(expectedValue);
    });
  });
});
```

### Testing All Board Types

```typescript
import { BOARD_CONFIGS } from '../utils/fixtures';

describe.each([
  ['square8', BOARD_CONFIGS.square8],
  ['square19', BOARD_CONFIGS.square19],
  ['hexagonal', BOARD_CONFIGS.hexagonal],
])('Feature on %s board', (boardType, config) => {
  it('should work correctly', () => {
    const board = createTestBoard(config.type);
    // Test logic...
  });
});
```

## Coverage Reports

After running `npm run test:coverage`, coverage reports are generated in:

- **Terminal**: Text summary
- **HTML**: `coverage/lcov-report/index.html` (open in browser)
- **LCOV**: `coverage/lcov.info` (for CI tools)
- **JSON**: `coverage/coverage-final.json`

## CI/CD Integration

The `npm run test:ci` command is optimized for CI/CD pipelines:

- Runs in CI mode (no watch)
- Generates coverage reports
- Limits workers for resource efficiency
- Fails if coverage thresholds not met

## Best Practices

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Follow the AAA pattern for test structure
3. **One Assertion**: Focus each test on one specific behavior
4. **Use Fixtures**: Leverage test utilities for consistent test data
5. **Mock Carefully**: Only mock what's necessary for isolation
6. **Clean Up**: Tests clean up automatically via `afterEach` hooks
7. **Coverage**: Aim for 80%+ coverage on all metrics

## Debugging Tests

```bash
# Run single test file
npm test -- tests/unit/board.test.ts

# Run tests matching pattern
npm test -- --testNamePattern="createTestBoard"

# Run in debug mode
node --inspect-brk node_modules/.bin/jest --runInBand
```

## Common Issues

### localStorage SecurityError

Fixed by using custom test environment (`tests/test-environment.js`). If you encounter issues, ensure `testEnvironment` in `jest.config.js` points to the custom environment.

### TypeScript Errors

Ensure your test files are included in `tsconfig.json` or create a separate `tsconfig.test.json` if needed.

### Coverage Not Collecting

Check `collectCoverageFrom` patterns in `jest.config.js` to ensure your source files are included.

## Next Steps

See `TODO.md` Phase 2 for comprehensive test coverage tasks:

- Unit tests for all BoardManager, GameEngine, RuleEngine methods
- Integration tests for complete game flows
- Scenario tests from rules document
- Edge case coverage

---

**Last Updated**: November 13, 2025  
**Framework**: Jest 29.7.0 + ts-jest 29.1.1
