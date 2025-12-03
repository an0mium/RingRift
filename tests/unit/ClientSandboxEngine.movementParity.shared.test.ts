import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { GameState, BoardType, Position, RingStack, Move } from '../../src/shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  applySimpleMovement as applySimpleMovementAggregate,
  hashGameState,
} from '../../src/shared/engine';
import type { PlayerChoiceResponseFor, CaptureDirectionChoice } from '../../src/shared/types/game';
import {
  enumerateMovementTargets,
  validateMovement,
} from '../../src/shared/engine/aggregates/MovementAggregate';
import type { MoveStackAction } from '../../src/shared/engine/types';

describe('ClientSandboxEngine movement parity with shared MovementAggregate', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually trigger PlayerChoices in practice,
      // but we provide a trivial handler to satisfy the constructor and keep
      // types aligned with other sandbox tests.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function createHexEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // As above, choices are not exercised in this test; always pick first option.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function createSquare19Engine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: 'square19',
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function makeStack(
    playerNumber: number,
    height: number,
    position: Position,
    state: GameState
  ): void {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    state.board.stacks.set(positionToString(position), stack);
  }

  it('simple non-capture move via sandbox movement engine matches aggregate outcome', async () => {
    const engine = createEngine();
    // Now uses orchestrator-backed engine which delegates to shared MovementAggregate

    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Place a single two-ring stack at (2,2) for player 1.
    const origin: Position = { x: 2, y: 2 };
    makeStack(1, 2, origin, internalState);

    // Destination two steps to the east with an empty path and landing cell.
    const dest: Position = { x: 4, y: 2 };

    const config = BOARD_CONFIGS[boardType];
    expect(dest.x).toBeGreaterThanOrEqual(0);
    expect(dest.x).toBeLessThan(config.size);
    expect(dest.y).toBeGreaterThanOrEqual(0);
    expect(dest.y).toBeLessThan(config.size);
    expect(board.stacks.has(positionToString(dest))).toBe(false);

    // Shared-core path: start from a defensive snapshot and apply the aggregate.
    const startingState: GameState = engine.getGameState();
    const coreOutcome = applySimpleMovementAggregate(startingState, {
      from: origin,
      to: dest,
      player: 1,
    });

    // Sandbox/orchestrator path: apply canonical move via applyCanonicalMove
    // which delegates to the orchestrator.
    engineAny.gameState = startingState;

    const moveStackMove = {
      id: 'move-1',
      type: 'move_stack' as const,
      player: 1,
      from: origin,
      to: dest,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    await engine.applyCanonicalMove(moveStackMove);

    const sandboxStateAfter: GameState = engine.getGameState();

    // Compare board-level changes (stacks, markers) which is what the aggregate
    // transforms. The orchestrator also advances the turn, so we compare
    // specific board effects rather than full state hash.
    const coreBoard = coreOutcome.nextState.board;
    const sandboxBoard = sandboxStateAfter.board;

    // Origin should be empty in both
    expect(coreBoard.stacks.has(positionToString(origin))).toBe(false);
    expect(sandboxBoard.stacks.has(positionToString(origin))).toBe(false);

    // Destination should have the stack in both
    const coreDestStack = coreBoard.stacks.get(positionToString(dest));
    const sandboxDestStack = sandboxBoard.stacks.get(positionToString(dest));

    expect(coreDestStack).toBeDefined();
    expect(sandboxDestStack).toBeDefined();
    expect(sandboxDestStack?.controllingPlayer).toBe(coreDestStack?.controllingPlayer);
    expect(sandboxDestStack?.stackHeight).toBe(coreDestStack?.stackHeight);
    expect(sandboxDestStack?.capHeight).toBe(coreDestStack?.capHeight);

    // Markers left behind should match
    const coreMarkerAtOrigin = coreBoard.markers.get(positionToString(origin));
    const sandboxMarkerAtOrigin = sandboxBoard.markers.get(positionToString(origin));
    expect(sandboxMarkerAtOrigin).toEqual(coreMarkerAtOrigin);
  });

  it('movement-created collapsed spaces on hex board update territorySpaces in sandbox', async () => {
    const engine = createHexEngine();
    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Hex board: use cube coordinates with x + y + z = 0.
    const origin: Position = { x: 0, y: 0, z: 0 };
    const dest: Position = { x: 3, y: -3, z: 0 };

    // Height-3 stack so minimum-distance requirement is satisfied.
    const rings = Array(3).fill(1);
    const originStack: RingStack = {
      position: origin,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    board.stacks.set(positionToString(origin), originStack);

    // Place two P1 markers strictly along the movement ray between origin and dest.
    const pathMarkers: Position[] = [
      { x: 1, y: -1, z: 0 },
      { x: 2, y: -2, z: 0 },
    ];
    for (const mPos of pathMarkers) {
      board.markers.set(positionToString(mPos), {
        player: 1,
        position: mPos,
        type: 'regular',
      });
    }

    const startingState: GameState = engine.getGameState();

    const beforeCollapsedForP1 = Array.from(startingState.board.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const beforeP1 = startingState.players.find((p) => p.playerNumber === 1)!;
    const beforeTerritory = beforeP1.territorySpaces;

    // Apply canonical move via orchestrator-backed engine.
    engineAny.gameState = startingState;

    const move: Move = {
      id: 'hex-move-1',
      type: 'move_stack',
      player: 1,
      from: origin,
      to: dest,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    await engine.applyCanonicalMove(move);

    const afterState: GameState = engine.getGameState();

    const afterCollapsedForP1 = Array.from(afterState.board.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const afterP1 = afterState.players.find((p) => p.playerNumber === 1)!;
    const afterTerritory = afterP1.territorySpaces;

    const deltaCollapsed = afterCollapsedForP1 - beforeCollapsedForP1;
    const deltaTerritory = afterTerritory - beforeTerritory;

    expect(deltaCollapsed).toBeGreaterThan(0);
    expect(deltaTerritory).toBe(deltaCollapsed);
  });

  it('square19 movement enumeration near edges matches MovementAggregate and excludes invalid landings (Movement M2)', () => {
    const engine = createSquare19Engine();
    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    const origin: Position = { x: 0, y: 0 };
    // Height-2 stack: minimum distance 2.
    makeStack(1, 2, origin, internalState);

    // Place a blocking stack along the eastward ray to exercise path blocking.
    const blockedPathPos: Position = { x: 1, y: 0 };
    makeStack(2, 1, blockedPathPos, internalState);

    const startingState: GameState = engine.getGameState();

    const aggregateTargets = enumerateMovementTargets(startingState, origin);
    const aggregateKeys = new Set(aggregateTargets.map((p) => positionToString(p)));

    engineAny.gameState = startingState;
    const simpleLandingsRaw = engineAny.enumerateSimpleMovementLandings(1) as Array<{
      fromKey: string;
      to: Position;
    }>;
    const originKey = positionToString(origin);
    const simpleLandings = simpleLandingsRaw
      .filter((m) => m.fromKey === originKey)
      .map((m) => m.to);
    const sandboxKeys = new Set(simpleLandings.map((p) => positionToString(p)));

    const normalize = (arr: Position[]) =>
      arr.map((p) => positionToString(p)).sort((a, b) => a.localeCompare(b));

    expect(normalize(simpleLandings)).toEqual(normalize(aggregateTargets));

    // Explicitly verify that clearly invalid destinations are both un-enumerated
    // and rejected by validateMovement (Movement rows M1–M3).
    const offBoard: Position = { x: -1, y: 0 };
    const blockedDest: Position = { x: 3, y: 0 };

    const offBoardAction: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: origin,
      to: offBoard,
    };
    const blockedAction: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: origin,
      to: blockedDest,
    };

    const offBoardResult = validateMovement(startingState, offBoardAction);
    const blockedResult = validateMovement(startingState, blockedAction);

    expect(offBoardResult.valid).toBe(false);
    expect(blockedResult.valid).toBe(false);

    const offBoardKey = positionToString(offBoard);
    const blockedKey = positionToString(blockedDest);

    expect(aggregateKeys.has(offBoardKey)).toBe(false);
    expect(aggregateKeys.has(blockedKey)).toBe(false);
    expect(sandboxKeys.has(offBoardKey)).toBe(false);
    expect(sandboxKeys.has(blockedKey)).toBe(false);
  });

  it('hexagonal movement enumeration along cube axes matches MovementAggregate and excludes invalid landings (Movement M1/M2)', () => {
    const engine = createHexEngine();
    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    const origin: Position = { x: 0, y: 0, z: 0 };
    // Height-2 stack so minimum distance requirement applies.
    makeStack(1, 2, origin, internalState);

    // Block one of the forward hex directions to test path blocking parity.
    const pathBlock: Position = { x: 1, y: -1, z: 0 };
    makeStack(2, 1, pathBlock, internalState);

    const startingState: GameState = engine.getGameState();

    const aggregateTargets = enumerateMovementTargets(startingState, origin);
    const aggregateKeys = new Set(aggregateTargets.map((p) => positionToString(p)));

    engineAny.gameState = startingState;
    const simpleLandingsRaw = engineAny.enumerateSimpleMovementLandings(1) as Array<{
      fromKey: string;
      to: Position;
    }>;
    const originKey = positionToString(origin);
    const simpleLandings = simpleLandingsRaw
      .filter((m) => m.fromKey === originKey)
      .map((m) => m.to);
    const sandboxKeys = new Set(simpleLandings.map((p) => positionToString(p)));

    const normalize = (arr: Position[]) =>
      arr.map((p) => positionToString(p)).sort((a, b) => a.localeCompare(b));

    expect(normalize(simpleLandings)).toEqual(normalize(aggregateTargets));

    // Off-axis / invalid cube coordinate destination (x+y+z != 0).
    const offAxis: Position = { x: 1, y: 0, z: 0 };
    // Destination whose ray is blocked by the pathBlock stack.
    const blockedDest: Position = { x: 2, y: -2, z: 0 };

    const offAxisAction: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: origin,
      to: offAxis,
    };
    const blockedAction: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: origin,
      to: blockedDest,
    };

    const offAxisResult = validateMovement(startingState, offAxisAction);
    const blockedResult = validateMovement(startingState, blockedAction);

    expect(offAxisResult.valid).toBe(false);
    expect(blockedResult.valid).toBe(false);

    const offAxisKey = positionToString(offAxis);
    const blockedKey = positionToString(blockedDest);

    expect(aggregateKeys.has(offAxisKey)).toBe(false);
    expect(aggregateKeys.has(blockedKey)).toBe(false);
    expect(sandboxKeys.has(offAxisKey)).toBe(false);
    expect(sandboxKeys.has(blockedKey)).toBe(false);
  });
});
