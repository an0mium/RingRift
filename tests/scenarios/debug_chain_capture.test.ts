/**
 * Debug test for chain capture - temporary
 */
import { GameEngine } from '../../src/server/game/GameEngine';
import {
  Position,
  Player,
  TimeControl,
  RingStack,
  GameState,
  positionToString,
} from '../../src/shared/types/game';
import { createChainCapture4Fixture } from '../fixtures/chainCaptureExtendedFixture';

describe('Debug chain capture', () => {
  beforeAll(() => {
    jest.useFakeTimers();
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  test('debug chain capture positions', async () => {
    const fixture = createChainCapture4Fixture();
    const engine = new GameEngine('debug-test', 'square8', basePlayers, timeControl, false);

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager;
    const gameState = engineAny.gameState;

    // Clear existing stacks
    gameState.board.stacks.clear();

    // Set up stacks from fixture
    for (const [, stack] of fixture.gameState.board.stacks) {
      boardManager.setStack(stack.position, stack, gameState.board);
    }

    // Force movement phase
    gameState.currentPhase = 'movement';
    gameState.currentPlayer = 1;

    console.log('Initial stacks:');
    for (const [key, stack] of gameState.board.stacks as Map<string, RingStack>) {
      console.log(`  ${key}: player=${stack.controllingPlayer}, height=${stack.stackHeight}`);
    }

    // Start the chain with the initial overtaking_capture
    const initialMove = fixture.initialMove;
    const step1 = await engine.makeMove({
      player: initialMove.player,
      type: 'overtaking_capture',
      from: initialMove.from,
      captureTarget: initialMove.captureTarget,
      to: initialMove.to,
    } as any);

    console.log('\nInitial move result:', step1.success);
    console.log('Phase after initial:', gameState.currentPhase);
    console.log('Stacks after initial:');
    for (const [key, stack] of gameState.board.stacks as Map<string, RingStack>) {
      console.log(`  ${key}: player=${stack.controllingPlayer}, height=${stack.stackHeight}`);
    }

    // Resolve chain
    let step = 0;
    while (gameState.currentPhase === 'chain_capture' && step < 10) {
      step++;
      const moves = engine
        .getValidMoves(1)
        .filter((m: any) => m.type === 'continue_capture_segment');
      console.log(`\nStep ${step}: ${moves.length} chain moves available`);

      if (moves.length === 0) {
        console.log('No chain moves, breaking');
        break;
      }

      const next = moves[0];
      console.log(
        `  Executing: from=${positionToString(next.from!)} target=${positionToString(next.captureTarget!)} to=${positionToString(next.to!)}`
      );

      await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      console.log(`  Phase after: ${gameState.currentPhase}`);
      console.log('  Stacks:');
      for (const [key, stack] of gameState.board.stacks as Map<string, RingStack>) {
        console.log(`    ${key}: player=${stack.controllingPlayer}, height=${stack.stackHeight}`);
      }
    }

    console.log('\n=== FINAL STATE ===');
    console.log('Final phase:', gameState.currentPhase);
    console.log('Final stacks:');
    for (const [key, stack] of gameState.board.stacks as Map<string, RingStack>) {
      console.log(`  ${key}: player=${stack.controllingPlayer}, height=${stack.stackHeight}`);
    }

    console.log('\nExpected final position:', positionToString(fixture.expectedFinalPosition));
    console.log('Expected final height:', fixture.expectedFinalHeight);

    // This should pass
    expect(true).toBe(true);
  });
});
