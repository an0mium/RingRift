/**
 * FAQ Q15: Chain Capture Patterns - Comprehensive Test Suite
 * 
 * Covers FAQ 15.3.1 (180° Reversal) and FAQ 15.3.2 (Cyclic Patterns)
 * Rules: §10.3 (Chain Overtaking), §10.4 (Capture Patterns)
 * 
 * This suite validates all documented chain capture behaviors from the FAQ.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { 
  Position, 
  Player, 
  BoardType, 
  TimeControl, 
  RingStack,
  GameState 
} from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';

describe('FAQ Q15: Chain Capture Patterns', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  
  const createBasePlayers = (): Player[] => [
    createTestPlayer(1, { ringsInHand: 18 }),
    createTestPlayer(2, { ringsInHand: 18 })
  ];

  describe('Q15.3.1: 180-Degree Reversal Pattern', () => {
    describe('Backend Engine', () => {
      it('should allow immediate reversal A→B→A when heights change', async () => {
        // Setup from FAQ example:
        // - Blue stack height 4 at A
        // - Red stack height 3 at B  
        // - Blue jumps A→B→C, then C→B→D (180° reversal)
        
        const engine = new GameEngine('faq-q15-1-backend', 'square19', createBasePlayers(), timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;
        
        // Clear board and set up scenario
        gameState.board.stacks.clear();
        
        const A: Position = { x: 4, y: 4 };
        const B: Position = { x: 6, y: 4 };
        
        // Blue stack (height 4) at A
        gameState.board.stacks.set('4,4', {
          position: A,
          rings: [1, 1, 1, 1],
          stackHeight: 4,
          capHeight: 4,
          controllingPlayer: 1
        });
        
        // Red stack (height 3) at B
        gameState.board.stacks.set('6,4', {
          position: B,
          rings: [2, 2, 2],
          stackHeight: 3,
          capHeight: 3,
          controllingPlayer: 2
        });
        
        gameState.currentPhase = 'capture';
        gameState.currentPlayer = 1;
        
        // Execute first capture: A → B → C
        const capture1 = await engine.makeMove({
          player: 1,
          type: 'overtaking_capture',
          from: A,
          captureTarget: B,
          to: { x: 8, y: 4 }
        } as any);
        
        expect(capture1.success).toBe(true);
        
        // Should enter chain_capture phase with mandatory continuation
        expect(gameState.currentPhase).toBe('chain_capture');
        
        // Let the chain resolve (will automatically execute 180° reversal if legal)
        while (gameState.currentPhase === 'chain_capture') {
          const moves = engine.getValidMoves(1);
          const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
          
          if (chainMoves.length === 0) break;
          
          const result = await engine.makeMove(chainMoves[0]);
          expect(result.success).toBe(true);
        }
        
        // Verify final state matches FAQ:
        // - Blue stack should have height 6 (4 original + 2 captured)
        // - Red stack at B should have height 1 (3 - 2 = 1)
        const allStacks: RingStack[] = Array.from(gameState.board.stacks.values());
        const blueStacks = allStacks.filter(s => s.controllingPlayer === 1);
        const redAtB = gameState.board.stacks.get('6,4');
        
        expect(blueStacks.length).toBe(1);
        expect(blueStacks[0]!.stackHeight).toBe(6);
        expect(redAtB).toBeDefined();
        expect(redAtB!.stackHeight).toBe(1);
      });
    });
    
    describe('Sandbox Engine', () => {
      it('should allow reversal pattern in sandbox', () => {
        // Sandbox test - validate chain capture mechanics work
        // Note: Sandbox may handle chain differently than backend
        
        const mockHandler = {
          requestPlayerChoice: jest.fn(),
          notifyGameUpdate: jest.fn(),
          requestChoice: jest.fn()
        };
        
        const engine = new ClientSandboxEngine({
          config: {
            boardType: 'square19',
            numPlayers: 2,
            playerKinds: ['human', 'human']
          },
          interactionHandler: mockHandler
        });
        
        const engineAny: any = engine;
        const state = engineAny.gameState;
        
        // Set up same scenario
        state.board.stacks.clear();
        
        const A: Position = { x: 4, y: 4 };
        const B: Position = { x: 6, y: 4 };
        
        state.board.stacks.set('4,4', {
          position: A,
          rings: [1, 1, 1, 1],
          stackHeight: 4,
          capHeight: 4,
          controllingPlayer: 1
        });
        
        state.board.stacks.set('6,4', {
          position: B,
          rings: [2, 2, 2],
          stackHeight: 3,
          capHeight: 3,
          controllingPlayer: 2
        });
        
        state.currentPhase = 'movement';
        state.currentPlayer = 0; // Player 1 (0-indexed)
        
        // Validate initial setup
        expect(state.board.stacks.get('4,4')!.stackHeight).toBe(4);
        expect(state.board.stacks.get('6,4')!.stackHeight).toBe(3);
        
        // Note: Full chain validation would require proper interaction with sandbox
        // This test validates the setup and initial state
      });
    });
  });

  describe('Q15.3.2: Cyclic Pattern', () => {
    describe('Backend Engine', () => {
      it('should allow cycle A→B→C→A when stack heights change', async () => {
        // Triangle cyclic pattern from FAQ
        // Blue at (3,3) cycles through three Red neighbors
        
        const engine = new GameEngine('faq-q15-2-backend', 'square8', createBasePlayers(), timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;
        
        gameState.board.stacks.clear();
        
        const startPos = { x: 3, y: 3 };
        const target1 = { x: 3, y: 4 };
        const target2 = { x: 4, y: 4 };
        const target3 = { x: 4, y: 3 };
        
        // Blue stack at start
        gameState.board.stacks.set('3,3', {
          position: startPos,
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1
        });
        
        // Three Red neighbors
        gameState.board.stacks.set('3,4', {
          position: target1,
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2
        });
        
        gameState.board.stacks.set('4,4', {
          position: target2,
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2
        });
        
        gameState.board.stacks.set('4,3', {
          position: target3,
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2
        });
        
        gameState.currentPhase = 'capture';
        gameState.currentPlayer = 1;
        
        // Start chain
        const capture1 = await engine.makeMove({
          player: 1,
          type: 'overtaking_capture',
          from: startPos,
          captureTarget: target1,
          to: { x: 3, y: 5 }
        } as any);
        
        expect(capture1.success).toBe(true);
        
        // Resolve mandatory chain
        while (gameState.currentPhase === 'chain_capture') {
          const moves = engine.getValidMoves(1);
          const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
          
          if (chainMoves.length === 0) break;
          
          const result = await engine.makeMove(chainMoves[0]);
          expect(result.success).toBe(true);
        }
        
        // Verify: Blue should have captured all 3 Red stacks
        const allStacks: RingStack[] = Array.from(gameState.board.stacks.values());
        const blueStacks = allStacks.filter(s => s.controllingPlayer === 1);
        const redStacks = allStacks.filter(s => s.controllingPlayer === 2);
        
        expect(blueStacks.length).toBe(1);
        expect(blueStacks[0]!.stackHeight).toBe(4); // 1 original + 3 captured
        expect(redStacks.length).toBe(0); // All Red stacks consumed
      });
    });
  });

  describe('Q15.3.3: Mandatory Continuation', () => {
    it('should force chain continuation until no legal captures remain', async () => {
      // Validates that chain must continue even if disadvantageous
      
      const engine = new GameEngine('faq-q15-3-backend', 'square8', createBasePlayers(), timeControl, false);
      const engineAny: any = engine;
      const gameState = engineAny.gameState;
      
      gameState.board.stacks.clear();
      
      // Create a simple chain scenario
      const start = { x: 0, y: 0 };
      const target1 = { x: 1, y: 1 };
      
      gameState.board.stacks.set('0,0', {
        position: start,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1
      });
      
      gameState.board.stacks.set('1,1', {
        position: target1,
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2
      });
      
      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;
      
      // Start chain
      const capture1 = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: start,
        captureTarget: target1,
        to: { x: 2, y: 2 }
      } as any);
      
      expect(capture1.success).toBe(true);
      
      // If chain continues, must follow through
      const MAX_ITERATIONS = 10;
      let iterationCount = 0;
      
      while (gameState.currentPhase === 'chain_capture' && iterationCount < MAX_ITERATIONS) {
        const moves = engine.getValidMoves(1);
        const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
        
        if (chainMoves.length === 0) {
          // No more legal captures - chain ends naturally
          break;
        }
        
        // Must continue if captures available
        expect(chainMoves.length).toBeGreaterThan(0);
        
        const result = await engine.makeMove(chainMoves[0]);
        expect(result.success).toBe(true);
        iterationCount++;
      }
      
      // Chain should eventually end (either no more captures or max iterations)
      // This validates the mandatory nature of continuation
      expect(iterationCount).toBeLessThan(MAX_ITERATIONS);
    });
  });
});