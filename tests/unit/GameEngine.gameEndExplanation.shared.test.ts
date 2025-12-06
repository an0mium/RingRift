import { toVictoryState } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState } from '../../src/shared/types/game';
import { createTestGameState, addMarker } from '../utils/fixtures';

describe('shared engine – GameEndExplanation wiring', () => {
  it('attaches ring-majority GameEndExplanation on ring-elimination victory', () => {
    const state = createTestGameState();
    const p1 = state.players[0];

    state.victoryThreshold = 5;
    p1.eliminatedRings = 5;
    state.players[1].eliminatedRings = 0;

    // Victory is purely threshold-based here; phase is irrelevant for evaluateVictory.
    const victory = toVictoryState(state);

    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('ring_elimination');
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('ring_elimination');
    expect(explanation.victoryReasonCode).toBe('victory_ring_majority');
    expect(explanation.winnerPlayerId).toBe('P1');
    expect(explanation.boardType).toBe(state.boardType);
    expect(explanation.numPlayers).toBe(state.players.length);

    // No weird-state or telemetry context for simple ring-majority endings.
    expect(explanation.weirdStateContext).toBeUndefined();
    expect(explanation.telemetry).toBeUndefined();
    expect(explanation.teaching).toBeUndefined();
  });

  it('attaches LPS GameEndExplanation with weird-state context for last_player_standing endings', () => {
    const state = createTestGameState();

    // Bare-board structural situation mirroring victory.shared marker tiebreak test.
    state.board.stacks.clear();
    state.board.markers.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 2;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    // Two markers for Player 1, one for Player 2 -> marker tiebreak → last_player_standing.
    addMarker(state.board, { x: 0, y: 0 }, 1);
    addMarker(state.board, { x: 1, y: 0 }, 1);
    addMarker(state.board, { x: 0, y: 1 }, 2);

    // Structural bare-board LPS via marker tiebreak; call victory helper directly.
    const victory = toVictoryState(state);

    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('last_player_standing');
    expect(victory.winner).toBe(1);
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('last_player_standing');
    expect(explanation.victoryReasonCode).toBe('victory_last_player_standing');
    expect(explanation.primaryConceptId).toBe('lps_real_actions');

    expect(explanation.weirdStateContext).toBeDefined();
    const ctx = explanation.weirdStateContext!;

    expect(ctx.reasonCodes).toContain('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
    expect(ctx.primaryReasonCode).toBe('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
    expect(ctx.rulesContextTags).toContain('last_player_standing');
    expect(ctx.teachingTopicIds).toContain('teaching.victory_stalemate');

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(telemetry.weirdStateReasonCodes).toEqual(
      expect.arrayContaining(['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'])
    );
    expect(telemetry.rulesContextTags).toEqual(expect.arrayContaining(['last_player_standing']));
  });

  it('attaches structural-stalemate GameEndExplanation with territory tiebreak details', () => {
    const state = createTestGameState();

    // Bare-board global stalemate with differing territory counts, mirroring
    // the victory.shared territory tiebreak test.
    state.board.stacks.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    state.players[0].territorySpaces = 3;
    state.players[1].territorySpaces = 1;

    // Bare-board territory tiebreak; invoke victory helper directly.
    const victory = toVictoryState(state);

    // Aggregate reason remains 'territory_control' but explanation should
    // classify this as a structural stalemate tiebreak.
    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('territory_control');
    expect(victory.winner).toBe(1);
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('structural_stalemate');
    expect(explanation.victoryReasonCode).toBe('victory_structural_stalemate_tiebreak');
    expect(explanation.primaryConceptId).toBe('structural_stalemate');

    expect(explanation.tiebreakSteps).toBeDefined();
    const steps = explanation.tiebreakSteps!;
    expect(steps.length).toBeGreaterThanOrEqual(1);

    const firstStep = steps[0];
    expect(firstStep.kind).toBe('territory_spaces');
    expect(firstStep.winnerPlayerId).toBe('P1');
    expect(firstStep.valuesByPlayer.P1).toBe(3);
    expect(firstStep.valuesByPlayer.P2).toBe(1);

    expect(explanation.weirdStateContext).toBeDefined();
    const ctx = explanation.weirdStateContext!;
    expect(ctx.reasonCodes).toContain('STRUCTURAL_STALEMATE_TIEBREAK');
    expect(ctx.rulesContextTags).toContain('structural_stalemate');

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;
    expect(telemetry.rulesContextTags).toEqual(['structural_stalemate']);
    expect(telemetry.weirdStateReasonCodes).toEqual(['STRUCTURAL_STALEMATE_TIEBREAK']);
  });
});
