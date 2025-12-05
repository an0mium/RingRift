import { Router } from 'express';
import type { Request, Response } from 'express';
import { asyncHandler, createError } from '../middleware/errorHandler';
import { getMetricsService } from '../services/MetricsService';
import { isRulesUxEventType, type RulesUxEventPayload } from '../../shared/telemetry/rulesUxEvents';

const router = Router();

/**
 * Coerce and validate an arbitrary payload into a RulesUxEventPayload.
 *
 * This performs minimal runtime validation to keep the telemetry surface
 * low-cardinality and free of obviously malformed events. Additional
 * normalisation is applied inside MetricsService.recordRulesUxEvent.
 */
function coerceRulesUxEventPayload(raw: unknown): RulesUxEventPayload {
  const body = (raw ?? {}) as Record<string, unknown>;

  const {
    type,
    boardType,
    numPlayers,
    aiDifficulty,
    topic,
    rulesConcept,
    scenarioId,
    weirdStateType,
    undoStreak,
    repeatCount,
    secondsSinceWeirdState,
  } = body;

  if (!isRulesUxEventType(type)) {
    throw createError('Invalid rules UX telemetry type', 400, 'INVALID_RULES_UX_EVENT_TYPE');
  }

  if (typeof boardType !== 'string' || boardType.trim().length === 0) {
    throw createError(
      'Invalid board type for rules UX telemetry',
      400,
      'INVALID_RULES_UX_BOARD_TYPE'
    );
  }

  if (
    typeof numPlayers !== 'number' ||
    !Number.isFinite(numPlayers) ||
    numPlayers < 1 ||
    numPlayers > 4
  ) {
    throw createError(
      'Invalid numPlayers for rules UX telemetry (expected 1-4)',
      400,
      'INVALID_RULES_UX_NUM_PLAYERS'
    );
  }

  const payload: RulesUxEventPayload = {
    type,
    // RulesUxEventPayload expects BoardType but we accept any string at runtime
    // and rely on MetricsService to normalise label values.
    boardType: boardType as unknown as RulesUxEventPayload['boardType'],
    numPlayers,
  };

  if (typeof aiDifficulty === 'number' && Number.isFinite(aiDifficulty)) {
    payload.aiDifficulty = aiDifficulty;
  }

  if (typeof topic === 'string' && topic.length > 0) {
    payload.topic = topic;
  }

  if (typeof rulesConcept === 'string' && rulesConcept.length > 0) {
    payload.rulesConcept = rulesConcept;
  }

  if (typeof scenarioId === 'string' && scenarioId.length > 0) {
    payload.scenarioId = scenarioId;
  }

  if (typeof weirdStateType === 'string' && weirdStateType.length > 0) {
    payload.weirdStateType = weirdStateType;
  }

  if (typeof undoStreak === 'number' && Number.isFinite(undoStreak) && undoStreak > 0) {
    payload.undoStreak = undoStreak;
  }

  if (typeof repeatCount === 'number' && Number.isFinite(repeatCount) && repeatCount > 0) {
    payload.repeatCount = repeatCount;
  }

  if (
    typeof secondsSinceWeirdState === 'number' &&
    Number.isFinite(secondsSinceWeirdState) &&
    secondsSinceWeirdState >= 0
  ) {
    payload.secondsSinceWeirdState = secondsSinceWeirdState;
  }

  return payload;
}

/**
 * Core handler for POST /api/telemetry/rules-ux.
 *
 * This endpoint is intentionally lightweight and privacy-aware:
 * - No user identifiers or raw board positions are recorded.
 * - Only coarse board / player / AI context and small enums are accepted.
 */
export function handleRulesUxTelemetry(req: Request, res: Response): void {
  const payload = coerceRulesUxEventPayload(req.body);

  const metrics = getMetricsService();
  metrics.recordRulesUxEvent(payload);

  // No body required; telemetry is fire-and-forget.
  res.status(204).send();
}

router.post(
  '/rules-ux',
  asyncHandler(async (req, res) => {
    handleRulesUxTelemetry(req, res);
  })
);

export default router;
