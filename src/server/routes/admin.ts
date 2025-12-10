import { Router } from 'express';
import type { AuthenticatedRequest } from '../middleware/auth';
import { authenticate, authorize } from '../middleware/auth';
import { orchestratorRollout } from '../services/OrchestratorRolloutService';
import { config } from '../config';

const router = Router();

/**
 * @openapi
 * /admin/orchestrator/status:
 *   get:
 *     summary: Get orchestrator rollout status
 *     description: |
 *       Returns the current orchestrator rollout configuration, circuit breaker
 *       state, and error rates. FSM is now the canonical validator.
 *       This endpoint is restricted to admin users.
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Orchestrator status snapshot
 */
router.get(
  '/orchestrator/status',
  authenticate,
  authorize(['admin']),
  (_req: AuthenticatedRequest, res) => {
    const orchestratorConfig = config.featureFlags.orchestrator;
    const cbState = orchestratorRollout.getCircuitBreakerState();
    const errorRatePercent = orchestratorRollout.getErrorRate();

    res.json({
      success: true,
      data: {
        config: {
          adapterEnabled: orchestratorConfig.adapterEnabled,
          allowlistUsers: orchestratorConfig.allowlistUsers,
          denylistUsers: orchestratorConfig.denylistUsers,
          circuitBreaker: {
            enabled: orchestratorConfig.circuitBreaker.enabled,
            errorThresholdPercent: orchestratorConfig.circuitBreaker.errorThresholdPercent,
            errorWindowSeconds: orchestratorConfig.circuitBreaker.errorWindowSeconds,
          },
        },
        circuitBreaker: {
          isOpen: cbState.isOpen,
          errorCount: cbState.errorCount,
          requestCount: cbState.requestCount,
          windowStart: new Date(cbState.windowStart).toISOString(),
          errorRatePercent,
        },
      },
    });
  }
);

export default router;
