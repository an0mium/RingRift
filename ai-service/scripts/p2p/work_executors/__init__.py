"""Work executor modules extracted from P2POrchestrator._execute_claimed_work.

Each executor handles a specific work type (training, selfplay, tournament, gauntlet).
The dispatch function routes work items to the appropriate executor.

Feb 2026: Extracted from p2p_orchestrator.py (~406 lines) for maintainability.
"""

from scripts.p2p.work_executors.gauntlet_executor import execute_gauntlet_work
from scripts.p2p.work_executors.selfplay_executor import execute_selfplay_work
from scripts.p2p.work_executors.tournament_executor import execute_tournament_work
from scripts.p2p.work_executors.training_executor import execute_training_work

__all__ = [
    "execute_training_work",
    "execute_selfplay_work",
    "execute_tournament_work",
    "execute_gauntlet_work",
]
