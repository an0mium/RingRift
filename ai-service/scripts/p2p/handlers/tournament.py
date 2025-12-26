"""Tournament HTTP Handlers Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides distributed tournament endpoints for model evaluation.

Usage:
    class P2POrchestrator(TournamentHandlersMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from scripts.p2p.models import NodeRole

logger = logging.getLogger(__name__)


class TournamentHandlersMixin:
    """Mixin providing distributed tournament HTTP handlers.

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - peers: dict
    - peers_lock: threading.Lock
    - distributed_tournament_state: dict
    - _propose_tournament() method
    - _run_distributed_tournament() method
    - _play_tournament_match() method
    """

    # Type hints for IDE support
    node_id: str
    role: Any  # NodeRole
    peers: dict
    peers_lock: Any
    distributed_tournament_state: dict

    async def handle_tournament_start(self, request: web.Request) -> web.Response:
        """Start or propose a distributed tournament.

        DISTRIBUTED TOURNAMENT SCHEDULING:
        - Leaders can start tournaments directly (immediate)
        - Non-leaders can propose tournaments (gossip-based consensus)

        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "agent_ids": ["agent1", "agent2", "agent3"],
            "games_per_pairing": 2
        }
        """
        from scripts.p2p.models import DistributedTournamentState
        from scripts.p2p.types import NodeRole

        try:
            data = await request.json()

            # Non-leaders propose tournaments via gossip consensus
            if self.role != NodeRole.LEADER:
                agent_ids = data.get("agent_ids", [])
                if len(agent_ids) < 2:
                    return web.json_response({"error": "At least 2 agents required"}, status=400)

                proposal = self._propose_tournament(
                    board_type=data.get("board_type", "square8"),
                    num_players=data.get("num_players", 2),
                    agent_ids=agent_ids,
                    games_per_pairing=data.get("games_per_pairing", 2),
                )

                return web.json_response({
                    "success": True,
                    "mode": "proposal",
                    "proposal_id": proposal["proposal_id"],
                    "status": "Proposal created, awaiting gossip consensus",
                    "agents": agent_ids,
                })

            # Leader can start tournaments directly
            job_id = f"tournament_{uuid.uuid4().hex[:8]}"

            agent_ids = data.get("agent_ids", [])
            if len(agent_ids) < 2:
                return web.json_response({"error": "At least 2 agents required"}, status=400)

            # Create round-robin pairings
            pairings = []
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    for game_num in range(data.get("games_per_pairing", 2)):
                        pairings.append({
                            "agent1": a1,
                            "agent2": a2,
                            "game_num": game_num,
                            "status": "pending",
                        })

            state = DistributedTournamentState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                agent_ids=agent_ids,
                games_per_pairing=data.get("games_per_pairing", 2),
                total_matches=len(pairings),
                pending_matches=pairings,
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
            state.worker_nodes = workers

            if not state.worker_nodes:
                return web.json_response({"error": "No workers available"}, status=503)

            self.distributed_tournament_state[job_id] = state

            logger.info(f"Started tournament {job_id}: {len(agent_ids)} agents, {len(pairings)} matches, {len(workers)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_tournament(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "agents": agent_ids,
                "total_matches": len(pairings),
                "workers": workers,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_match(self, request: web.Request) -> web.Response:
        """Request a tournament match to be played by a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_info = data.get("match")

            if not job_id or not match_info:
                return web.json_response({"error": "job_id and match required"}, status=400)

            logger.info(f"Received tournament match request: {match_info}")

            # Start match in background
            asyncio.create_task(self._play_tournament_match(job_id, match_info))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": "match_started",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_tournament_state:
                    return web.json_response({"error": "Tournament not found"}, status=404)
                state = self.distributed_tournament_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_tournament_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_result(self, request: web.Request) -> web.Response:
        """Receive match result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_result = data.get("result", {})
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_tournament_state:
                return web.json_response({"error": "Tournament not found"}, status=404)

            state = self.distributed_tournament_state[job_id]
            state.results.append(match_result)
            state.completed_matches += 1
            state.last_update = time.time()

            logger.info(f"Tournament result: {state.completed_matches}/{state.total_matches} matches from {worker_id}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "completed": state.completed_matches,
                "total": state.total_matches,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
