"""P2P Serf Event HTTP handlers.

January 2026: Extracted from p2p_orchestrator.py to reduce file size (~255 LOC).

Endpoints:
- POST /serf/event - Receive events from Serf event handler

This mixin provides integration with HashiCorp Serf for reliable membership
and failure detection using the SWIM gossip protocol.

The handler accesses orchestrator state via `self.*` since it's designed
as a mixin that gets inherited by P2POrchestrator.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import NodeInfo

logger = logging.getLogger(__name__)


class SerfHandlersMixin:
    """Mixin providing Serf event HTTP handlers.

    Must be mixed into a class that provides:
    - self.auth_token
    - self._is_request_authorized()
    - self.node_id
    - self.peers (dict)
    - self.leader_id
    - self._set_leader()
    - self.election_in_progress
    - self._cooldown_manager
    - self._dead_peer_timestamps
    - self._sync_peer_snapshot()
    - NodeInfo class available in scope

    January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
    """

    async def handle_serf_event(self, request: web.Request) -> web.Response:
        """POST /serf/event - Receive events from Serf event handler.

        SERF INTEGRATION: HashiCorp Serf provides battle-tested SWIM gossip
        for membership and failure detection. This endpoint receives events
        from the serf_event_handler.py script.

        Event types:
        - member-join: New node joined the cluster
        - member-leave: Node gracefully left
        - member-failed: Node failed (detected by SWIM)
        - member-update: Node tags changed
        - member-reap: Failed node was reaped from membership list
        - user: Custom user event (training-complete, model-promoted, etc.)

        Request body:
        {
            "event_type": "member-join",
            "timestamp": "2025-12-26T...",
            "payload": { event-specific data }
        }

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)

            data = await request.json()
            event_type = data.get("event_type", "")
            timestamp = data.get("timestamp", "")
            payload = data.get("payload", {})

            logger.info(f"Serf event received: {event_type} at {timestamp}")

            # Process based on event type
            if event_type == "member-join":
                await self._handle_serf_member_join(payload.get("members", []))
            elif event_type == "member-leave":
                await self._handle_serf_member_leave(payload.get("members", []))
            elif event_type == "member-failed":
                await self._handle_serf_member_failed(payload.get("members", []))
            elif event_type == "member-update":
                await self._handle_serf_member_update(payload.get("members", []))
            elif event_type == "member-reap":
                await self._handle_serf_member_reap(payload.get("members", []))
            elif event_type == "user":
                await self._handle_serf_user_event(payload)
            else:
                logger.warning(f"Unknown Serf event type: {event_type}")

            return web.json_response({
                "status": "processed",
                "event_type": event_type,
                "node_id": self.node_id,
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing Serf event: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_serf_member_join(self, members: list) -> None:
        """Handle Serf member-join events.

        When Serf detects new members, update our peer list and mark them alive.
        This is more reliable than our custom gossip because Serf uses SWIM
        with indirect probing.

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        # Import NodeInfo here to avoid circular import at module level
        try:
            from scripts.p2p_orchestrator import NodeInfo
        except ImportError:
            NodeInfo = None

        for member in members:
            node_name = member.get("name", "")
            addr = member.get("addr", "")
            tags = member.get("tags", {})

            if not node_name or node_name == self.node_id:
                continue

            logger.info(f"Serf: member joined: {node_name} @ {addr}")

            # Update peer state
            now = time.time()
            if node_name not in self.peers:
                # Parse addr to get host:port (format: "ip:port")
                host, port = (addr.rsplit(":", 1) + ["8770"])[:2] if ":" in addr else (addr, "8770")
                try:
                    port_int = int(port)
                except ValueError:
                    port_int = 8770

                if NodeInfo:
                    self.peers[node_name] = NodeInfo(
                        node_id=node_name,
                        host=host or "unknown",
                        port=port_int,
                        last_heartbeat=now,
                    )
            else:
                peer = self.peers[node_name]
                if hasattr(peer, 'last_heartbeat'):
                    peer.last_heartbeat = now
                    if addr:
                        host, _ = (addr.rsplit(":", 1) + [""])[:2] if ":" in addr else (addr, "")
                        if host:
                            peer.host = host

            # Extract tags into peer info (store as capability hints)
            # Note: Serf tags are for reference only, NodeInfo uses capabilities list

        # C2 fix: Sync peer snapshot after Serf member join updates
        if members:
            self._sync_peer_snapshot()

    async def _handle_serf_member_leave(self, members: list) -> None:
        """Handle Serf member-leave events (graceful departure).

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        for member in members:
            node_name = member.get("name", "")

            if not node_name or node_name == self.node_id:
                continue

            logger.info(f"Serf: member left gracefully: {node_name}")

            if node_name in self.peers:
                peer = self.peers[node_name]
                if hasattr(peer, 'retired'):
                    # Mark as retired (NodeInfo equivalent of "left")
                    peer.retired = True
                    peer.retired_at = time.time()
                    # Jan 20, 2026: Use adaptive cooldown manager
                    if self._cooldown_manager:
                        self._cooldown_manager.record_death(node_name)
                    else:
                        self._dead_peer_timestamps[node_name] = time.time()

        # C2 fix: Sync peer snapshot after Serf member leave updates
        if members:
            self._sync_peer_snapshot()

    async def _handle_serf_member_failed(self, members: list) -> None:
        """Handle Serf member-failed events (SWIM failure detection).

        SWIM's failure detection is more reliable than our custom ping/pong
        because it uses indirect probing through multiple peers.

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        for member in members:
            node_name = member.get("name", "")
            addr = member.get("addr", "")

            if not node_name or node_name == self.node_id:
                continue

            logger.warning(f"Serf: member FAILED (SWIM detected): {node_name} @ {addr}")

            if node_name in self.peers:
                peer = self.peers[node_name]
                if hasattr(peer, 'consecutive_failures'):
                    # Mark with consecutive failures (triggers dead/suspect state)
                    peer.consecutive_failures += 1
                    peer.last_failure_time = time.time()

            # If the failed node was leader, trigger election
            if node_name == self.leader_id:
                logger.warning(f"Leader {node_name} failed (Serf detected) - triggering election")
                # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                self._set_leader(None, reason="serf_leader_failed", save_state=True)
                self.election_in_progress = False  # Allow new election

        # C2 fix: Sync peer snapshot after Serf member failed updates
        if members:
            self._sync_peer_snapshot()

    async def _handle_serf_member_update(self, members: list) -> None:
        """Handle Serf member-update events (tag changes).

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        for member in members:
            node_name = member.get("name", "")
            tags = member.get("tags", {})

            if not node_name or node_name == self.node_id:
                continue

            logger.info(f"Serf: member updated: {node_name}")

            if node_name in self.peers:
                peer = self.peers[node_name]
                if hasattr(peer, 'last_heartbeat'):
                    peer.last_heartbeat = time.time()
                    # Tags can update capabilities if structured appropriately
                    if "capabilities" in tags and isinstance(tags["capabilities"], list):
                        peer.capabilities = tags["capabilities"]

    async def _handle_serf_member_reap(self, members: list) -> None:
        """Handle Serf member-reap events (failed nodes removed from list).

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        for member in members:
            node_name = member.get("name", "")

            if not node_name or node_name == self.node_id:
                continue

            logger.info(f"Serf: member reaped (final cleanup): {node_name}")

            # Mark as retired (reaped means permanently gone)
            if node_name in self.peers:
                peer = self.peers[node_name]
                if hasattr(peer, 'retired'):
                    peer.retired = True
                    peer.retired_at = time.time()
                    # Jan 20, 2026: Use adaptive cooldown manager
                    if self._cooldown_manager:
                        self._cooldown_manager.record_death(node_name)
                    else:
                        self._dead_peer_timestamps[node_name] = time.time()

    async def _handle_serf_user_event(self, payload: dict) -> None:
        """Handle Serf user events (custom RingRift events).

        User events include:
        - training-complete: Training job finished
        - model-promoted: Model was promoted to canonical
        - selfplay-started: Selfplay jobs started on a node
        - node-status: Periodic node status broadcast

        January 2026: Moved from p2p_orchestrator.py to SerfHandlersMixin.
        """
        event_name = payload.get("name", "")
        event_payload = payload.get("payload", {})
        ltime = payload.get("ltime", "0")

        logger.info(f"Serf user event: {event_name} (ltime={ltime})")

        if event_name == "training-complete":
            config_key = event_payload.get("config_key", "")
            model_path = event_payload.get("model_path", "")
            metrics = event_payload.get("metrics", {})
            logger.info(f"Training complete via Serf: {config_key} -> {model_path}")
            # Could trigger evaluation here

        elif event_name == "model-promoted":
            config_key = event_payload.get("config_key", "")
            model_path = event_payload.get("model_path", "")
            elo_gain = event_payload.get("elo_gain", 0)
            logger.info(f"Model promoted via Serf: {config_key} (+{elo_gain} Elo)")
            # Could trigger model distribution here

        elif event_name == "selfplay-started":
            node = event_payload.get("node", "")
            config_key = event_payload.get("config_key", "")
            job_count = event_payload.get("job_count", 1)
            logger.info(f"Selfplay started via Serf: {node} running {config_key} x{job_count}")

        elif event_name == "node-status":
            # Status updates from nodes - could merge with gossip state
            node_id = event_payload.get("node_id", "")
            if node_id and node_id in self.peers:
                # Update peer with status info
                status_fields = ["gpu_util", "gpu_mem_used", "cpu_percent", "memory_percent"]
                for field in status_fields:
                    if field in event_payload:
                        self.peers[node_id][field] = event_payload[field]
