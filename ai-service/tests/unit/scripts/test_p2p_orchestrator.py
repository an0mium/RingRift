"""Unit tests for P2P Orchestrator components.

Tests cover standalone components that can be tested in isolation:
- Constants and configuration
- Basic imports verification

December 2025: Initial test coverage for P2P orchestrator components.
Manager classes require full orchestrator context and are tested via integration tests
in tests/unit/p2p/.
"""

from __future__ import annotations

import pytest

# =============================================================================
# Test Constants and Configuration
# =============================================================================


class TestP2PConstants:
    """Tests for P2P constants and configuration."""

    def test_default_port(self):
        """Default port is set correctly."""
        from scripts.p2p.constants import DEFAULT_PORT

        assert DEFAULT_PORT == 8770

    def test_bootstrap_seeds_exist(self):
        """Bootstrap seeds are defined."""
        from scripts.p2p.constants import BOOTSTRAP_SEEDS

        assert isinstance(BOOTSTRAP_SEEDS, (list, tuple))

    def test_node_role_enum(self):
        """NodeRole enum has expected values."""
        from scripts.p2p.models import NodeRole

        assert NodeRole.LEADER.value == "leader"
        assert NodeRole.FOLLOWER.value == "follower"
        assert NodeRole.CANDIDATE.value == "candidate"


# =============================================================================
# Test Module Imports
# =============================================================================


class TestModuleImports:
    """Tests for module import availability."""

    def test_models_module_importable(self):
        """Models module is importable."""
        import scripts.p2p.models as models
        assert models is not None

    def test_leader_election_module_importable(self):
        """Leader election module is importable."""
        import scripts.p2p.leader_election as leader_election
        assert leader_election is not None

    def test_gossip_protocol_module_importable(self):
        """Gossip protocol module is importable."""
        import scripts.p2p.gossip_protocol as gossip_protocol
        assert gossip_protocol is not None

    def test_network_utils_module_importable(self):
        """Network utils module is importable."""
        import scripts.p2p.network_utils as network_utils
        assert network_utils is not None

    def test_orchestrator_module_importable(self):
        """Orchestrator module is importable."""
        import scripts.p2p_orchestrator as p2p_orchestrator
        assert hasattr(p2p_orchestrator, "P2POrchestrator")

    def test_managers_importable(self):
        """Manager modules are importable."""
        import scripts.p2p.managers.job_manager as job_manager
        import scripts.p2p.managers.selfplay_scheduler as selfplay_scheduler
        import scripts.p2p.managers.state_manager as state_manager
        assert state_manager is not None
        assert job_manager is not None
        assert selfplay_scheduler is not None
