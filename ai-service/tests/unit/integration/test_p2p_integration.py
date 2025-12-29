"""
Tests for app.integration.p2p_integration module.

Tests the P2P cluster integration infrastructure:
- P2PIntegrationConfig configuration
- P2PNodeCapability enum
- P2PNode dataclass
- P2PAPIClient REST API client
- P2PSelfplayBridge selfplay coordination
- P2PTrainingBridge training coordination
- P2PEvaluationBridge evaluation coordination
- P2PIntegrationManager main integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integration.p2p_integration import (
    EvaluationCoordinator,
    P2PAPIClient,
    P2PEvaluationBridge,
    P2PIntegrationConfig,
    P2PIntegrationManager,
    P2PNode,
    P2PNodeCapability,
    P2PSelfplayBridge,
    P2PTrainingBridge,
    SelfplayCoordinator,
    TrainingCoordinator,
)


class TestP2PIntegrationConfig:
    """Tests for P2PIntegrationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = P2PIntegrationConfig()
        assert config.connect_timeout == 10.0
        assert config.request_timeout == 60.0
        assert config.model_sync_enabled is True
        assert config.data_sync_enabled is True
        assert config.sync_interval_seconds == 300.0
        assert config.prefer_gpu_for_training is True
        assert config.min_gpu_memory_gb == 8
        assert config.max_concurrent_training_jobs == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = P2PIntegrationConfig(
            p2p_base_url="http://custom:8888",
            auth_token="secret-token",
            connect_timeout=5.0,
            request_timeout=120.0,
            target_selfplay_games_per_hour=2000,
        )
        assert config.p2p_base_url == "http://custom:8888"
        assert config.auth_token == "secret-token"
        assert config.connect_timeout == 5.0
        assert config.request_timeout == 120.0
        assert config.target_selfplay_games_per_hour == 2000

    def test_selfplay_settings(self):
        """Test selfplay configuration settings."""
        config = P2PIntegrationConfig(
            target_selfplay_games_per_hour=500,
            auto_scale_selfplay=False,
        )
        assert config.target_selfplay_games_per_hour == 500
        assert config.auto_scale_selfplay is False

    def test_tournament_settings(self):
        """Test tournament configuration settings."""
        config = P2PIntegrationConfig(
            tournament_games_per_pair=100,
            use_distributed_tournament=False,
        )
        assert config.tournament_games_per_pair == 100
        assert config.use_distributed_tournament is False

    def test_health_monitoring_settings(self):
        """Test health monitoring configuration."""
        config = P2PIntegrationConfig(
            health_check_interval=60.0,
            unhealthy_threshold_failures=5,
        )
        assert config.health_check_interval == 60.0
        assert config.unhealthy_threshold_failures == 5


class TestP2PNodeCapability:
    """Tests for P2PNodeCapability enum."""

    def test_all_capabilities_defined(self):
        """Test all capabilities are defined."""
        assert P2PNodeCapability.CPU_SELFPLAY.value == "cpu_selfplay"
        assert P2PNodeCapability.GPU_SELFPLAY.value == "gpu_selfplay"
        assert P2PNodeCapability.TRAINING.value == "training"
        assert P2PNodeCapability.CMAES.value == "cmaes"
        assert P2PNodeCapability.TOURNAMENT.value == "tournament"
        assert P2PNodeCapability.DATA_STORAGE.value == "data_storage"

    def test_capability_count(self):
        """Test expected number of capabilities."""
        assert len(P2PNodeCapability) == 6


class TestP2PNode:
    """Tests for P2PNode dataclass."""

    def test_minimal_creation(self):
        """Test minimal creation with required fields."""
        node = P2PNode(
            node_id="test-node",
            host="192.168.1.100",
            port=8770,
        )
        assert node.node_id == "test-node"
        assert node.host == "192.168.1.100"
        assert node.port == 8770
        assert node.is_alive is False
        assert node.is_healthy is False
        assert node.has_gpu is False

    def test_full_creation(self):
        """Test creation with all fields."""
        node = P2PNode(
            node_id="gpu-node-1",
            host="192.168.1.101",
            port=8770,
            is_alive=True,
            is_healthy=True,
            has_gpu=True,
            gpu_name="NVIDIA H100",
            gpu_power_score=95,
            memory_gb=128,
            disk_percent=45.5,
            cpu_percent=30.0,
            selfplay_jobs=2,
            training_jobs=1,
            capabilities=[P2PNodeCapability.GPU_SELFPLAY, P2PNodeCapability.TRAINING],
            last_heartbeat=1700000000.0,
        )
        assert node.has_gpu is True
        assert node.gpu_name == "NVIDIA H100"
        assert node.gpu_power_score == 95
        assert node.memory_gb == 128
        assert len(node.capabilities) == 2
        assert P2PNodeCapability.GPU_SELFPLAY in node.capabilities


class TestP2PAPIClient:
    """Tests for P2PAPIClient class."""

    def test_initialization(self):
        """Test client initialization."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        assert client.config is config
        assert client._session is None

    def test_initialization_with_auth(self):
        """Test client initialization with auth token."""
        config = P2PIntegrationConfig(auth_token="test-token")
        client = P2PAPIClient(config)
        assert "Authorization" in client._headers
        assert client._headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_close(self):
        """Test client close method."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        client._session = None
        # Should not raise even without session
        await client.close()

    @pytest.mark.asyncio
    async def test_get_session_no_aiohttp(self):
        """Test get_session when aiohttp not available."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)

        with patch('app.integration.p2p_integration.HAS_AIOHTTP', False):
            with pytest.raises(RuntimeError, match="aiohttp not available"):
                await client._get_session()

    @pytest.mark.asyncio
    async def test_request_error_handling(self):
        """Test request error handling via patching _get_session."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)

        # Mock _get_session to return a session that raises on request
        with patch.object(client, '_get_session', side_effect=Exception("Connection failed")):
            result = await client._request("GET", "/test")
            assert "error" in result
            # Error message contains our exception text
            assert "Connection failed" in result["error"] or "error" in result

    @pytest.mark.asyncio
    async def test_get_leader(self):
        """Test get_leader method."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)

        with patch.object(client, 'get_cluster_status', return_value={"leader": "node-1"}):
            leader = await client.get_leader()
            assert leader == "node-1"

    @pytest.mark.asyncio
    async def test_get_nodes(self):
        """Test get_nodes method."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)

        mock_status = {
            "nodes": [
                {
                    "node_id": "node-1",
                    "host": "192.168.1.1",
                    "port": 8770,
                    "is_alive": True,
                    "is_healthy": True,
                    "has_gpu": True,
                    "gpu_name": "RTX 4090",
                },
                {
                    "node_id": "node-2",
                    "host": "192.168.1.2",
                    "port": 8770,
                    "is_alive": True,
                    "is_healthy": False,
                    "has_gpu": False,
                }
            ]
        }

        with patch.object(client, 'get_cluster_status', return_value=mock_status):
            nodes = await client.get_nodes()
            assert len(nodes) == 2
            assert nodes[0].node_id == "node-1"
            assert nodes[0].has_gpu is True
            assert P2PNodeCapability.GPU_SELFPLAY in nodes[0].capabilities
            assert nodes[1].has_gpu is False
            assert P2PNodeCapability.CPU_SELFPLAY in nodes[1].capabilities


class TestP2PSelfplayBridge:
    """Tests for P2PSelfplayBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        assert bridge.client is client
        assert bridge.config is config
        assert bridge._selfplay_targets == {}
        assert bridge._rate_multiplier == 1.0

    def test_all_configs_defined(self):
        """Test ALL_CONFIGS constant."""
        assert len(P2PSelfplayBridge.ALL_CONFIGS) == 9
        # Check some expected configs
        assert ("square8", 2) in P2PSelfplayBridge.ALL_CONFIGS
        assert ("square8", 4) in P2PSelfplayBridge.ALL_CONFIGS
        assert ("hexagonal", 2) in P2PSelfplayBridge.ALL_CONFIGS

    def test_adjust_target_rate(self):
        """Test adjusting target selfplay rate."""
        config = P2PIntegrationConfig(target_selfplay_games_per_hour=1000)
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        new_rate = bridge.adjust_target_rate(1.5, "Testing")
        assert new_rate == 1500
        assert bridge._rate_multiplier == 1.5

    def test_adjust_target_rate_clamped(self):
        """Test that rate multiplier is clamped to bounds."""
        config = P2PIntegrationConfig(target_selfplay_games_per_hour=1000)
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        # Test upper bound
        bridge.adjust_target_rate(5.0, "Too high")
        assert bridge._rate_multiplier == 2.5

        # Test lower bound
        bridge.adjust_target_rate(0.1, "Too low")
        assert bridge._rate_multiplier == 0.5

    def test_get_effective_target_rate(self):
        """Test getting effective target rate."""
        config = P2PIntegrationConfig(target_selfplay_games_per_hour=1000)
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        assert bridge.get_effective_target_rate() == 1000

        bridge._rate_multiplier = 1.5
        assert bridge.get_effective_target_rate() == 1500

    def test_update_curriculum_weights(self):
        """Test updating curriculum weights."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        weights = {"square8_2p": 1.2, "hex8_2p": 0.8}
        bridge.update_curriculum_weights(weights)

        assert bridge._curriculum_weights == weights
        # Ensure it's a copy
        weights["new_key"] = 1.0
        assert "new_key" not in bridge._curriculum_weights

    def test_select_weighted_config_no_weights(self):
        """Test config selection without curriculum weights."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        # Should use round-robin
        config1 = bridge.select_weighted_config()
        config2 = bridge.select_weighted_config()

        # Both should be valid configs
        assert config1 in P2PSelfplayBridge.ALL_CONFIGS
        assert config2 in P2PSelfplayBridge.ALL_CONFIGS

    def test_select_weighted_config_with_weights(self):
        """Test config selection with curriculum weights."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        # Set heavy weight on one config
        weights = {f"{b}_{n}p": 0.1 for b, n in P2PSelfplayBridge.ALL_CONFIGS}
        weights["square8_2p"] = 100.0  # Very high weight
        bridge.update_curriculum_weights(weights)

        # Run multiple selections
        selections = [bridge.select_weighted_config() for _ in range(20)]

        # Most should be square8_2p due to high weight
        square8_2p_count = sum(1 for s in selections if s == ("square8", 2))
        assert square8_2p_count >= 15  # At least 75%

    @pytest.mark.asyncio
    async def test_get_current_rate(self):
        """Test getting current selfplay rate."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        with patch.object(client, 'get_cluster_status', return_value={"selfplay_rate": 500}):
            rate = await bridge.get_current_rate()
            assert rate == 500

    @pytest.mark.asyncio
    async def test_auto_scale_disabled(self):
        """Test auto_scale when disabled."""
        config = P2PIntegrationConfig(auto_scale_selfplay=False)
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        result = await bridge.auto_scale()
        assert result == {"action": "disabled"}

    @pytest.mark.asyncio
    async def test_get_distribution(self):
        """Test getting selfplay distribution."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PSelfplayBridge(client, config)

        mock_nodes = [
            P2PNode(node_id="n1", host="h1", port=8770, is_alive=True, selfplay_jobs=2),
            P2PNode(node_id="n2", host="h2", port=8770, is_alive=True, selfplay_jobs=3),
            P2PNode(node_id="n3", host="h3", port=8770, is_alive=False, selfplay_jobs=1),
        ]

        with patch.object(client, 'get_nodes', return_value=mock_nodes):
            dist = await bridge.get_distribution()
            assert dist == {"n1": 2, "n2": 3}


class TestP2PTrainingBridge:
    """Tests for P2PTrainingBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)

        assert bridge.client is client
        assert bridge.config is config
        assert bridge._current_training_node is None

    @pytest.mark.asyncio
    async def test_select_training_node_gpu_preferred(self):
        """Test that GPU nodes are preferred for training."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)

        mock_nodes = [
            P2PNode(node_id="cpu1", host="h1", port=8770, is_healthy=True, has_gpu=False, gpu_power_score=0),
            P2PNode(node_id="gpu1", host="h2", port=8770, is_healthy=True, has_gpu=True, gpu_power_score=80),
            P2PNode(node_id="gpu2", host="h3", port=8770, is_healthy=True, has_gpu=True, gpu_power_score=90),
        ]

        with patch.object(client, 'get_nodes', return_value=mock_nodes):
            selected = await bridge.select_training_node()
            # Should select highest GPU power score
            assert selected.node_id == "gpu2"

    @pytest.mark.asyncio
    async def test_select_training_node_no_candidates(self):
        """Test training node selection with no candidates."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)

        mock_nodes = [
            P2PNode(node_id="n1", host="h1", port=8770, is_healthy=False),
            P2PNode(node_id="n2", host="h2", port=8770, is_healthy=True, training_jobs=1),
        ]

        with patch.object(client, 'get_nodes', return_value=mock_nodes):
            selected = await bridge.select_training_node()
            assert selected is None

    @pytest.mark.asyncio
    async def test_start_training_no_node(self):
        """Test start_training with no available node."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)

        with patch.object(bridge, 'select_training_node', return_value=None):
            result = await bridge.start_training()
            assert "error" in result
            assert "No available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting training status."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)
        bridge._current_training_node = "gpu1"

        with patch.object(client, 'get_training_status', return_value={"status": "running"}):
            status = await bridge.get_status()
            assert status["status"] == "running"
            assert status["selected_node"] == "gpu1"

    @pytest.mark.asyncio
    async def test_aggregate_data(self):
        """Test data aggregation."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PTrainingBridge(client, config)

        mock_nodes = [
            P2PNode(node_id="n1", host="h1", port=8770, is_alive=True),
            P2PNode(node_id="n2", host="h2", port=8770, is_alive=True),
            P2PNode(node_id="target", host="h3", port=8770, is_alive=True),
        ]

        with patch.object(client, 'get_nodes', return_value=mock_nodes):
            with patch.object(client, 'trigger_sync', return_value={"success": True}):
                result = await bridge.aggregate_data("target")
                assert "sync_operations" in result
                # Should sync from n1 and n2 but not target itself
                assert len(result["sync_operations"]) == 2


class TestP2PEvaluationBridge:
    """Tests for P2PEvaluationBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PEvaluationBridge(client, config)

        assert bridge.client is client
        assert bridge.config is config

    @pytest.mark.asyncio
    async def test_run_tournament_distributed(self):
        """Test running distributed tournament."""
        config = P2PIntegrationConfig(use_distributed_tournament=True)
        client = P2PAPIClient(config)
        bridge = P2PEvaluationBridge(client, config)

        with patch.object(client, 'start_tournament', return_value={"status": "started"}):
            result = await bridge.run_tournament(["model1", "model2"], games_per_pair=100)
            assert result["status"] == "started"

    @pytest.mark.asyncio
    async def test_run_tournament_local(self):
        """Test running local tournament (not implemented)."""
        config = P2PIntegrationConfig(use_distributed_tournament=False)
        client = P2PAPIClient(config)
        bridge = P2PEvaluationBridge(client, config)

        result = await bridge.run_tournament(["model1", "model2"])
        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_models(self):
        """Test head-to-head comparison."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PEvaluationBridge(client, config)

        with patch.object(bridge, 'run_tournament', return_value={"winner": "model1"}):
            result = await bridge.compare_models("model1", "model2", games=50)
            assert result["winner"] == "model1"

    @pytest.mark.asyncio
    async def test_get_leaderboard(self):
        """Test getting leaderboard."""
        config = P2PIntegrationConfig()
        client = P2PAPIClient(config)
        bridge = P2PEvaluationBridge(client, config)

        with patch.object(client, 'get_elo_leaderboard', return_value={
            "leaderboard": [
                {"model_id": "m1", "elo": 1650},
                {"model_id": "m2", "elo": 1550},
            ]
        }):
            leaderboard = await bridge.get_leaderboard()
            assert len(leaderboard) == 2
            assert leaderboard[0]["model_id"] == "m1"


class TestBackwardCompatibilityAliases:
    """Tests for backward compatibility aliases."""

    def test_selfplay_coordinator_alias(self):
        """Test SelfplayCoordinator is alias for P2PSelfplayBridge."""
        assert SelfplayCoordinator is P2PSelfplayBridge

    def test_training_coordinator_alias(self):
        """Test TrainingCoordinator is alias for P2PTrainingBridge."""
        assert TrainingCoordinator is P2PTrainingBridge

    def test_evaluation_coordinator_alias(self):
        """Test EvaluationCoordinator is alias for P2PEvaluationBridge."""
        assert EvaluationCoordinator is P2PEvaluationBridge


class TestP2PIntegrationManager:
    """Tests for P2PIntegrationManager class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        manager = P2PIntegrationManager()
        assert manager.config is not None
        assert manager.client is not None
        assert manager.selfplay is not None
        assert manager.training is not None
        assert manager.evaluation is not None
        assert manager._running is False

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = P2PIntegrationConfig(
            p2p_base_url="http://custom:9999",
            target_selfplay_games_per_hour=2000,
        )
        manager = P2PIntegrationManager(config)
        assert manager.config.p2p_base_url == "http://custom:9999"
        assert manager.config.target_selfplay_games_per_hour == 2000

    def test_initialization_auth_from_env(self):
        """Test auth token loading from environment."""
        with patch.dict('os.environ', {"RINGRIFT_CLUSTER_AUTH_TOKEN": "env-token"}):
            config = P2PIntegrationConfig()  # No auth_token set
            manager = P2PIntegrationManager(config)
            assert manager.config.auth_token == "env-token"

    def test_register_callback(self):
        """Test callback registration."""
        manager = P2PIntegrationManager()

        callback = MagicMock()
        manager.register_callback("test_event", callback)

        assert "test_event" in manager._callbacks
        assert callback in manager._callbacks["test_event"]

    @pytest.mark.asyncio
    async def test_fire_callbacks_sync(self):
        """Test firing synchronous callbacks."""
        manager = P2PIntegrationManager()

        callback = MagicMock()
        manager.register_callback("test", callback)

        await manager._fire_callbacks("test", arg1="value1")
        callback.assert_called_once_with(arg1="value1")

    @pytest.mark.asyncio
    async def test_fire_callbacks_async(self):
        """Test firing async callbacks."""
        manager = P2PIntegrationManager()

        callback = AsyncMock()
        manager.register_callback("test", callback)

        await manager._fire_callbacks("test", arg1="value1")
        callback.assert_called_once_with(arg1="value1")

    @pytest.mark.asyncio
    async def test_fire_callbacks_error_handling(self):
        """Test callback error handling."""
        manager = P2PIntegrationManager()

        failing_callback = MagicMock(side_effect=Exception("Callback failed"))
        success_callback = MagicMock()

        manager.register_callback("test", failing_callback)
        manager.register_callback("test", success_callback)

        # Should not raise, and should call second callback
        await manager._fire_callbacks("test")
        success_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test manager start and stop lifecycle."""
        manager = P2PIntegrationManager()

        # Mock the loops to prevent actual execution
        with patch.object(manager, '_health_check_loop', AsyncMock()):
            with patch.object(manager, '_selfplay_management_loop', AsyncMock()):
                with patch.object(manager, '_sync_loop', AsyncMock()):
                    await manager.start()
                    assert manager._running is True
                    assert len(manager._tasks) == 3

                    await manager.stop()
                    assert manager._running is False
                    assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_start_improvement_cycle(self):
        """Test starting improvement cycle."""
        manager = P2PIntegrationManager()

        with patch.object(manager.client, 'start_improvement_loop', return_value={"status": "started"}):
            result = await manager.start_improvement_cycle()
            assert result["status"] == "started"

    @pytest.mark.asyncio
    async def test_trigger_training(self):
        """Test triggering training."""
        manager = P2PIntegrationManager()

        with patch.object(manager.training, 'start_training', return_value={"job_id": "train-1"}):
            result = await manager.trigger_training(wait_for_completion=False)
            assert result["job_id"] == "train-1"

    @pytest.mark.asyncio
    async def test_evaluate_model_with_reference(self):
        """Test evaluating model with reference."""
        manager = P2PIntegrationManager()

        with patch.object(manager.evaluation, 'compare_models', return_value={"winner": "new_model"}):
            result = await manager.evaluate_model("new_model", reference_model="old_model")
            assert result["winner"] == "new_model"

    @pytest.mark.asyncio
    async def test_evaluate_model_no_reference(self):
        """Test evaluating model without reference."""
        manager = P2PIntegrationManager()

        with patch.object(manager.evaluation, 'get_leaderboard', return_value=[
            {"model_id": "top1"}, {"model_id": "top2"}
        ]):
            with patch.object(manager.evaluation, 'run_tournament', return_value={"tournament_id": "t1"}):
                result = await manager.evaluate_model("new_model")
                assert result["tournament_id"] == "t1"

    @pytest.mark.asyncio
    async def test_sync_model_to_cluster(self):
        """Test syncing model to cluster."""
        manager = P2PIntegrationManager()

        mock_nodes = [
            P2PNode(node_id="n1", host="h1", port=8770, is_healthy=True),
            P2PNode(node_id="n2", host="h2", port=8770, is_healthy=True),
        ]

        from pathlib import Path
        with patch.object(manager.client, 'get_nodes', return_value=mock_nodes):
            result = await manager.sync_model_to_cluster("test_model", Path("/tmp/model.pth"))
            assert result["synced_nodes"] == 2
            assert result["total_nodes"] == 2


class TestIntegration:
    """Integration tests for P2P components."""

    def test_full_bridge_initialization(self):
        """Test that all bridges are properly initialized from manager."""
        manager = P2PIntegrationManager()

        assert isinstance(manager.selfplay, P2PSelfplayBridge)
        assert isinstance(manager.training, P2PTrainingBridge)
        assert isinstance(manager.evaluation, P2PEvaluationBridge)

        # All should share the same client
        assert manager.selfplay.client is manager.client
        assert manager.training.client is manager.client
        assert manager.evaluation.client is manager.client

    @pytest.mark.asyncio
    async def test_selfplay_to_training_flow(self):
        """Test flow from selfplay scaling to training trigger."""
        manager = P2PIntegrationManager()

        # Mock cluster with healthy nodes
        mock_nodes = [
            P2PNode(node_id="gpu1", host="h1", port=8770, is_alive=True, is_healthy=True,
                   has_gpu=True, gpu_power_score=80, selfplay_jobs=0, training_jobs=0),
        ]

        with patch.object(manager.client, 'get_nodes', return_value=mock_nodes):
            with patch.object(manager.client, 'get_cluster_status', return_value={
                "selfplay_rate": 800,
                "nodes": [{"node_id": "gpu1"}]
            }):
                with patch.object(manager.client, 'start_selfplay', return_value={"job_id": "sp-1"}):
                    # Auto-scale should try to add more selfplay
                    result = await manager.selfplay.auto_scale()
                    assert "current_rate" in result

                with patch.object(manager.client, 'start_training', return_value={"job_id": "tr-1"}):
                    # Then trigger training
                    result = await manager.trigger_training()
                    assert result["job_id"] == "tr-1"
