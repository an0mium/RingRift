import yaml

from app.coordination.node_policies import NodePolicyManager


def test_node_policies_load_and_match(tmp_path):
    config = {
        "default": {
            "allowed": ["training"],
            "priorities": {"training": 100},
        },
        "gpu_heavy": {
            "patterns": ["gpu-*"],
            "allowed": ["selfplay"],
            "priorities": {"selfplay": 90},
        },
        "overrides": {
            "node-1": {
                "allowed": ["training", "selfplay"],
                "priorities": {"training": 80},
            },
        },
    }

    config_path = tmp_path / "node_policies.yaml"
    config_path.write_text(yaml.safe_dump(config))

    manager = NodePolicyManager(config_path=config_path)

    assert manager.is_work_allowed("node-1", "training") is True
    assert manager.is_work_allowed("node-1", "cpu_cmaes") is False
    assert manager.is_work_allowed("gpu-7", "selfplay") is True
    assert manager.is_work_allowed("gpu-7", "training") is False
    assert manager.get_node_policy("node-1").get_priority("training") == 80
