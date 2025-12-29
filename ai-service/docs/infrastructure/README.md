# Infrastructure Documentation

Cluster setup, operations, and cloud infrastructure.

> **Update (Dec 28, 2025):** Lambda Labs account restored. 6 GH200 nodes (96GB each) now available for training.
> Current active providers: Lambda GH200 (6), Vast.ai (~14 nodes), RunPod (6), Nebius (3), Vultr (2), Hetzner (3), Local (2).
> Total: ~36 active nodes in `config/distributed_hosts.yaml`.

## Contents

| Document                                                                          | Description                     |
| --------------------------------------------------------------------------------- | ------------------------------- |
| [CLUSTER_SETUP_GUIDE](CLUSTER_SETUP_GUIDE.md)                                     | Cluster setup instructions      |
| [CLUSTER_OPERATIONS_RUNBOOK](CLUSTER_OPERATIONS_RUNBOOK.md)                       | Cluster operations              |
| [CLOUD_TRAINING_INFRASTRUCTURE_PLAN](CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md)       | Cloud infrastructure plan       |
| [OPERATIONAL_RUNBOOK](OPERATIONAL_RUNBOOK.md)                                     | General operations              |
| [DATA_VALIDATION](DATA_VALIDATION.md)                                             | Data validation framework       |
| [VAST_P2P_ORCHESTRATION](VAST_P2P_ORCHESTRATION.md)                               | Vast.ai P2P orchestration       |
| [VAST_LIFECYCLE](VAST_LIFECYCLE.md)                                               | Vast.ai instance lifecycle      |
| [P2P_ORCHESTRATOR_AUTH](P2P_ORCHESTRATOR_AUTH.md)                                 | P2P authentication              |
| [ORCHESTRATOR_SELECTION](ORCHESTRATOR_SELECTION.md)                               | Orchestrator selection          |
| [SLURM_BACKEND_DESIGN](SLURM_BACKEND_DESIGN.md)                                   | Slurm backend design            |
| [UNIFIED_AI_LOOP_DEPLOYMENT](UNIFIED_AI_LOOP_DEPLOYMENT.md)                       | Deployment guide                |
| [PIPELINE_ORCHESTRATOR](PIPELINE_ORCHESTRATOR.md)                                 | Pipeline orchestration          |
| [BOTTLENECK_ANALYSIS](BOTTLENECK_ANALYSIS.md)                                     | Performance bottlenecks         |
| `GPU_RULES_PARITY_AUDIT.md` (local-only, gitignored)                              | GPU/CPU parity audit            |
| [RESOURCE_MANAGEMENT](RESOURCE_MANAGEMENT.md)                                     | Resource management             |
| [CLUSTER_MONITOR_QUICKSTART](../../app/distributed/CLUSTER_MONITOR_QUICKSTART.md) | Cluster monitor quick reference |
