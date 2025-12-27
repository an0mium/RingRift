# Deprecated Configs

## Config Templates (archived 2025-12-27)

These example/template files are no longer needed - the canonical config files
(`distributed_hosts.yaml`, `cluster.yaml`, etc.) serve as the source of truth:

- `cluster_nodes.env.example` - Legacy env file template (use distributed_hosts.yaml)
- `cluster_nodes.example.yaml` - Legacy cluster config template
- `cluster_nodes.yaml.example` - Duplicate template
- `distributed_hosts.example.yaml` - Template for distributed_hosts.yaml
- `distributed_hosts.template.yaml` - Another template variant
- `distributed_hosts.yaml.example` - Third template variant
- `orchestrator_hosts.example.sh` - Shell script template (use P2P instead)
- `p2p_orchestrator.env.example` - P2P env template (use distributed_hosts.yaml)
- `remote_hosts.example.yaml` - Remote hosts template
- `remote_hosts.yaml.example` - Duplicate template
- `selfplay_workers.example.yaml` - Worker config template
- `sync_hosts.env.example` - Sync hosts template (use distributed_hosts.yaml)
- `unified_loop.slurm.example.yaml` - SLURM template

All configuration now flows from `distributed_hosts.yaml` as single source of truth.

## Legacy Backups (archived 2025-12-26)

- pipeline.json.bak: retired 2025-12-26; legacy pipeline config referenced old hosts
  (ringrift-staging, ringrift-selfplay-extra). Keep only for historical reference.
- unified_loop.full.yaml.bak: retired 2025-12-26; merged optimized values into
  ai-service/config/unified_loop.yaml.
- unified_loop.optimized.yaml.bak: retired 2025-12-26; optimized-only config
  kept for reference after merging into unified_loop.yaml.
- nginx-cluster.conf.bak: retired 2025-12-26; Lambda-only nginx upstreams.
- cluster_nodes.env: retired 2025-12-27; referenced retired Lambda Labs nodes
  (H100, GH200, A10 - all terminated Dec 2025).
