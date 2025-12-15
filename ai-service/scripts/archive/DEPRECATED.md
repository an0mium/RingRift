# Deprecated Scripts Archive

This directory contains scripts that have been superseded by `unified_ai_loop.py` and related infrastructure. These scripts are kept for reference but should not be used in production.

## Archived Scripts

### cluster_orchestrator.py

**Archived:** 2025-12-15
**Reason:** Functionality consolidated into `unified_ai_loop.py`
**Replacement:** Use `unified_ai_loop.py --help` for cluster orchestration

The cluster orchestrator managed compute resources with resource-aware scheduling. This functionality is now integrated into:

- `unified_ai_loop.py` - Main orchestration loop
- `p2p_orchestrator.py` - P2P cluster coordination
- `app/coordination/` - Task coordination and scheduling

### pipeline_orchestrator.py

**Archived:** 2025-12-15
**Reason:** Functionality consolidated into `unified_ai_loop.py`
**Replacement:** Use `unified_ai_loop.py --help` for pipeline management

The pipeline orchestrator managed the complete AI training pipeline. Its phases are now handled by:

- Data ingestion → `UnifiedLoop._sync_data_from_hosts()`
- Training → `UnifiedLoop._run_training()`
- Evaluation → `UnifiedLoop._run_shadow_evaluation()`
- Promotion → `UnifiedLoop._check_promotion()`
- Elo calibration → `UnifiedLoop._run_calibration_analysis()`

### Other Archived Scripts (from previous consolidation)

- `master_self_improvement.py` - Superseded by unified_ai_loop.py
- `unified_improvement_controller.py` - Superseded by unified_ai_loop.py
- `integrated_self_improvement.py` - Superseded by unified_ai_loop.py
- `export_replay_dataset.py` - Functionality in data pipeline
- `validate_canonical_training_sources.py` - Functionality in validation module

## Migration Guide

To migrate from deprecated scripts to the unified system:

1. **For cluster management:**

   ```bash
   # Old
   python scripts/cluster_orchestrator.py

   # New
   python scripts/unified_ai_loop.py --coordinator
   ```

2. **For pipeline execution:**

   ```bash
   # Old
   python scripts/pipeline_orchestrator.py --phase selfplay

   # New
   python scripts/unified_ai_loop.py  # Handles all phases automatically
   ```

3. **For configuration:**
   - All settings are now in `config/unified_loop.yaml`
   - Use `app/config/unified_config.py` for programmatic access

## Questions?

If you need functionality from these scripts that isn't available in the unified system, please open an issue or check if the feature should be added to `unified_ai_loop.py`.
