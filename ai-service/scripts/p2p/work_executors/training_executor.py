"""Training work executor - handles GPU model training subprocess.

Extracted from P2POrchestrator._execute_claimed_work (Feb 2026).

Critical fixes preserved:
- Feb 2026: Awaits subprocess completion (not fire-and-forget)
- Feb 2026: Saves to candidate_ (not canonical_) to prevent untested overwrites
- Feb 2026: Parses loss from stdout and populates work_item["result"]
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.managers.job_orchestration_manager import JobOrchestrationManager

logger = logging.getLogger("p2p_orchestrator")


async def execute_training_work(
    work_item: dict[str, Any],
    config: dict[str, Any],
    node_id: str,
    ringrift_path: str | Path,
    job_orchestration: "JobOrchestrationManager | None" = None,
) -> bool:
    """Execute a training work item as a subprocess.

    Args:
        work_item: Full work item dict (modified in-place to add result data).
        config: Work config sub-dict (board_type, num_players, etc.).
        node_id: This node's identifier.
        ringrift_path: Path to ai-service root (used as subprocess cwd).
        job_orchestration: Optional manager for recording execution metrics.

    Returns:
        True on success, False on failure.
    """
    work_id = work_item.get("work_id", "")

    # Prevent coordinator from running training locally
    from scripts.p2p.managers.work_discovery_manager import _is_training_enabled_for_node
    if not _is_training_enabled_for_node():
        logger.info(f"Skipping training work {work_id}: training_enabled=false for this node")
        return True  # "handled" (just skipped)

    board_type = config.get("board_type", "square8")
    num_players = config.get("num_players", 2)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 256)
    learning_rate = config.get("learning_rate", 1e-3)
    model_version = config.get("model_version", "v2")

    config_key = f"{board_type}_{num_players}p"

    # Save to candidate_ instead of canonical_ to prevent overwriting the
    # production model before evaluation confirms improvement.
    if model_version and model_version != "v2":
        model_filename = f"candidate_{config_key}_{model_version}.pth"
    else:
        model_filename = f"candidate_{config_key}.pth"

    cmd = [
        sys.executable, "-m", "app.training.train",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--model-version", model_version,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--save-path", f"models/{model_filename}",
        "--allow-stale-data",
        "--max-data-age-hours", "168",  # 7 days tolerance
    ]

    logger.info(
        f"Executing training work {work_id}: {config_key} with {model_version} "
        f"(epochs={epochs}, batch={batch_size})"
    )

    # Await training subprocess and capture results.
    # Previously used asyncio.create_task() (fire-and-forget) which
    # returned True immediately, causing loss=0.0000 and empty model_path.
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(ringrift_path)),
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode() if stdout else ""

        if proc.returncode == 0:
            # Parse training output for loss value
            final_loss = 0.0
            training_samples = 0
            for line in reversed(output.splitlines()):
                if "loss" in line.lower() and "=" in line:
                    loss_match = re.search(r'loss[=:\s]+([0-9]+\.?[0-9]*)', line.lower())
                    if loss_match:
                        final_loss = float(loss_match.group(1))
                        break
                if "samples" in line.lower() and final_loss > 0:
                    samples_match = re.search(r'(\d+)\s*samples', line.lower())
                    if samples_match:
                        training_samples = int(samples_match.group(1))

            model_path = f"models/{model_filename}"
            logger.info(
                f"Training completed successfully: {config_key}/{model_version} "
                f"(work_id={work_id}, loss={final_loss:.4f}, samples={training_samples})"
            )

            # Populate work_item result so report_work_result sends real data
            work_item["result"] = {
                "model_path": model_path,
                "final_loss": final_loss,
                "training_samples": training_samples,
                "config_key": config_key,
                "model_version": model_version,
            }

            # Emit training completed event
            try:
                from app.distributed.data_events import DataEventType
                from app.coordination.event_router import emit_event
                emit_event(DataEventType.TRAINING_COMPLETED, {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_version": model_version,
                    "model_path": model_path,
                    "final_loss": final_loss,
                    "training_samples": training_samples,
                    "work_id": work_id,
                })
            except ImportError:
                pass
            return True
        else:
            truncated = output[:2000] if output else "no output"
            logger.error(
                f"Training failed: {config_key}/{model_version}: "
                f"returncode={proc.returncode}, output={truncated}"
            )
            return False
    except Exception as e:
        logger.exception(f"Training subprocess error for {config_key}: {e}")
        return False
