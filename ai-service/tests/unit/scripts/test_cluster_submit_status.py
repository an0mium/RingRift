import argparse
import json

import pytest

from app.coordination.slurm_backend import SlurmJobState, SlurmJobStatus, SlurmNode
from app.coordination.unified_scheduler import (
    Backend,
    JobState,
    JobType,
    UnifiedJob,
    UnifiedScheduler,
)
from scripts import cluster_submit


class _StubSlurmBackend:
    async def get_nodes(self, refresh: bool = True):
        return {
            "lambda-gh200-a": SlurmNode(
                name="lambda-gh200-a",
                partition="gpu-selfplay",
                state="idle",
                cpus=96,
                memory_mb=512000,
                features=["gh200"],
                gres="gpu:1",
            ),
            "lambda-gh200-b": SlurmNode(
                name="lambda-gh200-b",
                partition="gpu-selfplay",
                state="alloc",
                cpus=96,
                memory_mb=512000,
                features=["gh200"],
                gres="gpu:1",
            ),
        }

    async def get_jobs(self, refresh: bool = True):
        return {
            101: SlurmJobStatus(
                job_id=101,
                name="test-job-running",
                state=SlurmJobState.RUNNING,
                partition="gpu-selfplay",
                node="lambda-gh200-a",
                start_time=None,
                run_time=None,
            ),
            102: SlurmJobStatus(
                job_id=102,
                name="test-job-pending",
                state=SlurmJobState.PENDING,
                partition="gpu-selfplay",
                node=None,
                start_time=None,
                run_time=None,
            ),
        }


@pytest.mark.asyncio
async def test_cluster_submit_status_json_reflects_backend_counts(
    tmp_path,
    monkeypatch,
    capsys,
):
    scheduler = UnifiedScheduler(
        db_path=str(tmp_path / "unified_scheduler.db"),
        enable_slurm=True,
        enable_vast=False,
        enable_p2p=False,
    )
    scheduler._slurm_backend = _StubSlurmBackend()

    running_job = UnifiedJob(name="slurm-running", job_type=JobType.SELFPLAY)
    pending_job = UnifiedJob(name="slurm-pending", job_type=JobType.SELFPLAY)

    scheduler._record_job(running_job, Backend.SLURM)
    scheduler._record_job(pending_job, Backend.SLURM)
    scheduler._update_job(running_job.id, backend_job_id="101", state=JobState.QUEUED)
    scheduler._update_job(pending_job.id, backend_job_id="102", state=JobState.QUEUED)

    monkeypatch.setattr(cluster_submit, "get_scheduler", lambda: scheduler)

    args = argparse.Namespace(json=True, detailed=False)
    await cluster_submit.cmd_status(args)
    output = capsys.readouterr().out
    status = json.loads(output)

    assert status["slurm"]["jobs_running"] == 1
    assert status["slurm"]["jobs_pending"] == 1
    assert status["jobs"]["total"] == 2
    assert status["jobs"]["running"] == 1
    assert status["jobs"]["pending"] == 1
