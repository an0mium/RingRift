"""Training scheduling (December 2025).

Consolidates scheduling from job_scheduler.py, duration_scheduler.py, and unified_scheduler.py.

Usage:
    from app.coordination.training.scheduler import (
        PriorityJobScheduler,
        JobPriority,
        DurationScheduler,
    )
"""

from __future__ import annotations

# Re-export from job_scheduler
from app.coordination.job_scheduler import (
    PriorityJobScheduler,
    JobPriority,
    ScheduledJob,
    HostDeadJobMigrator,
)

# Re-export from duration_scheduler
from app.coordination.duration_scheduler import (
    DurationScheduler,
    ScheduledTask,
    TaskDurationRecord,
    estimate_task_duration,
    can_schedule_task,
)

# Re-export from work_distributor
from app.coordination.work_distributor import (
    WorkDistributor,
)

# Re-export from unified_scheduler
from app.coordination.unified_scheduler import (
    UnifiedScheduler,
    get_scheduler as get_unified_scheduler,
)

__all__ = [
    # From job_scheduler
    "PriorityJobScheduler",
    "JobPriority",
    "ScheduledJob",
    "HostDeadJobMigrator",
    # From duration_scheduler
    "DurationScheduler",
    "ScheduledTask",
    "TaskDurationRecord",
    "estimate_task_duration",
    "can_schedule_task",
    # From work_distributor
    "WorkDistributor",
    # From unified_scheduler
    "UnifiedScheduler",
    "get_unified_scheduler",
]
