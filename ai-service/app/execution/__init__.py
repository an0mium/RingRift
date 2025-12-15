"""Unified execution framework for the RingRift AI service.

This package provides abstracted execution backends for running commands
locally, via SSH, or on cloud workers. All orchestrators should use this
instead of implementing their own SSH/subprocess logic.

Usage:
    from app.execution import SSHExecutor, LocalExecutor, ExecutionResult

    # SSH execution
    executor = SSHExecutor(host="worker-1", user="ringrift")
    result = await executor.run("python scripts/run_selfplay.py")

    # Local execution
    executor = LocalExecutor()
    result = await executor.run("python scripts/train.py")

    # Check result
    if result.success:
        print(result.stdout)
    else:
        print(f"Failed: {result.stderr}")
"""

from app.execution.executor import (
    ExecutionResult,
    BaseExecutor,
    LocalExecutor,
    SSHExecutor,
    ExecutorPool,
    run_command,
    run_command_async,
    run_ssh_command,
    run_ssh_command_async,
)

__all__ = [
    "ExecutionResult",
    "BaseExecutor",
    "LocalExecutor",
    "SSHExecutor",
    "ExecutorPool",
    "run_command",
    "run_command_async",
    "run_ssh_command",
    "run_ssh_command_async",
]
