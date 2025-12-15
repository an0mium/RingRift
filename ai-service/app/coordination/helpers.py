"""Coordination Helper Module - Safe coordination utilities.

This module provides safe, reusable coordination functions to eliminate duplicate
try/except import patterns found across 13+ scripts in the codebase.

Instead of this pattern in every script:
    try:
        from app.coordination import (
            TaskCoordinator, TaskType, can_spawn,
            OrchestratorRole, acquire_orchestrator_role, get_registry,
        )
        HAS_COORDINATION = True
    except ImportError:
        HAS_COORDINATION = False
        TaskCoordinator = None
        TaskType = None

Use this:
    from app.coordination.helpers import (
        has_coordination, can_spawn_safe, register_task_safe,
        acquire_role_safe, has_role, get_coordinator_safe,
    )

Usage:
    # Check if coordination is available
    if has_coordination():
        coordinator = get_coordinator_safe()

    # Or use safe wrappers that handle unavailability gracefully
    allowed, reason = can_spawn_safe(TaskType.SELFPLAY, "node-1")
    if allowed:
        register_task_safe(task_id, TaskType.SELFPLAY, "node-1", os.getpid())
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Optional, Tuple, List, Dict, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Try to import coordination components
_HAS_COORDINATION = False
_TaskCoordinator = None
_TaskType = None
_TaskLimits = None
_OrchestratorRole = None
_OrchestratorRegistry = None
_Safeguards = None
_CircuitBreaker = None
_CircuitState = None

# Import functions
_can_spawn = None
_get_coordinator = None
_get_registry = None
_acquire_orchestrator_role = None
_release_orchestrator_role = None
_check_before_spawn = None

try:
    from app.coordination import (
        TaskCoordinator,
        TaskType,
        TaskLimits,
        OrchestratorRole,
        OrchestratorRegistry,
        Safeguards,
        CircuitBreaker,
        CircuitState,
        can_spawn,
        get_coordinator,
        get_registry,
        acquire_orchestrator_role,
        release_orchestrator_role,
        check_before_spawn,
    )
    _HAS_COORDINATION = True
    _TaskCoordinator = TaskCoordinator
    _TaskType = TaskType
    _TaskLimits = TaskLimits
    _OrchestratorRole = OrchestratorRole
    _OrchestratorRegistry = OrchestratorRegistry
    _Safeguards = Safeguards
    _CircuitBreaker = CircuitBreaker
    _CircuitState = CircuitState
    _can_spawn = can_spawn
    _get_coordinator = get_coordinator
    _get_registry = get_registry
    _acquire_orchestrator_role = acquire_orchestrator_role
    _release_orchestrator_role = release_orchestrator_role
    _check_before_spawn = check_before_spawn
except ImportError as e:
    logger.debug(f"Coordination module not available: {e}")


def has_coordination() -> bool:
    """Check if the coordination module is available.

    Returns:
        True if app.coordination is importable, False otherwise.
    """
    return _HAS_COORDINATION


def get_task_types():
    """Get the TaskType enum if available.

    Returns:
        TaskType enum or None if not available.
    """
    return _TaskType


def get_orchestrator_roles():
    """Get the OrchestratorRole enum if available.

    Returns:
        OrchestratorRole enum or None if not available.
    """
    return _OrchestratorRole


# =============================================================================
# Coordinator Functions
# =============================================================================

def get_coordinator_safe() -> Optional[Any]:
    """Get the task coordinator instance if available.

    Returns:
        TaskCoordinator instance or None if not available.
    """
    if not _HAS_COORDINATION or _get_coordinator is None:
        return None
    try:
        return _get_coordinator()
    except Exception as e:
        logger.debug(f"Failed to get coordinator: {e}")
        return None


def can_spawn_safe(
    task_type: Any,
    node_id: Optional[str] = None
) -> Tuple[bool, str]:
    """Safely check if a task can be spawned.

    Args:
        task_type: TaskType enum value
        node_id: Node identifier (defaults to hostname)

    Returns:
        Tuple of (allowed: bool, reason: str)
        If coordination unavailable, returns (True, "coordination_unavailable")
    """
    if not _HAS_COORDINATION or _can_spawn is None:
        return (True, "coordination_unavailable")

    if node_id is None:
        node_id = socket.gethostname()

    try:
        return _can_spawn(task_type, node_id)
    except Exception as e:
        logger.warning(f"can_spawn check failed: {e}")
        return (True, f"check_failed: {e}")


def register_task_safe(
    task_id: str,
    task_type: Any,
    node_id: Optional[str] = None,
    pid: Optional[int] = None
) -> bool:
    """Safely register a task with the coordinator.

    Args:
        task_id: Unique task identifier
        task_type: TaskType enum value
        node_id: Node identifier (defaults to hostname)
        pid: Process ID (defaults to current process)

    Returns:
        True if registration succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    if node_id is None:
        node_id = socket.gethostname()
    if pid is None:
        pid = os.getpid()

    try:
        coordinator.register_task(task_id, task_type, node_id, pid)
        return True
    except Exception as e:
        logger.warning(f"Task registration failed: {e}")
        return False


def complete_task_safe(task_id: str) -> bool:
    """Safely mark a task as completed.

    Args:
        task_id: Task identifier to complete

    Returns:
        True if completion succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    try:
        coordinator.complete_task(task_id)
        return True
    except Exception as e:
        logger.warning(f"Task completion failed: {e}")
        return False


def fail_task_safe(task_id: str, error: str = "") -> bool:
    """Safely mark a task as failed.

    Args:
        task_id: Task identifier to fail
        error: Error message

    Returns:
        True if operation succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    try:
        coordinator.fail_task(task_id, error)
        return True
    except Exception as e:
        logger.warning(f"Task failure recording failed: {e}")
        return False


# =============================================================================
# Orchestrator Role Functions
# =============================================================================

def get_registry_safe() -> Optional[Any]:
    """Get the orchestrator registry if available.

    Returns:
        OrchestratorRegistry instance or None if not available.
    """
    if not _HAS_COORDINATION or _get_registry is None:
        return None
    try:
        return _get_registry()
    except Exception as e:
        logger.debug(f"Failed to get registry: {e}")
        return None


def acquire_role_safe(role: Any) -> bool:
    """Safely attempt to acquire an orchestrator role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role was acquired, False otherwise.
    """
    if not _HAS_COORDINATION or _acquire_orchestrator_role is None:
        return False

    try:
        return _acquire_orchestrator_role(role)
    except Exception as e:
        logger.warning(f"Role acquisition failed: {e}")
        return False


def release_role_safe(role: Any) -> bool:
    """Safely release an orchestrator role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role was released, False otherwise.
    """
    if not _HAS_COORDINATION or _release_orchestrator_role is None:
        return False

    try:
        _release_orchestrator_role(role)
        return True
    except Exception as e:
        logger.warning(f"Role release failed: {e}")
        return False


def has_role(role: Any) -> bool:
    """Check if the current process holds a role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role is held, False otherwise.
    """
    registry = get_registry_safe()
    if registry is None:
        return False

    try:
        return registry.is_role_held(role)
    except Exception as e:
        logger.debug(f"Role check failed: {e}")
        return False


def get_role_holder(role: Any) -> Optional[Any]:
    """Get information about who holds a role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        OrchestratorInfo if role is held, None otherwise.
    """
    registry = get_registry_safe()
    if registry is None:
        return None

    try:
        return registry.get_role_holder(role)
    except Exception:
        return None


# =============================================================================
# Safeguards Functions
# =============================================================================

def check_spawn_allowed(
    task_type: str = "unknown",
    config_key: str = ""
) -> Tuple[bool, str]:
    """Check safeguards before spawning a task.

    Args:
        task_type: Type of task being spawned
        config_key: Configuration key (e.g., "square8_2p")

    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    if not _HAS_COORDINATION or _check_before_spawn is None:
        return (True, "safeguards_unavailable")

    try:
        return _check_before_spawn(task_type, config_key)
    except Exception as e:
        logger.warning(f"Safeguard check failed: {e}")
        return (True, f"check_failed: {e}")


def get_safeguards() -> Optional[Any]:
    """Get the Safeguards instance if available.

    Returns:
        Safeguards instance or None if not available.
    """
    if not _HAS_COORDINATION or _Safeguards is None:
        return None
    try:
        return _Safeguards()
    except Exception:
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def get_current_node_id() -> str:
    """Get the current node's identifier.

    Returns:
        Hostname of current machine.
    """
    return socket.gethostname()


def is_unified_loop_running() -> bool:
    """Check if a unified loop orchestrator is already running.

    Useful for daemons that should defer to the main orchestrator.

    Returns:
        True if unified loop holds the role, False otherwise.
    """
    if not _HAS_COORDINATION or _OrchestratorRole is None:
        return False

    try:
        return has_role(_OrchestratorRole.UNIFIED_LOOP)
    except Exception:
        return False


def warn_if_orchestrator_running(daemon_name: str = "daemon") -> None:
    """Print a warning if the unified orchestrator is already running.

    Args:
        daemon_name: Name of the daemon for the warning message.
    """
    if not _HAS_COORDINATION or _OrchestratorRole is None:
        return

    registry = get_registry_safe()
    if registry is None:
        return

    try:
        if registry.is_role_held(_OrchestratorRole.UNIFIED_LOOP):
            holder = registry.get_role_holder(_OrchestratorRole.UNIFIED_LOOP)
            existing_pid = holder.pid if holder else "unknown"
            print(f"[{daemon_name}] WARNING: Unified orchestrator is running (PID {existing_pid})")
            print(f"[{daemon_name}] The orchestrator handles this work - this {daemon_name} may duplicate work")
    except Exception:
        pass


# =============================================================================
# Re-exports for convenience
# =============================================================================

# These allow direct use of the types when coordination is available
TaskCoordinator = _TaskCoordinator
TaskType = _TaskType
TaskLimits = _TaskLimits
OrchestratorRole = _OrchestratorRole
OrchestratorRegistry = _OrchestratorRegistry
Safeguards = _Safeguards
CircuitBreaker = _CircuitBreaker
CircuitState = _CircuitState
