"""Consolidated SSH recovery utilities.

This module provides unified SSH execution and recovery command generation
for use by recovery daemons across the coordination layer.

Consolidates duplicated patterns from:
- connectivity_recovery_coordinator.py
- recovery_orchestrator.py
- availability/recovery_engine.py

Created: Jan 3, 2026
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Default SSH configuration
SSH_DEFAULT_TIMEOUT = 30
SSH_DEFAULT_CONNECT_TIMEOUT = 10
SSH_MAX_RETRIES = 3
SSH_BASE_DELAY = 1.0  # seconds
SSH_MAX_DELAY = 16.0  # seconds


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    user: str = "root"
    key_path: Optional[str] = None
    connect_timeout: int = SSH_DEFAULT_CONNECT_TIMEOUT
    command_timeout: int = SSH_DEFAULT_TIMEOUT


@dataclass
class SSHResult:
    """Result of SSH command execution."""
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    attempts: int = 1

    @property
    def output(self) -> str:
        """Combined output for convenience."""
        return self.stdout + self.stderr


@dataclass
class RecoveryCommandConfig:
    """Configuration for building recovery commands."""
    is_container: bool = False
    hostname: str = "unknown"
    authkey: Optional[str] = None
    ringrift_path: str = "~/ringrift/ai-service"


class SSHRecoveryHelper:
    """Unified SSH recovery helper.

    Provides:
    1. SSH command execution with retry and exponential backoff
    2. Standard recovery command builders (P2P, Tailscale)
    3. Verification utilities for recovery success
    """

    def __init__(
        self,
        max_retries: int = SSH_MAX_RETRIES,
        base_delay: float = SSH_BASE_DELAY,
        max_delay: float = SSH_MAX_DELAY,
    ):
        """Initialize SSH recovery helper.

        Args:
            max_retries: Maximum retry attempts for transient failures
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    async def execute_ssh_command(
        self,
        config: SSHConfig,
        command: str,
        timeout: Optional[int] = None,
        retry_on_timeout: bool = True,
    ) -> SSHResult:
        """Execute command via SSH with retry.

        Args:
            config: SSH connection configuration
            command: Command to execute
            timeout: Override command timeout
            retry_on_timeout: Whether to retry on timeout

        Returns:
            SSHResult with execution details
        """
        effective_timeout = timeout or config.command_timeout
        last_error: Optional[str] = None
        attempts = 0

        for attempt in range(self._max_retries):
            attempts = attempt + 1
            try:
                result = await self._execute_once(config, command, effective_timeout)

                if result.success:
                    if attempt > 0:
                        logger.info(
                            f"[SSHRecoveryHelper] SSH succeeded on attempt {attempts} "
                            f"to {config.host}"
                        )
                    result.attempts = attempts
                    return result

                # Command executed but returned non-zero - don't retry
                # (SSH connection worked, command failed)
                if result.exit_code != 255:  # 255 typically means SSH connection failed
                    result.attempts = attempts
                    return result

                last_error = result.error or result.stderr

            except asyncio.TimeoutError:
                last_error = "SSH timeout"
                if not retry_on_timeout:
                    return SSHResult(
                        success=False,
                        exit_code=124,  # Timeout exit code
                        error=last_error,
                        attempts=attempts,
                    )
            except Exception as e:
                last_error = str(e)

            # Apply backoff before retry
            if attempt < self._max_retries - 1:
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                logger.warning(
                    f"[SSHRecoveryHelper] SSH attempt {attempts}/{self._max_retries} "
                    f"to {config.host} failed: {last_error}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"[SSHRecoveryHelper] SSH to {config.host} failed after {self._max_retries} "
            f"attempts: {last_error}"
        )
        return SSHResult(
            success=False,
            exit_code=1,
            error=f"SSH failed after {self._max_retries} attempts: {last_error}",
            attempts=attempts,
        )

    async def _execute_once(
        self,
        config: SSHConfig,
        command: str,
        timeout: int,
    ) -> SSHResult:
        """Execute SSH command once without retry."""
        ssh_cmd = [
            "ssh",
            "-o", f"ConnectTimeout={config.connect_timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]

        if config.port != 22:
            ssh_cmd.extend(["-p", str(config.port)])

        if config.key_path:
            ssh_cmd.extend(["-i", os.path.expanduser(config.key_path)])

        ssh_cmd.append(f"{config.user}@{config.host}")
        ssh_cmd.append(command)

        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""

        return SSHResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode or 0,
            stdout=stdout_str,
            stderr=stderr_str,
        )

    # =========================================================================
    # Recovery Command Builders
    # =========================================================================

    def build_tailscale_recovery_command(
        self,
        config: RecoveryCommandConfig,
    ) -> str:
        """Build Tailscale recovery command.

        Handles both container (userspace networking) and host (systemctl) environments.

        Args:
            config: Recovery command configuration

        Returns:
            Shell command to restart Tailscale and verify connectivity
        """
        authkey = config.authkey or os.environ.get("TAILSCALE_AUTH_KEY", "")
        authkey_arg = f"--authkey={authkey}" if authkey else ""

        if config.is_container:
            # Container: Use userspace networking (Vast.ai, RunPod, etc.)
            return f"""
pkill -9 tailscaled 2>/dev/null || true
sleep 2
mkdir -p /var/lib/tailscale /var/run/tailscale
nohup tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale > /tmp/tailscaled.log 2>&1 &
sleep 5
tailscale up {authkey_arg} --accept-routes --hostname='{config.hostname}'
tailscale ip -4
"""
        else:
            # Regular host: Use systemctl (Lambda, Nebius, etc.)
            return f"""
systemctl restart tailscaled 2>/dev/null || {{
    pkill -9 tailscaled
    sleep 2
    tailscaled --state=/var/lib/tailscale/tailscaled.state &
    sleep 5
}}
tailscale up {authkey_arg} --accept-routes --hostname='{config.hostname}'
tailscale ip -4
"""

    def build_p2p_restart_command(
        self,
        config: RecoveryCommandConfig,
    ) -> str:
        """Build P2P orchestrator restart command.

        Args:
            config: Recovery command configuration

        Returns:
            Shell command to kill and restart P2P orchestrator
        """
        # Jan 2026: Switched to nohup and added screen cleanup to prevent dead sessions
        return f"""
pkill -f 'python.*p2p_orchestrator' 2>/dev/null || true
screen -X -S p2p quit 2>/dev/null || true
screen -wipe 2>/dev/null || true
sleep 2
cd {config.ringrift_path} && \\
mkdir -p logs && \\
PYTHONPATH=. nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &
sleep 3
pgrep -f p2p_orchestrator
"""

    def build_health_check_command(self) -> str:
        """Build command to check P2P health status."""
        return "curl -s --connect-timeout 5 http://localhost:8770/status | head -c 500"

    # =========================================================================
    # ControlMaster Cache Management
    # =========================================================================

    def get_control_master_path(self, host: str, user: str = "root") -> str:
        """Get the ControlMaster socket path for a host.

        Args:
            host: Target host
            user: SSH user

        Returns:
            Path to the ControlMaster socket file
        """
        # Standard ControlMaster path pattern: ~/.ssh/cm-%r@%h:%p
        import os
        ssh_dir = os.path.expanduser("~/.ssh")
        # Use the common ControlPath pattern
        return os.path.join(ssh_dir, f"cm-{user}@{host}:22")

    async def invalidate_control_master(
        self,
        host: str,
        user: str = "root",
        port: int = 22,
    ) -> bool:
        """Invalidate cached SSH ControlMaster connection.

        Call this after an authentication failure to ensure the next
        connection attempt doesn't use a cached (potentially invalid) socket.

        Args:
            host: Target host
            user: SSH user
            port: SSH port

        Returns:
            True if socket was removed or didn't exist, False on error
        """
        import os

        # Try multiple common ControlPath patterns
        patterns = [
            f"cm-{user}@{host}:{port}",
            f"cm-{user}@{host}:*",
            f"sockets/{host}",
            f"sockets/{user}@{host}",
        ]

        ssh_dir = os.path.expanduser("~/.ssh")
        removed = False

        for pattern in patterns:
            socket_path = os.path.join(ssh_dir, pattern)

            # Handle wildcard patterns
            if "*" in pattern:
                import glob
                for path in glob.glob(socket_path):
                    try:
                        os.remove(path)
                        logger.info(f"[SSHRecoveryHelper] Removed ControlMaster socket: {path}")
                        removed = True
                    except OSError as e:
                        logger.debug(f"[SSHRecoveryHelper] Could not remove {path}: {e}")
            elif os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                    logger.info(f"[SSHRecoveryHelper] Removed ControlMaster socket: {socket_path}")
                    removed = True
                except OSError as e:
                    logger.warning(f"[SSHRecoveryHelper] Failed to remove {socket_path}: {e}")
                    return False

        # Also try to kill the ControlMaster process with ssh -O exit
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-O", "exit",
                "-o", "ControlPath=~/.ssh/cm-%r@%h:%p",
                f"{user}@{host}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            if proc.returncode == 0:
                logger.info(f"[SSHRecoveryHelper] Closed ControlMaster for {user}@{host}")
                removed = True
        except (asyncio.TimeoutError, OSError) as e:
            logger.debug(f"[SSHRecoveryHelper] ssh -O exit failed (may be normal): {e}")

        if removed:
            logger.info(f"[SSHRecoveryHelper] ControlMaster invalidated for {user}@{host}")
        else:
            logger.debug(f"[SSHRecoveryHelper] No ControlMaster found for {user}@{host}")

        return True

    async def invalidate_all_control_masters(self) -> int:
        """Invalidate all cached SSH ControlMaster connections.

        Returns:
            Number of sockets removed
        """
        import os
        import glob

        ssh_dir = os.path.expanduser("~/.ssh")
        removed = 0

        # Common ControlMaster socket patterns
        patterns = [
            os.path.join(ssh_dir, "cm-*"),
            os.path.join(ssh_dir, "sockets/*"),
        ]

        for pattern in patterns:
            for socket_path in glob.glob(pattern):
                try:
                    if os.path.exists(socket_path):
                        os.remove(socket_path)
                        logger.info(f"[SSHRecoveryHelper] Removed: {socket_path}")
                        removed += 1
                except OSError as e:
                    logger.debug(f"[SSHRecoveryHelper] Could not remove {socket_path}: {e}")

        logger.info(f"[SSHRecoveryHelper] Invalidated {removed} ControlMaster sockets")
        return removed

    # =========================================================================
    # Verification Utilities
    # =========================================================================

    def verify_tailscale_output(self, output: str) -> bool:
        """Verify Tailscale recovery output indicates success.

        Args:
            output: Combined stdout/stderr from recovery command

        Returns:
            True if Tailscale IP is present in output
        """
        # Tailscale IPs start with 100.x.x.x
        return "100." in output

    def verify_p2p_output(self, output: str) -> bool:
        """Verify P2P restart output indicates success.

        Args:
            output: Combined stdout/stderr from restart command

        Returns:
            True if P2P process is running
        """
        # pgrep returns PID if process is running
        try:
            lines = output.strip().split("\n")
            for line in lines:
                if line.strip().isdigit():
                    return True
        except (ValueError, AttributeError):
            pass
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

_default_helper: Optional[SSHRecoveryHelper] = None


def get_ssh_recovery_helper() -> SSHRecoveryHelper:
    """Get singleton SSH recovery helper instance."""
    global _default_helper
    if _default_helper is None:
        _default_helper = SSHRecoveryHelper()
    return _default_helper


async def execute_ssh_recovery(
    host: str,
    command: str,
    port: int = 22,
    user: str = "root",
    key_path: Optional[str] = None,
    timeout: int = SSH_DEFAULT_TIMEOUT,
) -> SSHResult:
    """Convenience function for SSH command execution with retry.

    Args:
        host: Target host
        command: Command to execute
        port: SSH port
        user: SSH user
        key_path: Path to SSH key
        timeout: Command timeout

    Returns:
        SSHResult with execution details
    """
    helper = get_ssh_recovery_helper()
    config = SSHConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        command_timeout=timeout,
    )
    return await helper.execute_ssh_command(config, command)


async def restart_tailscale_via_ssh(
    host: str,
    hostname: str,
    is_container: bool = False,
    port: int = 22,
    user: str = "root",
    key_path: Optional[str] = None,
    timeout: int = 60,
) -> SSHResult:
    """Restart Tailscale on a remote host via SSH.

    Args:
        host: Target SSH host
        hostname: Tailscale hostname for the node
        is_container: Whether the host is a container (userspace networking)
        port: SSH port
        user: SSH user
        key_path: Path to SSH key
        timeout: Command timeout

    Returns:
        SSHResult with success/failure and output
    """
    helper = get_ssh_recovery_helper()

    cmd_config = RecoveryCommandConfig(
        is_container=is_container,
        hostname=hostname,
    )
    command = helper.build_tailscale_recovery_command(cmd_config)

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        command_timeout=timeout,
    )

    result = await helper.execute_ssh_command(ssh_config, command)

    # Enhanced success check
    if result.success or result.exit_code == 0:
        result.success = helper.verify_tailscale_output(result.output)

    return result


async def invalidate_control_master(
    host: str,
    user: str = "root",
    port: int = 22,
) -> bool:
    """Convenience function to invalidate SSH ControlMaster for a host.

    Args:
        host: Target host
        user: SSH user
        port: SSH port

    Returns:
        True if successful
    """
    helper = get_ssh_recovery_helper()
    return await helper.invalidate_control_master(host, user, port)


async def handle_ssh_auth_failure(
    node_id: str,
    host: str,
    error_message: str,
    exit_code: int = 1,
    user: str = "root",
) -> bool:
    """Handle an SSH authentication failure by invalidating caches and resetting circuit.

    This combines:
    1. Error classification to confirm it's an auth failure
    2. ControlMaster invalidation
    3. Circuit breaker reset for SSH transport

    Args:
        node_id: Node ID for circuit breaker
        host: SSH host for ControlMaster
        error_message: The SSH error message
        exit_code: SSH exit code
        user: SSH user

    Returns:
        True if auth failure was handled, False if not an auth failure
    """
    # Import here to avoid circular imports
    try:
        from app.coordination.ssh_error_classifier import classify_ssh_error
        from app.coordination.transport_circuit_breaker import get_transport_circuit_breaker
    except ImportError:
        logger.warning("[SSHRecoveryHelper] Could not import classifier or circuit breaker")
        return False

    # Classify the error
    classification = classify_ssh_error(error_message, exit_code)

    if not classification.should_invalidate_control_master:
        logger.debug(
            f"[SSHRecoveryHelper] SSH error is not auth failure: "
            f"{classification.error_type.value} ({classification.matched_pattern})"
        )
        return False

    logger.info(
        f"[SSHRecoveryHelper] Handling SSH auth failure for {node_id}/{host}: "
        f"{classification.error_type.value} ({classification.matched_pattern})"
    )

    # Invalidate ControlMaster
    helper = get_ssh_recovery_helper()
    await helper.invalidate_control_master(host, user)

    # Reset SSH circuit breaker
    if classification.should_reset_circuit:
        breaker = get_transport_circuit_breaker()
        breaker.reset_transport_circuit(
            node_id,
            "ssh",
            reason=f"auth_failure:{classification.matched_pattern}",
        )

    return True


async def restart_p2p_via_ssh(
    host: str,
    port: int = 22,
    user: str = "root",
    key_path: Optional[str] = None,
    ringrift_path: str = "~/ringrift/ai-service",
    timeout: int = 30,
) -> SSHResult:
    """Restart P2P orchestrator on a remote host via SSH.

    Args:
        host: Target SSH host
        port: SSH port
        user: SSH user
        key_path: Path to SSH key
        ringrift_path: Path to ringrift on remote host
        timeout: Command timeout

    Returns:
        SSHResult with success/failure and output
    """
    helper = get_ssh_recovery_helper()

    cmd_config = RecoveryCommandConfig(ringrift_path=ringrift_path)
    command = helper.build_p2p_restart_command(cmd_config)

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        command_timeout=timeout,
    )

    result = await helper.execute_ssh_command(ssh_config, command)

    # Enhanced success check
    if result.success or result.exit_code == 0:
        result.success = helper.verify_p2p_output(result.output)

    return result
