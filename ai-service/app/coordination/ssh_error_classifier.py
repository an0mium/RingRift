"""SSH Error Classifier - Distinguish auth failures from transient errors.

January 2026: Created for Phase 3.1 of automation hardening.

Problem:
    When SSH connections fail, the circuit breaker treats all failures the same.
    But auth failures (wrong key, permission denied) are permanent and need
    different handling than transient errors (timeout, connection refused).

Solution:
    Classify SSH errors by type and provide appropriate recovery actions:
    - AUTH_FAILURE: Permanent - invalidate ControlMaster, don't retry
    - TRANSIENT: Temporary - retry with backoff
    - NETWORK: Network issue - try alternate transport
    - UNKNOWN: Can't determine - treat as transient

Usage:
    from app.coordination.ssh_error_classifier import (
        SSHErrorClassifier,
        SSHErrorType,
        classify_ssh_error,
    )

    classifier = SSHErrorClassifier()
    result = classifier.classify(error_message, exit_code)

    if result.error_type == SSHErrorType.AUTH_FAILURE:
        invalidate_control_master(host)
        circuit_breaker.reset_transport_circuit(node, "ssh")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SSHErrorType(Enum):
    """Classification of SSH error types."""

    AUTH_FAILURE = "auth_failure"  # Permission denied, key rejected
    TRANSIENT = "transient"  # Timeout, connection reset, temp failure
    NETWORK = "network"  # Host unreachable, DNS failure
    HOST_KEY = "host_key"  # Host key verification failed
    CONFIG = "config"  # Bad configuration, invalid options
    UNKNOWN = "unknown"  # Cannot classify


@dataclass
class SSHErrorClassification:
    """Result of SSH error classification."""

    error_type: SSHErrorType
    should_retry: bool
    should_invalidate_control_master: bool
    should_reset_circuit: bool
    recommended_action: str
    confidence: float  # 0.0 to 1.0
    matched_pattern: Optional[str] = None

    @property
    def is_permanent(self) -> bool:
        """Check if this is a permanent (non-recoverable) error."""
        return self.error_type in (SSHErrorType.AUTH_FAILURE, SSHErrorType.HOST_KEY)


# Auth failure patterns - these indicate permanent issues
AUTH_FAILURE_PATTERNS = [
    (r"Permission denied", "permission_denied"),
    (r"publickey.*denied", "publickey_denied"),
    (r"Authentication failed", "auth_failed"),
    (r"Too many authentication failures", "too_many_failures"),
    (r"no matching key exchange method", "key_exchange_mismatch"),
    (r"no matching cipher", "cipher_mismatch"),
    (r"no matching MAC", "mac_mismatch"),
    (r"key_load_public: No such file", "key_not_found"),
    (r"Could not load host key", "host_key_load_failed"),
    (r"sign_and_send_pubkey: signing failed", "signing_failed"),
    (r"Agent refused operation", "agent_refused"),
]

# Transient error patterns - these may resolve with retry
TRANSIENT_PATTERNS = [
    (r"Connection timed out", "timeout"),
    (r"Connection reset by peer", "reset"),
    (r"Connection refused", "refused"),
    (r"Broken pipe", "broken_pipe"),
    (r"Software caused connection abort", "abort"),
    (r"Connection closed", "closed"),
    (r"ssh_exchange_identification", "exchange_failed"),
    (r"kex_exchange_identification", "kex_failed"),
    (r"remote side unexpectedly closed", "unexpected_close"),
    (r"packet_write_wait", "packet_write_wait"),
    (r"Write failed: Broken pipe", "write_failed"),
    (r"Read from remote host", "read_failed"),
]

# Network error patterns
NETWORK_PATTERNS = [
    (r"No route to host", "no_route"),
    (r"Network is unreachable", "network_unreachable"),
    (r"Name or service not known", "dns_failure"),
    (r"Could not resolve hostname", "hostname_unresolved"),
    (r"Temporary failure in name resolution", "dns_temp_failure"),
    (r"Host is down", "host_down"),
    (r"getaddrinfo", "getaddrinfo_failed"),
]

# Host key patterns
HOST_KEY_PATTERNS = [
    (r"Host key verification failed", "host_key_verification"),
    (r"REMOTE HOST IDENTIFICATION HAS CHANGED", "host_key_changed"),
    (r"Offending .* key", "offending_key"),
    (r"WARNING: POSSIBLE DNS SPOOFING DETECTED", "dns_spoofing"),
    (r"known_hosts", "known_hosts_issue"),
]

# Config error patterns
CONFIG_PATTERNS = [
    (r"Bad configuration option", "bad_option"),
    (r"command not found", "command_not_found"),
    (r"No such file or directory", "file_not_found"),
    (r"Invalid configuration", "invalid_config"),
]


class SSHErrorClassifier:
    """Classifier for SSH error messages.

    Analyzes SSH error output and exit codes to determine the error type
    and appropriate recovery action.
    """

    def __init__(self):
        """Initialize the classifier with compiled regex patterns."""
        self._auth_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in AUTH_FAILURE_PATTERNS
        ]
        self._transient_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in TRANSIENT_PATTERNS
        ]
        self._network_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in NETWORK_PATTERNS
        ]
        self._host_key_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in HOST_KEY_PATTERNS
        ]
        self._config_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in CONFIG_PATTERNS
        ]

    def classify(
        self,
        error_message: str,
        exit_code: int = 1,
        stderr: str = "",
    ) -> SSHErrorClassification:
        """Classify an SSH error.

        Args:
            error_message: The error message or combined output
            exit_code: SSH exit code (255 typically means connection failed)
            stderr: Separate stderr output if available

        Returns:
            SSHErrorClassification with error type and recommended actions
        """
        # Combine all error text for pattern matching
        full_text = f"{error_message}\n{stderr}"

        # Check patterns in priority order
        # Auth failures first (permanent)
        for pattern, name in self._auth_patterns:
            if pattern.search(full_text):
                return SSHErrorClassification(
                    error_type=SSHErrorType.AUTH_FAILURE,
                    should_retry=False,
                    should_invalidate_control_master=True,
                    should_reset_circuit=True,
                    recommended_action="Invalidate cached credentials and check SSH keys",
                    confidence=0.95,
                    matched_pattern=name,
                )

        # Host key issues (permanent until manually resolved)
        for pattern, name in self._host_key_patterns:
            if pattern.search(full_text):
                return SSHErrorClassification(
                    error_type=SSHErrorType.HOST_KEY,
                    should_retry=False,
                    should_invalidate_control_master=True,
                    should_reset_circuit=True,
                    recommended_action="Update known_hosts file or verify host identity",
                    confidence=0.95,
                    matched_pattern=name,
                )

        # Network errors (may recover with alternate transport)
        for pattern, name in self._network_patterns:
            if pattern.search(full_text):
                return SSHErrorClassification(
                    error_type=SSHErrorType.NETWORK,
                    should_retry=True,
                    should_invalidate_control_master=False,
                    should_reset_circuit=False,  # Try alternate transport
                    recommended_action="Try alternate transport (Tailscale, HTTP)",
                    confidence=0.90,
                    matched_pattern=name,
                )

        # Config errors (permanent until config is fixed)
        for pattern, name in self._config_patterns:
            if pattern.search(full_text):
                return SSHErrorClassification(
                    error_type=SSHErrorType.CONFIG,
                    should_retry=False,
                    should_invalidate_control_master=False,
                    should_reset_circuit=False,
                    recommended_action="Check SSH configuration and command syntax",
                    confidence=0.85,
                    matched_pattern=name,
                )

        # Transient errors (may recover with retry)
        for pattern, name in self._transient_patterns:
            if pattern.search(full_text):
                return SSHErrorClassification(
                    error_type=SSHErrorType.TRANSIENT,
                    should_retry=True,
                    should_invalidate_control_master=False,
                    should_reset_circuit=False,
                    recommended_action="Retry with exponential backoff",
                    confidence=0.85,
                    matched_pattern=name,
                )

        # Exit code 255 without other patterns is typically connection failure
        if exit_code == 255:
            return SSHErrorClassification(
                error_type=SSHErrorType.TRANSIENT,
                should_retry=True,
                should_invalidate_control_master=False,
                should_reset_circuit=False,
                recommended_action="SSH connection failed, retry or try alternate transport",
                confidence=0.60,
                matched_pattern="exit_code_255",
            )

        # Unknown error
        return SSHErrorClassification(
            error_type=SSHErrorType.UNKNOWN,
            should_retry=True,  # Default to retry
            should_invalidate_control_master=False,
            should_reset_circuit=False,
            recommended_action="Unknown error, retry with caution",
            confidence=0.30,
            matched_pattern=None,
        )

    def is_auth_failure(self, error_message: str, exit_code: int = 1) -> bool:
        """Quick check if error is an authentication failure."""
        result = self.classify(error_message, exit_code)
        return result.error_type == SSHErrorType.AUTH_FAILURE

    def is_retryable(self, error_message: str, exit_code: int = 1) -> bool:
        """Quick check if error is retryable."""
        result = self.classify(error_message, exit_code)
        return result.should_retry


# Singleton instance
_instance: SSHErrorClassifier | None = None


def get_ssh_error_classifier() -> SSHErrorClassifier:
    """Get the singleton SSH error classifier."""
    global _instance
    if _instance is None:
        _instance = SSHErrorClassifier()
    return _instance


def classify_ssh_error(
    error_message: str,
    exit_code: int = 1,
    stderr: str = "",
) -> SSHErrorClassification:
    """Convenience function to classify an SSH error.

    Args:
        error_message: The error message or combined output
        exit_code: SSH exit code
        stderr: Separate stderr output if available

    Returns:
        SSHErrorClassification with error type and recommended actions
    """
    return get_ssh_error_classifier().classify(error_message, exit_code, stderr)


def is_ssh_auth_failure(error_message: str, exit_code: int = 1) -> bool:
    """Convenience function to check if error is auth failure."""
    return get_ssh_error_classifier().is_auth_failure(error_message, exit_code)


def is_ssh_retryable(error_message: str, exit_code: int = 1) -> bool:
    """Convenience function to check if error is retryable."""
    return get_ssh_error_classifier().is_retryable(error_message, exit_code)
