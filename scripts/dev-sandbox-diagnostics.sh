#!/usr/bin/env bash
set -euo pipefail

# dev-sandbox-diagnostics.sh
#
# Helper script for RingRift development:
# - Frees port 3000 if another Node server is already listening (fixes EADDRINUSE)
# - Then starts the dev stack with sandbox AI stall diagnostics enabled.
#
# Usage (from repo root):
#   bash scripts/dev-sandbox-diagnostics.sh
#
# Or via npm (after wiring package.json):
#   npm run dev:sandbox:diagnostics

PORT="${PORT:-3000}"

echo "[dev-sandbox] Checking for processes listening on TCP port ${PORT}..."

# lsof -ti gives just the PIDs, one per line. If lsof fails (no processes), ignore the error.
PIDS="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"

if [[ -n "${PIDS}" ]]; then
  echo "[dev-sandbox] Found process(es) on port ${PORT}: ${PIDS}"
  echo "[dev-sandbox] Attempting to gracefully kill process(es) on port ${PORT}..."
  kill ${PIDS} || true

  # Give the OS a brief moment to release the port.
  sleep 1

  # Double-check if anything is still listening.
  REMAINING="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"
  if [[ -n "${REMAINING}" ]]; then
    echo "[dev-sandbox] WARNING: process(es) still listening on port ${PORT}: ${REMAINING}"
    echo "[dev-sandbox] You may need to manually inspect and kill them, e.g.:"
    echo "  lsof -nP -iTCP:${PORT} | grep LISTEN"
    echo "  kill -9 <pid>"
  else
    echo "[dev-sandbox] Port ${PORT} successfully freed."
  fi
else
  echo "[dev-sandbox] No existing listeners on port ${PORT}."
fi

# Enable sandbox AI stall diagnostics for the dev run.
export RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1

echo "[dev-sandbox] Starting dev stack with diagnostics enabled..."
echo "[dev-sandbox] Command: RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1 npm run dev"

npm run dev
