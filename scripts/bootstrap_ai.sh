#!/usr/bin/env bash
#
# bootstrap_ai.sh
# ===============
#
# One-shot helper to bootstrap the Python AI service for local development.
# It delegates to ai-service/setup.sh to create a virtualenv and install
# dependencies, then prints a short checklist for running health/parity
# checks.
#
# Usage (from repo root):
#
#   ./scripts/bootstrap_ai.sh
#
# This script is intentionally conservative: it only prepares the environment
# and prints suggested follow-up commands; it does not run long pytest or
# soak jobs automatically.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_SERVICE_DIR="${ROOT_DIR}/ai-service"

if [[ ! -d "${AI_SERVICE_DIR}" ]]; then
  echo "ERROR: ai-service directory not found at ${AI_SERVICE_DIR}" >&2
  exit 1
fi

echo "=== RingRift AI Service bootstrap ==="
echo "Root:       ${ROOT_DIR}"
echo "AI service: ${AI_SERVICE_DIR}"
echo

cd "${AI_SERVICE_DIR}"

if [[ ! -x "./setup.sh" ]]; then
  echo "ERROR: ai-service/setup.sh is missing or not executable." >&2
  echo "       Please check the repository and retry." >&2
  exit 1
fi

echo "Running ai-service/setup.sh to create venv and install dependencies..."
./setup.sh

echo
echo "AI service environment prepared."
echo
echo "Next suggested checks (run from ai-service/):"
echo "  1) Verify basic service health:"
echo "       source venv/bin/activate"
echo "       python -m uvicorn app.main:app --reload --port 8001"
echo "     Then in another terminal:"
echo "       curl http://localhost:8001/health"
echo
echo "  2) Run core parity/contract tests:"
echo "       PYTHONPATH=. venv/bin/pytest tests/contracts/test_contract_vectors.py"
echo
echo "  3) Optionally run the self-play parity checker:"
echo "       PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --compact"
echo
echo "These steps help ensure the AI host, models, and TS↔Python fixtures are all in sync."

