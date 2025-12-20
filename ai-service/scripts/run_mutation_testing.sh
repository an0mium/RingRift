#!/bin/bash
# Mutation testing wrapper for mutmut 3.x
# Usage: ./scripts/run_mutation_testing.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Clean up any previous mutation testing artifacts
rm -rf .mutmut-cache mutants

# Verify the test passes first
echo "Running baseline test..."
python -m pytest tests/unit/ai/test_neural_losses.py -x -q --timeout=30
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline test failed. Fix tests before running mutation testing."
    exit 1
fi

echo ""
echo "Baseline test passed. Running mutation testing..."
echo "Note: mutmut 3.x has issues with module imports in copied directories."
echo "Using pytest directly with mutation testing would be more reliable."
echo ""

# Try running mutmut with PYTHONPATH set
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mutmut run || {
    echo ""
    echo "mutmut failed. This is a known issue with mutmut 3.x and complex project structures."
    echo "Consider using mutation testing alternatives like:"
    echo "  - mutmut 2.x (older but more stable)"
    echo "  - cosmic-ray"
    echo "  - mutatest"
    exit 1
}

echo ""
echo "Mutation testing complete. View results with: mutmut results"
