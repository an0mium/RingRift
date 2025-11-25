#!/bin/bash

# Python Test Runner with Hard Timeout
# Runs each test file individually with a hard process timeout
# Usage: ./scripts/run-python-tests-with-timeout.sh [pattern] [timeout_seconds]
# Example: ./scripts/run-python-tests-with-timeout.sh ".*" 30

PATTERN="${1:-.*}"
TIMEOUT_SECONDS="${2:-30}"
PYTHON_PATH="${3:-python}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arrays to track results
declare -a PASSED_TESTS
declare -a FAILED_TESTS
declare -a TIMEOUT_TESTS
declare -a OOM_TESTS

echo "============================================="
echo "Python Test Runner with Hard Timeout"
echo "Pattern: $PATTERN"
echo "Timeout: ${TIMEOUT_SECONDS}s per file"
echo "Directory: ai-service/tests"
echo "============================================="
echo ""

# Change to ai-service directory
cd ai-service || { echo "Error: ai-service directory not found"; exit 1; }

# Find all test files matching pattern
TEST_FILES=$(find tests -name "test_*.py" | grep -E "$PATTERN" | sort)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found matching pattern: $PATTERN"
    exit 1
fi

TOTAL_FILES=$(echo "$TEST_FILES" | wc -l | tr -d ' ')
CURRENT=0

echo "Found $TOTAL_FILES test files"
echo ""

for test_file in $TEST_FILES; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_FILES] Testing: $test_file"
    
    # Capture output to temp file
    TEMP_OUTPUT=$(mktemp)
    
    # Run test with timeout
    start_time=$(date +%s)
    
    # Use timeout command if available (gtimeout on macOS with coreutils)
    if command -v gtimeout &> /dev/null; then
        gtimeout --signal=KILL ${TIMEOUT_SECONDS}s $PYTHON_PATH -m pytest "$test_file" -v --tb=short > "$TEMP_OUTPUT" 2>&1
        EXIT_CODE=$?
    elif command -v timeout &> /dev/null; then
        timeout --signal=KILL ${TIMEOUT_SECONDS}s $PYTHON_PATH -m pytest "$test_file" -v --tb=short > "$TEMP_OUTPUT" 2>&1
        EXIT_CODE=$?
    else
        # Fallback: run in background with manual timeout
        $PYTHON_PATH -m pytest "$test_file" -v --tb=short > "$TEMP_OUTPUT" 2>&1 &
        PID=$!
        
        ELAPSED=0
        while [ $ELAPSED -lt $TIMEOUT_SECONDS ]; do
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID
                EXIT_CODE=$?
                break
            fi
            sleep 1
            ELAPSED=$((ELAPSED + 1))
        done
        
        if [ $ELAPSED -ge $TIMEOUT_SECONDS ]; then
            # Kill the process tree
            pkill -P $PID 2>/dev/null
            kill -9 $PID 2>/dev/null
            wait $PID 2>/dev/null
            EXIT_CODE=124
        fi
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Check for OOM or memory-related crashes in output
    if grep -q "MemoryError\|Killed\|Cannot allocate memory\|Out of memory\|SIGKILL" "$TEMP_OUTPUT"; then
        echo -e "  ${RED}💀 OOM - Out of Memory (${duration}s)${NC}"
        OOM_TESTS+=("$test_file")
    elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
        echo -e "  ${YELLOW}⏰ TIMEOUT - Forcibly killed after ${TIMEOUT_SECONDS}s${NC}"
        TIMEOUT_TESTS+=("$test_file")
    elif [ $EXIT_CODE -eq 0 ]; then
        # Extract pass count
        PASS_COUNT=$(grep -E "passed" "$TEMP_OUTPUT" | tail -1)
        echo -e "  ${GREEN}✓ PASSED (${duration}s)${NC} $PASS_COUNT"
        PASSED_TESTS+=("$test_file")
    else
        echo -e "  ${RED}✗ FAILED (exit code: $EXIT_CODE, ${duration}s)${NC}"
        FAILED_TESTS+=("$test_file")
        
        # Print brief error summary
        echo "    Error summary:"
        grep -E "FAILED|ERROR|assert|Error" "$TEMP_OUTPUT" | head -5 | sed 's/^/      /'
    fi
    
    rm -f "$TEMP_OUTPUT"
    echo ""
done

# Print summary
echo ""
echo "============================================="
echo "SUMMARY"
echo "============================================="
echo ""

echo -e "${GREEN}PASSED: ${#PASSED_TESTS[@]}${NC}"
for t in "${PASSED_TESTS[@]}"; do
    echo "  ✓ $t"
done

echo ""
echo -e "${RED}FAILED: ${#FAILED_TESTS[@]}${NC}"
for t in "${FAILED_TESTS[@]}"; do
    echo "  ✗ $t"
done

echo ""
echo -e "${YELLOW}TIMEOUT: ${#TIMEOUT_TESTS[@]}${NC}"
for t in "${TIMEOUT_TESTS[@]}"; do
    echo "  ⏰ $t"
done

echo ""
echo -e "${RED}OOM (Out of Memory): ${#OOM_TESTS[@]}${NC}"
for t in "${OOM_TESTS[@]}"; do
    echo "  💀 $t"
done

echo ""
echo "============================================="
TOTAL_PASSED=${#PASSED_TESTS[@]}
TOTAL_FAILED=${#FAILED_TESTS[@]}
TOTAL_TIMEOUT=${#TIMEOUT_TESTS[@]}
TOTAL_OOM=${#OOM_TESTS[@]}
TOTAL_PROBLEMATIC=$((TOTAL_FAILED + TOTAL_TIMEOUT + TOTAL_OOM))

echo "Total: $TOTAL_FILES files"
echo "Passed: $TOTAL_PASSED"
echo "Failed: $TOTAL_FAILED"
echo "Timeout: $TOTAL_TIMEOUT"
echo "OOM: $TOTAL_OOM"
echo ""

if [ $TOTAL_PROBLEMATIC -eq 0 ]; then
    echo -e "${GREEN}All tests completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}$TOTAL_PROBLEMATIC problematic test file(s) identified${NC}"
    exit 1
fi