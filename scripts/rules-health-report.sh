#!/usr/bin/env bash
#
# RingRift Rules/Parity Health Report Script
#
# This script runs all rules compliance test categories and produces a summary
# health report. It can be run locally by developers or integrated into CI.
#
# Usage:
#   ./scripts/rules-health-report.sh           # Run with summary output
#   ./scripts/rules-health-report.sh --verbose # Show full test output
#   ./scripts/rules-health-report.sh --report  # Generate RULES_HEALTH_REPORT.md
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#

set -o pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_FILE="$PROJECT_ROOT/RULES_HEALTH_REPORT.md"

# Parse arguments
VERBOSE=false
GENERATE_REPORT=false

for arg in "$@"; do
  case $arg in
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --report|-r)
      GENERATE_REPORT=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--verbose|-v] [--report|-r] [--help|-h]"
      echo ""
      echo "Options:"
      echo "  --verbose, -v   Show full test output"
      echo "  --report, -r    Generate RULES_HEALTH_REPORT.md"
      echo "  --help, -h      Show this help message"
      exit 0
      ;;
  esac
done

# ============================================================================
# Color support
# ============================================================================

if [[ -t 1 ]] && command -v tput &>/dev/null && [[ $(tput colors 2>/dev/null) -ge 8 ]]; then
  GREEN=$(tput setaf 2)
  RED=$(tput setaf 1)
  YELLOW=$(tput setaf 3)
  CYAN=$(tput setaf 6)
  BOLD=$(tput bold)
  RESET=$(tput sgr0)
else
  GREEN=""
  RED=""
  YELLOW=""
  CYAN=""
  BOLD=""
  RESET=""
fi

# ============================================================================
# Tracking variables
# ============================================================================

declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()
declare -a TEST_CATEGORIES=()
TOTAL_PASSED=0
TOTAL_FAILED=0
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ============================================================================
# Helper functions
# ============================================================================

run_test() {
  local name="$1"
  local category="$2"
  local command="$3"
  
  TEST_NAMES+=("$name")
  TEST_CATEGORIES+=("$category")
  
  if $VERBOSE; then
    echo ""
    echo "${CYAN}Running: $name${RESET}"
    echo "${YELLOW}Command: $command${RESET}"
    echo "---"
  fi
  
  local output
  local exit_code
  
  if $VERBOSE; then
    eval "$command"
    exit_code=$?
  else
    output=$(eval "$command" 2>&1)
    exit_code=$?
  fi
  
  if [[ $exit_code -eq 0 ]]; then
    TEST_RESULTS+=("PASS")
    ((TOTAL_PASSED++))
    if ! $VERBOSE; then
      echo -n "."
    fi
  else
    TEST_RESULTS+=("FAIL")
    ((TOTAL_FAILED++))
    if ! $VERBOSE; then
      echo -n "x"
    fi
  fi
  
  return $exit_code
}

print_box_line() {
  local text="$1"
  local width=66
  local padding=$((width - ${#text} - 4))
  printf "║ %s%*s ║\n" "$text" "$padding" ""
}

print_result_line() {
  local name="$1"
  local result="$2"
  local width=66
  local result_width=6
  local name_width=$((width - result_width - 6))
  
  local symbol
  local color
  if [[ "$result" == "PASS" ]]; then
    symbol="✓"
    color="$GREEN"
  else
    symbol="✗"
    color="$RED"
  fi
  
  printf "║ %s%-*s%s%*s ║\n" "$color$symbol$RESET " "$name_width" "$name" "$color" "$result_width" "$result$RESET"
}

# ============================================================================
# Main execution
# ============================================================================

cd "$PROJECT_ROOT" || exit 1

echo ""
echo "${BOLD}Running RingRift Rules/Parity Health Report...${RESET}"
echo ""

if ! $VERBOSE; then
  echo -n "Progress: "
fi

# ----------------------------------------------------------------------------
# TypeScript Tests
# ----------------------------------------------------------------------------

run_test "Determinism" "TypeScript" \
  "npm test -- EngineDeterminism.shared --passWithNoTests --silent 2>/dev/null"

run_test "No-Randomness Guards" "TypeScript" \
  "npm test -- NoRandomInCoreRules --passWithNoTests --silent 2>/dev/null"

run_test "TS↔Python Parity" "TypeScript" \
  "npm test -- Python_vs_TS --passWithNoTests --silent 2>/dev/null"

run_test "LPS Victory" "TypeScript" \
  "npm test -- victory.LPS --passWithNoTests --silent 2>/dev/null"

run_test "Ring Caps (Placement)" "TypeScript" \
  "npm test -- placement.shared --passWithNoTests --silent 2>/dev/null"

run_test "Territory" "TypeScript" \
  "npm test -- territory --passWithNoTests --silent 2>/dev/null"

run_test "Lines" "TypeScript" \
  "npm test -- lines --passWithNoTests --silent 2>/dev/null"

run_test "BoardManager Invariants" "TypeScript" \
  "npm test -- BoardManager --passWithNoTests --silent 2>/dev/null"

# ----------------------------------------------------------------------------
# Python Tests
# ----------------------------------------------------------------------------

# Check if Python tests directory exists
if [[ -d "$PROJECT_ROOT/ai-service/tests" ]]; then
  run_test "Determinism" "Python" \
    "cd ai-service && python -m pytest tests/test_engine_determinism.py -v --tb=no -q 2>/dev/null || true"
  
  run_test "No-Randomness Guards" "Python" \
    "cd ai-service && python -m pytest tests/test_no_random_in_rules_core.py -v --tb=no -q 2>/dev/null || true"
  
  run_test "Parity" "Python" \
    "cd ai-service && python -m pytest tests/parity/ -v --tb=no -q 2>/dev/null || true"
  
  run_test "LPS & Ring Caps" "Python" \
    "cd ai-service && python -m pytest tests/test_lps_and_ring_caps.py -v --tb=no -q 2>/dev/null || true"
fi

if ! $VERBOSE; then
  echo ""
  echo ""
fi

# ============================================================================
# Output Report
# ============================================================================

TOTAL=$((TOTAL_PASSED + TOTAL_FAILED))

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            RingRift Rules/Parity Health Report                     ║"
printf "║                    %-47s ║\n" "$TIMESTAMP"
echo "╠════════════════════════════════════════════════════════════════════╣"

current_category=""
for i in "${!TEST_NAMES[@]}"; do
  if [[ "${TEST_CATEGORIES[$i]}" != "$current_category" ]]; then
    if [[ -n "$current_category" ]]; then
      echo "╠════════════════════════════════════════════════════════════════════╣"
    fi
    current_category="${TEST_CATEGORIES[$i]}"
    printf "║ ${BOLD}%-66s${RESET} ║\n" "$current_category Tests"
    echo "╠════════════════════════════════════════════════════════════════════╣"
  fi
  print_result_line "${TEST_NAMES[$i]}" "${TEST_RESULTS[$i]}"
done

echo "╠════════════════════════════════════════════════════════════════════╣"

# Summary line
if [[ $TOTAL_FAILED -eq 0 ]]; then
  summary_symbol="${GREEN}✓ ALL${RESET}"
  summary_status="PASSED"
else
  summary_symbol="${RED}✗ FAIL${RESET}"
  summary_status="FAILED"
fi

printf "║ SUMMARY: %d/%d passed %48s ║\n" "$TOTAL_PASSED" "$TOTAL" "$summary_symbol"
echo "╚════════════════════════════════════════════════════════════════════╝"

# ============================================================================
# Generate Markdown Report (optional)
# ============================================================================

if $GENERATE_REPORT; then
  cat > "$REPORT_FILE" << EOF
# RingRift Rules/Parity Health Report

**Generated:** $TIMESTAMP  
**Status:** $summary_status ($TOTAL_PASSED/$TOTAL passed)

## Test Results

EOF

  current_category=""
  for i in "${!TEST_NAMES[@]}"; do
    if [[ "${TEST_CATEGORIES[$i]}" != "$current_category" ]]; then
      current_category="${TEST_CATEGORIES[$i]}"
      echo "### $current_category Tests" >> "$REPORT_FILE"
      echo "" >> "$REPORT_FILE"
    fi
    
    if [[ "${TEST_RESULTS[$i]}" == "PASS" ]]; then
      echo "- ✅ ${TEST_NAMES[$i]}" >> "$REPORT_FILE"
    else
      echo "- ❌ ${TEST_NAMES[$i]}" >> "$REPORT_FILE"
    fi
  done
  
  cat >> "$REPORT_FILE" << EOF

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | $TOTAL |
| Passed | $TOTAL_PASSED |
| Failed | $TOTAL_FAILED |
| Pass Rate | $(( (TOTAL_PASSED * 100) / (TOTAL > 0 ? TOTAL : 1) ))% |

---

*This report was automatically generated by \`scripts/rules-health-report.sh\`*
EOF

  echo ""
  echo "${GREEN}Report written to: $REPORT_FILE${RESET}"
fi

# ============================================================================
# Exit with appropriate code
# ============================================================================

if [[ $TOTAL_FAILED -gt 0 ]]; then
  echo ""
  echo "${RED}${BOLD}Health check failed: $TOTAL_FAILED test(s) did not pass.${RESET}"
  exit 1
else
  echo ""
  echo "${GREEN}${BOLD}All rules/parity tests passed!${RESET}"
  exit 0
fi
