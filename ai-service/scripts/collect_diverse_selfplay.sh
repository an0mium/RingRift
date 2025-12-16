#!/bin/bash
# Collect diverse selfplay games from distributed nodes
# Usage: ./scripts/collect_diverse_selfplay.sh [--dry-run]

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COLLECT_DIR="data/games/collected_diverse_${TIMESTAMP}"
DRY_RUN=false

if [ "${1:-}" == "--dry-run" ]; then
  DRY_RUN=true
  echo "[DRY RUN MODE]"
fi

echo "=============================================="
echo "  DIVERSE SELFPLAY COLLECTION"
echo "  Timestamp: ${TIMESTAMP}"
echo "=============================================="
echo ""

mkdir -p "$COLLECT_DIR"

# Define hosts with their paths
declare -A HOSTS=(
  ["lambda-gh200-a"]="ubuntu@192.222.51.29"
  ["lambda-gh200-k"]="ubuntu@192.222.51.150"
  ["lambda-gh200-l"]="ubuntu@192.222.51.233"
  ["lambda-h100"]="ubuntu@209.20.157.81"
  ["lambda-a10"]="ubuntu@150.136.65.197"
)

declare -A VAST_HOSTS=(
  ["vast-2x3060ti"]="root@ssh8.vast.ai:17016"
  ["vast-a40"]="root@ssh8.vast.ai:38742"
  ["vast-rtx4060ti"]="root@ssh1.vast.ai:14400"
  ["vast-rtx4060ti-48"]="root@ssh2.vast.ai:19768"
  ["vast-rtx3060ti-64"]="root@ssh3.vast.ai:19766"
  ["vast-rtx2060s"]="root@ssh2.vast.ai:14370"
  ["vast-3070-b"]="root@ssh7.vast.ai:10012"
)

COLLECTED=0
TOTAL_GAMES=0

echo "Step 1: Collecting from Lambda nodes..."
for name in "${!HOSTS[@]}"; do
  host="${HOSTS[$name]}"
  echo ""
  echo "Checking $name ($host)..."

  # Check for game files
  files=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$host" \
    "find ~/ringrift/ai-service/data/games -name '*.jsonl' -o -name '*.db' 2>/dev/null | head -20" 2>/dev/null || echo "")

  if [ -z "$files" ]; then
    echo "  No game files found"
    continue
  fi

  echo "  Found game files:"
  echo "$files" | head -5 | sed 's/^/    /'

  if [ "$DRY_RUN" == "false" ]; then
    mkdir -p "${COLLECT_DIR}/${name}"

    # Collect JSONL files from diverse selfplay
    rsync -avz --include='*.jsonl' --include='*.db' --exclude='*' \
      "${host}:~/ringrift/ai-service/data/games/" \
      "${COLLECT_DIR}/${name}/" 2>/dev/null || echo "  rsync partial success"

    count=$(find "${COLLECT_DIR}/${name}" -name '*.jsonl' -o -name '*.db' 2>/dev/null | wc -l)
    echo "  Collected $count files to ${COLLECT_DIR}/${name}/"
    ((COLLECTED+=count))
  fi
done

echo ""
echo "Step 2: Collecting from Vast nodes..."
for name in "${!VAST_HOSTS[@]}"; do
  hostinfo="${VAST_HOSTS[$name]}"
  host="${hostinfo%:*}"
  port="${hostinfo#*:}"

  echo ""
  echo "Checking $name ($host port $port)..."

  # Check for game files
  files=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$port" -i ~/.ssh/id_cluster "$host" \
    "find ~/ringrift/ai-service/data/games -name '*.jsonl' -o -name '*.db' 2>/dev/null | head -20" 2>/dev/null || echo "")

  if [ -z "$files" ]; then
    echo "  No game files found"
    continue
  fi

  echo "  Found game files:"
  echo "$files" | head -5 | sed 's/^/    /'

  if [ "$DRY_RUN" == "false" ]; then
    mkdir -p "${COLLECT_DIR}/${name}"

    rsync -avz -e "ssh -p $port -i ~/.ssh/id_cluster" \
      --include='*.jsonl' --include='*.db' --exclude='*' \
      "${host}:~/ringrift/ai-service/data/games/" \
      "${COLLECT_DIR}/${name}/" 2>/dev/null || echo "  rsync partial success"

    count=$(find "${COLLECT_DIR}/${name}" -name '*.jsonl' -o -name '*.db' 2>/dev/null | wc -l)
    echo "  Collected $count files to ${COLLECT_DIR}/${name}/"
    ((COLLECTED+=count))
  fi
done

echo ""
echo "=============================================="
echo "  COLLECTION SUMMARY"
echo "=============================================="
echo "Files collected: $COLLECTED"
echo "Collection directory: $COLLECT_DIR"

if [ "$DRY_RUN" == "false" ] && [ $COLLECTED -gt 0 ]; then
  echo ""
  echo "Step 3: Merging databases..."

  # Find all .db files and merge them
  dbs=$(find "$COLLECT_DIR" -name '*.db' 2>/dev/null | tr '\n' ' ')

  if [ -n "$dbs" ]; then
    echo "Found databases to merge: $dbs"

    # Activate venv
    if [ -f "venv/bin/activate" ]; then
      source venv/bin/activate
    fi

    # Merge into consolidated db
    MERGED_DB="data/games/merged_diverse_${TIMESTAMP}.db"
    python scripts/merge_game_dbs.py \
      --output "$MERGED_DB" \
      $dbs \
      --dedupe-by-game-id 2>&1 || echo "Merge completed with warnings"

    echo "Merged database: $MERGED_DB"
  fi
fi

echo ""
echo "Done!"
