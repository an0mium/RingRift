#!/bin/bash
# Cleanup script for unusable selfplay data
#
# This script removes selfplay data files that cannot be used for training:
# - Empty JSONL files (0 bytes)
# - SQLite DBs missing the 'moves' column
# - Gzip-compressed files with wrong .jsonl extension
# - Empty auto_* worker directories
#
# Usage:
#   ./scripts/cleanup_selfplay_data.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be deleted without actually deleting
#
# Run from ai-service directory:
#   cd ai-service && ./scripts/cleanup_selfplay_data.sh

set -e

# Parse arguments
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be deleted ==="
fi

# Ensure we're in ai-service directory
if [ ! -d "data/selfplay" ]; then
    echo "Error: Must run from ai-service directory"
    exit 1
fi

echo "=== Cleaning unusable selfplay data ==="

# Track stats
deleted_count=0
deleted_size=0

# Helper function
delete_file() {
    local file="$1"
    local reason="$2"
    local size=$(stat -c%s "$file" 2>/dev/null || echo 0)

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] would rm: $file ($reason)"
    else
        echo "  rm: $file ($reason)"
        rm -f "$file"
    fi
    deleted_count=$((deleted_count + 1))
    deleted_size=$((deleted_size + size))
}

delete_dir() {
    local dir="$1"
    local reason="$2"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] would rm -rf: $dir ($reason)"
    else
        echo "  rm -rf: $dir ($reason)"
        rm -rf "$dir"
    fi
    deleted_count=$((deleted_count + 1))
}

# 1. Remove empty JSONL files
echo ""
echo "Step 1: Removing empty JSONL files..."
while IFS= read -r f; do
    delete_file "$f" "empty"
done < <(find data/selfplay -name "*.jsonl" -size 0 -type f 2>/dev/null)

# 2. Remove DBs without moves data (neither 'moves' column nor 'game_moves' table)
# GameReplayDB format stores moves in a separate 'game_moves' table, not inline
echo ""
echo "Step 2: Checking p2p_hybrid DBs for moves data..."
while IFS= read -r db; do
    dir=$(dirname "$db")
    jsonl="$dir/games.jsonl"

    # Check if DB has moves column OR game_moves table (GameReplayDB format)
    has_moves_column=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$db')
cur = conn.cursor()
cur.execute('PRAGMA table_info(games)')
cols = [r[1] for r in cur.fetchall()]
print(1 if 'moves' in cols else 0)
conn.close()
" 2>/dev/null || echo 0)

    has_game_moves_table=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$db')
cur = conn.cursor()
cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'\")
print(1 if cur.fetchone() else 0)
conn.close()
" 2>/dev/null || echo 0)

    if [ "$has_moves_column" = "0" ] && [ "$has_game_moves_table" = "0" ]; then
        if [ -f "$jsonl" ] && [ -s "$jsonl" ]; then
            # JSONL exists and is not empty - safe to delete DB only
            delete_file "$db" "no moves data, jsonl backup exists"
        else
            # No valid JSONL - delete both
            delete_file "$db" "no moves data, no valid jsonl"
            [ -f "$jsonl" ] && delete_file "$jsonl" "companion to unusable db"
        fi
    fi
done < <(find data/selfplay/p2p_hybrid -name "games.db" 2>/dev/null)

# 3. Remove standalone DBs without moves data
echo ""
echo "Step 3: Checking standalone DBs for moves data..."
while IFS= read -r db; do
    has_moves_column=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$db')
cur = conn.cursor()
cur.execute('PRAGMA table_info(games)')
cols = [r[1] for r in cur.fetchall()]
print(1 if 'moves' in cols else 0)
conn.close()
" 2>/dev/null || echo 0)

    has_game_moves_table=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$db')
cur = conn.cursor()
cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'\")
print(1 if cur.fetchone() else 0)
conn.close()
" 2>/dev/null || echo 0)

    if [ "$has_moves_column" = "0" ] && [ "$has_game_moves_table" = "0" ]; then
        delete_file "$db" "no moves data"
    fi
done < <(find data/selfplay -maxdepth 2 -name "*.db" 2>/dev/null | grep -v p2p_hybrid)

# 4. Fix gzip-compressed files with wrong extension
echo ""
echo "Step 4: Fixing gzip-compressed files with .jsonl extension..."
for f in data/selfplay/*.jsonl; do
    if [ -f "$f" ]; then
        # Check if it's gzip compressed (magic bytes 1f 8b)
        if file "$f" 2>/dev/null | grep -q "gzip"; then
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY-RUN] would mv: $f -> ${f}.gz"
            else
                echo "  mv: $f -> ${f}.gz"
                mv "$f" "${f}.gz"
            fi
        fi
    fi
done

# 5. Remove directories that only contain empty files
echo ""
echo "Step 5: Removing auto_* directories with only empty files..."
while IFS= read -r dir; do
    if [ -d "$dir" ]; then
        non_empty=$(find "$dir" -type f -size +0 2>/dev/null | wc -l)
        if [ "$non_empty" = "0" ]; then
            delete_dir "$dir" "only empty files"
        fi
    fi
done < <(find data/selfplay -type d -name "auto_*" 2>/dev/null)

# 6. Remove empty directories
echo ""
echo "Step 6: Removing empty directories..."
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY-RUN] would remove empty directories"
else
    find data/selfplay -type d -empty -delete 2>/dev/null || true
fi

# Summary
echo ""
echo "=== Cleanup complete ==="
echo "Files/directories processed: $deleted_count"
if command -v bc &> /dev/null; then
    echo "Space freed: approximately $(echo "scale=2; $deleted_size / 1024 / 1024" | bc) MB"
else
    echo "Space freed: approximately $((deleted_size / 1024 / 1024)) MB"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a dry run. Run without --dry-run to actually delete files."
fi
