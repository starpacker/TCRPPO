#!/bin/bash
# Delete BF16-generated tFold feature caches
# WARNING: This will permanently delete ~699GB of cache data

set -e

DATA_DIR="/share/liuyutian/tcrppo_v2/data"

echo "=========================================="
echo "BF16 Cache Deletion"
echo "=========================================="
echo ""

# Files to delete
CACHES=(
    "tfold_feature_cache.db"
    "tfold_feature_cache_baseline_amp.db"
    "tfold_feature_cache_trace43_20pep.db"
)

echo "Checking for processes using these caches..."
PIDS_TO_KILL=""
for cache in "${CACHES[@]}"; do
    if [ -f "$DATA_DIR/$cache" ]; then
        pids=$(lsof "$DATA_DIR/$cache" 2>/dev/null | grep -v COMMAND | awk '{print $2}' | sort -u || echo "")
        if [ -n "$pids" ]; then
            echo "  $cache is being used by PIDs: $pids"
            PIDS_TO_KILL="$PIDS_TO_KILL $pids"
        fi
    fi
done

if [ -n "$PIDS_TO_KILL" ]; then
    echo ""
    echo "The following processes must be stopped:"
    for pid in $PIDS_TO_KILL; do
        ps aux | grep "^[^ ]* *$pid " | grep -v grep || echo "  PID $pid (already stopped)"
    done
    echo ""
    read -p "Kill these processes? (yes/no): " confirm_kill
    if [ "$confirm_kill" == "yes" ]; then
        for pid in $PIDS_TO_KILL; do
            echo "  Killing PID $pid..."
            kill $pid 2>/dev/null || echo "    (already dead)"
        done
        echo "Waiting 3 seconds for processes to exit..."
        sleep 3
    else
        echo "Aborted. Please stop these processes manually first."
        exit 1
    fi
fi

echo ""
echo "Files to delete:"
total_size=0
for cache in "${CACHES[@]}"; do
    if [ -f "$DATA_DIR/$cache" ]; then
        size=$(du -sh "$DATA_DIR/$cache" | cut -f1)
        echo "  - $cache ($size)"
        size_bytes=$(du -sb "$DATA_DIR/$cache" | cut -f1)
        total_size=$((total_size + size_bytes))
    else
        echo "  - $cache (NOT FOUND)"
    fi
done

total_size_gb=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc)
echo ""
echo "Total size to delete: ${total_size_gb} GB"
echo ""
echo "WARNING: This operation is PERMANENT and cannot be undone!"
echo ""

read -p "Proceed with deletion? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Deleting caches..."
for cache in "${CACHES[@]}"; do
    if [ -f "$DATA_DIR/$cache" ]; then
        echo "  Deleting $cache and associated files..."
        rm -f "$DATA_DIR/$cache"*
        echo "    Done"
    fi
done

echo ""
echo "=========================================="
echo "DELETION COMPLETE"
echo "=========================================="
echo ""
echo "Deleted ${total_size_gb} GB of BF16-generated cache"
echo ""
echo "Next time training runs with FP32, cache will be regenerated with correct precision."
echo ""
