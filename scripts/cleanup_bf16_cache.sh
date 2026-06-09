#!/bin/bash
# Backup and clear BF16-generated tFold feature caches
# DO NOT delete - move to backup directory for safety

set -e

BACKUP_DIR="/share/liuyutian/tcrppo_v2/data/cache_backup_bf16_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="/share/liuyutian/tcrppo_v2/data"

echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# List of caches confirmed to be BF16-generated
BF16_CACHES=(
    "tfold_feature_cache_baseline_amp.db"
    "tfold_feature_cache_trace43_20pep.db"
)

echo ""
echo "=========================================="
echo "BF16 Cache Cleanup - BACKUP MODE"
echo "=========================================="
echo ""
echo "The following caches will be MOVED (not deleted) to backup:"

total_size=0
for cache in "${BF16_CACHES[@]}"; do
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
echo "Total size to backup: ${total_size_gb} GB"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""

read -p "Proceed with backup? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Moving caches to backup..."
for cache in "${BF16_CACHES[@]}"; do
    if [ -f "$DATA_DIR/$cache" ]; then
        echo "  Moving $cache..."
        mv "$DATA_DIR/$cache"* "$BACKUP_DIR/" 2>/dev/null || echo "    (no associated files)"
    fi
done

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
echo ""
echo "Backed up caches are in: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Update trace93 launch script to remove --use-amp-wrapper flag"
echo "  2. Restart training - it will regenerate cache with FP32"
echo "  3. After confirming FP32 works, you can delete backup directory"
echo ""
