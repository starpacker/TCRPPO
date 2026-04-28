#!/bin/bash
# Monitor DeepAIR and ProtBert download progress

echo "=== Download Status Monitor ==="
echo "Time: $(date)"
echo ""

# Check DeepAIR download
echo "--- DeepAIR (Zenodo) ---"
if ps -p $(cat /tmp/deepair_download.pid 2>/dev/null) > /dev/null 2>&1; then
    echo "Status: Running (PID: $(cat /tmp/deepair_download.pid))"
    deepair_size=$(du -h /share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip 2>/dev/null | cut -f1)
    echo "Current size: $deepair_size"
    echo "Last 3 lines of log:"
    tail -3 /share/liuyutian/tcrppo_v2/models/deepair/download.log 2>/dev/null | grep -E "[0-9]+%" | tail -1
else
    if [ -f /share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip ]; then
        echo "Status: Complete"
        ls -lh /share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip
    else
        echo "Status: Not found"
    fi
fi
echo ""

# Check ProtBert download
echo "--- ProtBert (HuggingFace Mirror) ---"
if ps aux | grep -E "wget.*pytorch_model.bin" | grep -v grep > /dev/null; then
    echo "Status: Running"
    protbert_size=$(du -h /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin 2>/dev/null | cut -f1)
    echo "Current size: $protbert_size"
    echo "Last 3 lines of log:"
    tail -3 /share/liuyutian/tcrppo_v2/models/protbert/download_model.log 2>/dev/null | grep -E "[0-9]+%" | tail -1
else
    if [ -f /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin ]; then
        echo "Status: Complete"
        ls -lh /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin
    else
        echo "Status: Not found"
    fi
fi
echo ""

# Check if both complete
if [ -f /share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip ] && \
   [ -f /share/liuyutian/tcrppo_v2/models/protbert/pytorch_model.bin ] && \
   ! ps aux | grep -E "wget.*(DeepAIR|pytorch_model)" | grep -v grep > /dev/null; then
    echo "✓ All downloads complete!"
    echo ""
    echo "Next steps:"
    echo "1. Verify downloads: bash scripts/verify_downloads.sh"
    echo "2. Extract DeepAIR: cd models/deepair && unzip DeepAIR.zip"
    echo "3. Rebuild scorer: See docs/DEEPAIR_REBUILD_PLAN.md"
fi
