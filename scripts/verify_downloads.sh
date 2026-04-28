#!/bin/bash
# Verify downloaded DeepAIR and ProtBert files

set -e

echo "=== Verifying Downloads ==="
echo ""

# Verify DeepAIR.zip
echo "--- DeepAIR.zip ---"
DEEPAIR_ZIP="/share/liuyutian/tcrppo_v2/models/deepair/DeepAIR.zip"

if [ ! -f "$DEEPAIR_ZIP" ]; then
    echo "❌ DeepAIR.zip not found at $DEEPAIR_ZIP"
    exit 1
fi

size=$(du -h "$DEEPAIR_ZIP" | cut -f1)
echo "File size: $size"

# Test zip integrity
echo "Testing zip integrity..."
if unzip -t "$DEEPAIR_ZIP" > /tmp/deepair_test.log 2>&1; then
    echo "✓ ZIP file is valid"
    echo "Contents preview:"
    unzip -l "$DEEPAIR_ZIP" | head -20
else
    echo "❌ ZIP file is corrupted"
    tail -20 /tmp/deepair_test.log
    exit 1
fi
echo ""

# Verify ProtBert files
echo "--- ProtBert ---"
PROTBERT_DIR="/share/liuyutian/tcrppo_v2/models/protbert"

required_files=("pytorch_model.bin" "config.json" "vocab.txt" "tokenizer_config.json")
for file in "${required_files[@]}"; do
    if [ ! -f "$PROTBERT_DIR/$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
    size=$(du -h "$PROTBERT_DIR/$file" | cut -f1)
    echo "✓ $file ($size)"
done

# Check pytorch_model.bin size (should be ~1.5GB)
model_size_bytes=$(stat -c%s "$PROTBERT_DIR/pytorch_model.bin")
model_size_gb=$(echo "scale=2; $model_size_bytes / 1024 / 1024 / 1024" | bc)

echo ""
echo "pytorch_model.bin size: ${model_size_gb}GB"

if (( $(echo "$model_size_gb < 1.0" | bc -l) )); then
    echo "⚠️  WARNING: pytorch_model.bin seems too small (expected ~1.5GB)"
    echo "   This might be an incomplete download or wrong file"
    echo "   Checking file type..."
    file "$PROTBERT_DIR/pytorch_model.bin" | head -5
    echo ""
    echo "   First 100 bytes:"
    head -c 100 "$PROTBERT_DIR/pytorch_model.bin" | od -c | head -10
else
    echo "✓ Size looks reasonable"
fi

echo ""
echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Extract DeepAIR: cd $PROTBERT_DIR/../deepair && unzip DeepAIR.zip"
echo "2. Examine structure: ls -la DeepAIR/"
echo "3. Follow rebuild plan: cat docs/DEEPAIR_REBUILD_PLAN.md"
