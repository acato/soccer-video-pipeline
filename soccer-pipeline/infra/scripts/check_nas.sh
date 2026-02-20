#!/bin/bash
NAS_PATH="${NAS_MOUNT_PATH:-/mnt/nas/soccer}"
WARN_FREE_GB=500
echo "=== NAS Health Check: $NAS_PATH ==="
if ! ls "$NAS_PATH" > /dev/null 2>&1; then echo "❌ FAIL: Cannot access $NAS_PATH"; exit 1; fi
echo "✅ Mount accessible"
FREE_GB=$(df -BG "$NAS_PATH" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
[ "$FREE_GB" -lt "$WARN_FREE_GB" ] && echo "⚠️  WARN: Only ${FREE_GB}GB free" || echo "✅ Free space: ${FREE_GB}GB"
OUTPUT_PATH="${NAS_OUTPUT_PATH:-$NAS_PATH/output}"
mkdir -p "$OUTPUT_PATH" && echo "✅ Output dir writable" || echo "❌ FAIL: cannot write to $OUTPUT_PATH"
echo "=== Done ==="
