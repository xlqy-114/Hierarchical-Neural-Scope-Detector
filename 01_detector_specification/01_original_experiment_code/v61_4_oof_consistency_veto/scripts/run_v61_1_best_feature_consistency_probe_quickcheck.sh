#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/DriveVLA
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORMAL_DIR="${V61_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v59_3_targeted_feature_formal_2seeds_quickcheck}"
SEEDS="${V61_SEEDS:-955912,955913}"
OUT_DIR="${V61_OUT_DIR:-$FORMAL_DIR/v61_1_best_feature_consistency_probe_quickcheck}"
PACK_PATH="${V61_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v61_1_best_feature_consistency_probe_2seeds_quickcheck_for_return.tar.gz}"

mkdir -p "$OUT_DIR" /root/autodl-tmp/DriveVLA/pack_output_analysis

echo "[v61.1] FORMAL_DIR=$FORMAL_DIR"
echo "[v61.1] SEEDS=$SEEDS"
echo "[v61.1] OUT_DIR=$OUT_DIR"
echo "[v61.1] PACK_PATH=$PACK_PATH"

python -u "$SCRIPT_DIR/../tools_scan/run_v61_1_best_feature_consistency_probe.py" \
  --formal-dir "$FORMAL_DIR" \
  --seeds "$SEEDS" \
  --output-dir "$OUT_DIR"

rm -f "$PACK_PATH"
tar -czf "$PACK_PATH" -C "$FORMAL_DIR" "$(basename "$OUT_DIR")"
ls -lh "$PACK_PATH"
echo "[v61.1] return pack: $PACK_PATH"
