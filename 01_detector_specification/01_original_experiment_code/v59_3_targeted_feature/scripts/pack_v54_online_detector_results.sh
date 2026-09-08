#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <campaign_dir> [output_tar_gz]" >&2
  exit 2
fi
CAMPAIGN_DIR="$1"
OUT_PATH="${2:-}"
if [[ -z "$OUT_PATH" ]]; then
  OUT_PATH="/root/autodl-tmp/DriveVLA/pack_output_analysis/$(basename "$CAMPAIGN_DIR")_for_return.tar.gz"
fi
mkdir -p "$(dirname "$OUT_PATH")"
cd "$(dirname "$CAMPAIGN_DIR")"
BASE="$(basename "$CAMPAIGN_DIR")"
# Compact but complete audit pack. Includes inline records, feature/label matrices,
# v31-compatible multi-policy labels, target manifests, plan_conv, and fault plans.
# Use broad seed-result inclusion to avoid brittle path assumptions across v54.x.
tar -czf "$OUT_PATH" \
  "$BASE/logs" \
  "$BASE/resources" \
  "$BASE/v54_scdout_internal_effectiveness_summary.json" \
  "$BASE/v54_online_exports" \
  "$BASE/v54_inline_decision_parity_audit" \
  "$BASE/v54_online_smoke_requirements_check.json" \
  "$BASE"/A1_internal_neural_scdout_*_online_singlefault/seed*/plan_conv.json \
  "$BASE"/A1_internal_neural_scdout_*_online_singlefault/seed*/results \
  2>/tmp/v54_pack_tar_warnings.log || {
    echo "[WARN] tar failed; see /tmp/v54_pack_tar_warnings.log" >&2
    cat /tmp/v54_pack_tar_warnings.log >&2 || true
    exit 1
  }
echo "[v54 pack] wrote $OUT_PATH"
