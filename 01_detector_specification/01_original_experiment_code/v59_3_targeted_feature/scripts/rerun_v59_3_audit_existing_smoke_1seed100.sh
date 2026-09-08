#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla
mkdir -p "$REPO_ROOT/tools_scan" "$REPO_ROOT/pack_output_analysis"
cp -f "$PKG_DIR/tools_scan/"*.py "$REPO_ROOT/tools_scan/"
chmod +x "$REPO_ROOT/tools_scan/"*.py || true
FORMAL_DIR=${V59_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v59_2_targeted_feature_smoke_1seed100}
SEEDS_CSV=${V59_SEEDS:-955901}
EXPECTED_ROWS=${V59_EXPECTED_ROWS:-100}
SEED_ROOT=${V59_SEED_ROOT:-A1_internal_neural_scdout_v54_2_online_singlefault}
OUT=${V59_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v59_3_targeted_feature_smoke_1seed100_for_return.tar.gz}
mkdir -p "$FORMAL_DIR/logs"
echo "[v59.3 rerun] FORMAL_DIR=$FORMAL_DIR"
echo "[v59.3 rerun] This reruns v59 audit/pack only; it does not rerun inference."
python -u "$REPO_ROOT/tools_scan/audit_v59_targeted_features.py" \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --output-dir "$FORMAL_DIR/v59_targeted_feature_audit" \
  --abs-cap "${V59_SANITIZE_ABS_CAP:-1000000}" \
  --high-risk-p99-cap "${V59_HIGH_RISK_P99_CAP:-1000000}" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/audit_v59_targeted_features.log"
rm -f "$OUT"
mkdir -p "$(dirname "$OUT")"
cd "$FORMAL_DIR"
tar -czf "$OUT" --ignore-failed-read \
  v54_scdout_internal_effectiveness_summary.json \
  v54_online_smoke_requirements_check.json \
  v54_online_exports/v54_online_export_summary.json \
  v54_online_exports/v54_2_multipolicy_label_summary.json \
  v54_online_exports/v54_2_posthoc_metrics_multi_policy.csv \
  v54_online_exports/v54_feature_inventory_summary.json \
  v54_online_exports/v54_feature_inventory_existing_vs_new.csv \
  v54_online_exports/v54_target_pool_manifest_summary.json \
  v54_online_exports/v54_sampled_fault_target_summary.json \
  v54_inline_decision_parity_audit/v54_inline_decision_parity_summary.json \
  v54_inline_decision_parity_audit/v54_inline_decision_parity_summary.csv \
  v59_targeted_feature_audit/*.json \
  v59_targeted_feature_audit/*.csv \
  logs/audit_v59_targeted_features.log \
  A1_internal_neural_scdout_v54_2_online_singlefault/seed*/results/stage_hook_status_v26.json \
  A1_internal_neural_scdout_v54_2_online_singlefault/seed*/results/v53_inline_detector_summary.json
ls -lh "$OUT"
tar -tzf "$OUT" | sed -n '1,160p'
