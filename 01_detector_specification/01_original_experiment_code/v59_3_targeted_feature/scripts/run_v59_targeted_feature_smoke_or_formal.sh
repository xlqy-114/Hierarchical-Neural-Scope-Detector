#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla
mkdir -p "$REPO_ROOT/drivevla" "$REPO_ROOT/tools_scan" "$REPO_ROOT/pack_output_analysis"
cp -f "$PKG_DIR/drivevla/inference_drivevla_subset1200_v59_online_targeted_feature_audit.py" "$REPO_ROOT/drivevla/inference_drivevla_subset1200_v54_online_v35_1_scdout_record_audit.py"
cp -f "$PKG_DIR/tools_scan/"*.py "$REPO_ROOT/tools_scan/"
chmod +x "$REPO_ROOT/tools_scan/"*.py || true
export V54_2_SEEDS=${V59_SEEDS:-${V54_2_SEEDS:-955901}}
export V54_2_SUBSET_IDS=${V59_SUBSET_IDS:-${V54_2_SUBSET_IDS:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json}}
export V54_2_POSITIVE_MIN=${V59_POSITIVE_MIN:-${V54_2_POSITIVE_MIN:-70}}
export V54_2_POSITIVE_MAX=${V59_POSITIVE_MAX:-${V54_2_POSITIVE_MAX:-80}}
export V54_2_EXPECTED_ROWS=${V59_EXPECTED_ROWS:-${V54_2_EXPECTED_ROWS:-100}}
export V54_2_FORMAL_DIR=${V59_FORMAL_DIR:-${V54_2_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v59_3_targeted_feature_smoke_1seed100}}
export V54_PACK_PATH=${V59_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v59_3_targeted_feature_smoke_1seed100_for_return.tar.gz}
export V54_STAGE_HOOK_SPECS=${V59_STAGE_HOOK_SPECS:-$(cat "$PKG_DIR/resources/v59_targeted_stage_hook_specs.txt")}
export V54_STAGE_SUMMARY_MAX_SUMMARY_CALLS=${V59_STAGE_SUMMARY_MAX_SUMMARY_CALLS:-1}
export MASTER_PORT=${MASTER_PORT:-29659}
export V59_SANITIZE_ABS_CAP=${V59_SANITIZE_ABS_CAP:-1000000}
export V59_HIGH_RISK_P99_CAP=${V59_HIGH_RISK_P99_CAP:-1000000}
bash "$PKG_DIR/scripts/run_v54_2_online_detector_smoke_or_formal_no_pack.sh"
FORMAL_DIR="$V54_2_FORMAL_DIR"
SEEDS_CSV="$V54_2_SEEDS"
EXPECTED_ROWS="$V54_2_EXPECTED_ROWS"
SEED_ROOT=A1_internal_neural_scdout_v54_2_online_singlefault
python -u "$REPO_ROOT/tools_scan/audit_v59_targeted_features.py" \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --output-dir "$FORMAL_DIR/v59_targeted_feature_audit" \
  --abs-cap "$V59_SANITIZE_ABS_CAP" \
  --high-risk-p99-cap "$V59_HIGH_RISK_P99_CAP" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/audit_v59_targeted_features.log"
OUT=${V59_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v59_3_targeted_feature_smoke_1seed100_for_return.tar.gz}
rm -f "$OUT"
cd "$FORMAL_DIR"
tar -czf "$OUT" --ignore-failed-read \
  v54_scdout_internal_effectiveness_summary.json \
  v54_online_smoke_requirements_check.json \
  v54_online_exports/v54_online_export_summary.json \
  v54_online_exports/v54_2_multipolicy_label_summary.json \
  v54_online_exports/v54_2_posthoc_metrics_multi_policy.csv \
  v54_inline_decision_parity_audit/v54_inline_decision_parity_summary.csv \
  v59_targeted_feature_audit/*.json \
  v59_targeted_feature_audit/*.csv \
  logs/audit_v59_targeted_features.log \
  A1_internal_neural_scdout_v54_2_online_singlefault/seed*/results/stage_hook_status_v26.json \
  A1_internal_neural_scdout_v54_2_online_singlefault/seed*/results/v53_inline_detector_summary.json
ls -lh "$OUT"
tar -tzf "$OUT" | sed -n '1,120p'
