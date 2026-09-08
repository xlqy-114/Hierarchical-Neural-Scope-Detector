#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/drivevla:${PYTHONPATH:-}"

FORMAL_DIR=${V54_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v54_sample_inline_online_v35_1_scdout_record_audit_smoke_1seed100}
SEED_ROOT=${V54_SEED_ROOT:-A1_internal_neural_scdout_v54_online_singlefault}
SEEDS_CSV=${V54_SEEDS:-954001}
EXPECTED_ROWS=${V54_EXPECTED_ROWS:-100}
DETECTOR_ARTIFACT=${V54_DETECTOR_ARTIFACT:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz}

mkdir -p "$REPO_ROOT/tools_scan" "$REPO_ROOT/pack_output_analysis"
cp -f "$PKG_DIR/tools_scan/"*.py "$REPO_ROOT/tools_scan/"
chmod +x "$REPO_ROOT/tools_scan/"*.py || true

for p in "$FORMAL_DIR" "$DETECTOR_ARTIFACT"; do
  test -e "$p" || { echo "[ERROR] missing required path: $p" >&2; exit 1; }
done

echo "[v54.1 rerun] FORMAL_DIR=$FORMAL_DIR"
echo "[v54.1 rerun] This reruns export/check/pack only; it does not rerun inference."

# Re-run export so the zero-leakage audit CSV has stable headers.
python -u /root/autodl-tmp/DriveVLA/tools_scan/export_v54_online_record_audit_matrices.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --detector-artifact "$DETECTOR_ARTIFACT" \
  --resources-dir "$FORMAL_DIR/resources" \
  --c-target-specs "c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2" \
  --output-dir "$FORMAL_DIR/v54_online_exports" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/export_v54_1_online_record_audit_matrices.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/audit_v54_inline_decision_parity.py \
  --feature-matrix "$FORMAL_DIR/v54_online_exports/v54_online_deployable_feature_matrix.csv" \
  --score-cache "$FORMAL_DIR/v54_online_exports/v54_detector_score_cache.csv" \
  --detector-artifact "$DETECTOR_ARTIFACT" \
  --output-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" \
  --score-atol "${V54_INLINE_SCORE_ATOL:-1e-9}" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/audit_v54_1_inline_decision_parity.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/check_v54_online_smoke_outputs.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --export-dir "$FORMAL_DIR/v54_online_exports" \
  --parity-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" \
  --output-json "$FORMAL_DIR/v54_online_smoke_requirements_check.json" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/check_v54_1_online_smoke_outputs.log"

PACK_PATH=${V54_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v54_1_sample_inline_online_v35_1_scdout_record_audit_smoke_1seed100_for_return.tar.gz}
bash "$PKG_DIR/scripts/pack_v54_online_detector_results.sh" "$FORMAL_DIR" "$PACK_PATH"
