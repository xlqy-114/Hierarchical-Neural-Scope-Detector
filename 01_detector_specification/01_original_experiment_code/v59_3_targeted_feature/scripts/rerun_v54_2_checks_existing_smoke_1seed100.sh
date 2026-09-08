#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/drivevla:${PYTHONPATH:-}"
mkdir -p "$REPO_ROOT/tools_scan"
cp -f "$PKG_DIR/tools_scan/"*.py "$REPO_ROOT/tools_scan/"
chmod +x "$REPO_ROOT/tools_scan/"*.py || true
FORMAL_DIR=${V54_2_SMOKE_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v54_2_sample_inline_online_v35_1_scdout_multipolicy_smoke_1seed100}
SEED_ROOT=A1_internal_neural_scdout_v54_2_online_singlefault
SEEDS_CSV=${V54_2_SEEDS:-954201}
EXPECTED_ROWS=${V54_2_EXPECTED_ROWS:-100}
DETECTOR_ARTIFACT=${V54_2_DETECTOR_ARTIFACT:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz}
CAL_ROOT=${V35_1_CAL_ROOT:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration}
CLEAN_PRED_TRAJS=${V54_2_CLEAN_PRED_TRAJS:-$CAL_ROOT/pred_trajs_dict.json}
if [[ ! -f "$CLEAN_PRED_TRAJS" && -f "$CAL_ROOT/results/pred_trajs_dict.json" ]]; then CLEAN_PRED_TRAJS="$CAL_ROOT/results/pred_trajs_dict.json"; fi
C_TARGET_SPECS="c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2"
echo "[v54.2 rerun] FORMAL_DIR=$FORMAL_DIR"
echo "[v54.2 rerun] This reruns export/multipolicy/check/pack only; it does not rerun inference."
python -u /root/autodl-tmp/DriveVLA/tools_scan/export_v54_online_record_audit_matrices.py --campaign-dir "$FORMAL_DIR" --seed-root "$SEED_ROOT" --seeds "$SEEDS_CSV" --detector-artifact "$DETECTOR_ARTIFACT" --resources-dir "$FORMAL_DIR/resources" --c-target-specs "$C_TARGET_SPECS" --output-dir "$FORMAL_DIR/v54_online_exports" --strict
python -u /root/autodl-tmp/DriveVLA/tools_scan/build_v54_2_multipolicy_posthoc_labels.py --campaign-dir "$FORMAL_DIR" --seed-root "$SEED_ROOT" --seeds "$SEEDS_CSV" --export-dir "$FORMAL_DIR/v54_online_exports" --clean-pred-trajs "$CLEAN_PRED_TRAJS" --strict
python -u /root/autodl-tmp/DriveVLA/tools_scan/audit_v54_inline_decision_parity.py --feature-matrix "$FORMAL_DIR/v54_online_exports/v54_online_deployable_feature_matrix.csv" --score-cache "$FORMAL_DIR/v54_online_exports/v54_detector_score_cache.csv" --detector-artifact "$DETECTOR_ARTIFACT" --output-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" --score-atol "${V54_INLINE_SCORE_ATOL:-1e-9}" --strict
python -u /root/autodl-tmp/DriveVLA/tools_scan/check_v54_online_smoke_outputs.py --campaign-dir "$FORMAL_DIR" --seed-root "$SEED_ROOT" --seeds "$SEEDS_CSV" --expected-rows "$EXPECTED_ROWS" --export-dir "$FORMAL_DIR/v54_online_exports" --parity-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" --output-json "$FORMAL_DIR/v54_online_smoke_requirements_check.json" --strict
bash "$PKG_DIR/scripts/pack_v54_online_detector_results.sh" "$FORMAL_DIR" "${V54_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v54_2_sample_inline_online_v35_1_scdout_multipolicy_smoke_1seed100_for_return.tar.gz}"
