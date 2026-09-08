#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/drivevla:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29656}
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export NUSCENES_DATAROOT=/root/autodl-tmp/DriveVLA/data/nuscenes
export NUSCENES_VERSION=v1.0-trainval
export NUM_WORKERS=0
export SAMPLE_ISOLATION=1
export RESET_MODEL_STATE_BETWEEN_SAMPLES=1
export CLEAR_CUDA_CACHE_BETWEEN_SAMPLES=0
# The inference script still reads the V53_* runtime knob names; v54 sets them explicitly here.
export V53_STAGE_SUMMARY_MAX_SUMMARY_CALLS=${V54_STAGE_SUMMARY_MAX_SUMMARY_CALLS:--1}
export V53_MAX_NEW_TOKENS=${V54_MAX_NEW_TOKENS:-192}

mkdir -p "$REPO_ROOT/drivevla" "$REPO_ROOT/tools_scan" "$REPO_ROOT/pack_output_analysis"
cp -f "$PKG_DIR/drivevla/inference_drivevla_subset1200_v54_online_v35_1_scdout_record_audit.py" "$REPO_ROOT/drivevla/"
cp -f "$PKG_DIR/tools_scan/"*.py "$REPO_ROOT/tools_scan/"
chmod +x "$REPO_ROOT/tools_scan/"*.py || true

BASE_SUBSET=${V54_2_SUBSET_IDS:-${V54_SUBSET_IDS:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json}}
POSITIVE_MIN=${V54_2_POSITIVE_MIN:-${V54_POSITIVE_MIN:-800}}
POSITIVE_MAX=${V54_2_POSITIVE_MAX:-${V54_POSITIVE_MAX:-900}}
EXPECTED_ROWS=${V54_2_EXPECTED_ROWS:-${V54_EXPECTED_ROWS:-1200}}
FORMAL_DIR=${V54_2_FORMAL_DIR:-${V54_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v54_2_sample_inline_online_v35_1_scdout_multipolicy_formal_candidate}}
DETECTOR_ARTIFACT=${V54_2_DETECTOR_ARTIFACT:-${V54_DETECTOR_ARTIFACT:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz}}
CAL_ROOT=${V35_1_CAL_ROOT:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration}
CLEAN_PRED_TRAJS=${V54_2_CLEAN_PRED_TRAJS:-$CAL_ROOT/pred_trajs_dict.json}
if [[ ! -f "$CLEAN_PRED_TRAJS" && -f "$CAL_ROOT/results/pred_trajs_dict.json" ]]; then
  CLEAN_PRED_TRAJS="$CAL_ROOT/results/pred_trajs_dict.json"
fi
CAL_JSON="$CAL_ROOT/results/detector_thresholds.json"
CLEAN_REF="$CAL_ROOT/per_sample_planning_metrics.jsonl"
OUTPUT_SANITY_JSON="$CAL_ROOT/output_sanity_thresholds.json"
SEEDS_CSV=${V54_2_SEEDS:-${V54_SEEDS:-954201}}
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
SEED_ROOT=A1_internal_neural_scdout_v54_2_online_singlefault

REGISTRY_CSV=${REGISTRY_CSV:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260507_FullS_v36_6_required_scan_full_leaf/v36_swide_scan_per_target_effectiveness.csv}
if [[ ! -f "$REGISTRY_CSV" ]]; then
  ALT=/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260507_FullS_v36_6_required_scan_full_leaf/target_registry/v36_swide_targets_full_leaf.csv
  if [[ -f "$ALT" ]]; then REGISTRY_CSV="$ALT"; fi
fi

D_TARGETS_FILE="$PKG_DIR/resources/v49_d_full_targets.txt"
D_WEIGHTS_FILE="$PKG_DIR/resources/v49_d_full_weights.txt"
D_TARGETS="$(cat "$D_TARGETS_FILE")"
D_WEIGHTS="$(cat "$D_WEIGHTS_FILE")"

for p in "$BASE_SUBSET" "$CAL_JSON" "$CLEAN_REF" "$OUTPUT_SANITY_JSON" "$CLEAN_PRED_TRAJS" "$REGISTRY_CSV" "$D_TARGETS_FILE" "$D_WEIGHTS_FILE" "$DETECTOR_ARTIFACT"; do
  test -f "$p" || { echo "[ERROR] missing required file: $p" >&2; exit 1; }
done

rm -rf "$FORMAL_DIR"
mkdir -p "$FORMAL_DIR/logs" "$FORMAL_DIR/resources"
cp -f "$PKG_DIR/resources/"* "$FORMAL_DIR/resources/"

python -u "$PKG_DIR/tools_scan/preflight_v54_exact_requirements.py" \
  --repo-root "$REPO_ROOT" \
  --package-dir "$PKG_DIR" \
  --detector-artifact "$DETECTOR_ARTIFACT" \
  --subset-ids "$BASE_SUBSET" \
  --cal-json "$CAL_JSON" \
  --clean-ref "$CLEAN_REF" \
  --output-sanity-json "$OUTPUT_SANITY_JSON" \
  --model-path /root/autodl-tmp/DriveVLA/ckpts/OpenDriveVLA-0.5B \
  --registry-csv "$REGISTRY_CSV" \
  --output-json "$FORMAL_DIR/logs/v54_preflight_exact_requirements.json" \
  2>&1 | tee "$FORMAL_DIR/logs/v54_preflight_exact_requirements.log"

python -u "$REPO_ROOT/tools_scan/build_v49_s_full_specs_from_registry.py" \
  --registry-csv "$REGISTRY_CSV" \
  --output-dir "$FORMAL_DIR/resources" \
  --exclude-logical-substrings "${V54_S_EXCLUDE_SUBSTRINGS:-query_interact}" \
  2>&1 | tee "$FORMAL_DIR/logs/build_s_full_specs.log"

S_TARGET_SPECS="$(cat "$FORMAL_DIR/resources/v49_s_full_target_specs.txt")"
S_PRIMARY="$(cat "$FORMAL_DIR/resources/v49_s_primary_target_name.txt")"
C_TARGET_SPECS="c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2"

python -u "$REPO_ROOT/tools_scan/build_v50_global_random_weights.py" \
  --s-target-specs-file "$FORMAL_DIR/resources/v49_s_full_target_specs.txt" \
  --d-target-specs-file "$D_TARGETS_FILE" \
  --c-target-specs "$C_TARGET_SPECS" \
  --s0-target-count "${V54_S0_TARGET_COUNT:-7}" \
  --output-dir "$FORMAL_DIR/resources" \
  2>&1 | tee "$FORMAL_DIR/logs/build_global_random_weights.log"

S_TARGET_WEIGHTS="$(cat "$FORMAL_DIR/resources/v50_s_full_target_weights_global_random.txt")"
C_TARGET_WEIGHTS="$(cat "$FORMAL_DIR/resources/v50_c_target_weights_global_random.txt")"
D_WEIGHTS="$(cat "$FORMAL_DIR/resources/v50_d_target_weights_global_random.txt")"
SCOPE_WEIGHTS=${V54_SCOPE_WEIGHTS:-$(cat "$FORMAL_DIR/resources/v50_global_random_scope_weights.txt")}

STAGE_HOOK_ARGS=()
STAGE_HOOK_LABEL="INFERENCE_DEFAULT_FULL_LEGACY"
if [[ -n "${V54_STAGE_HOOK_SPECS:-}" ]]; then
  STAGE_HOOK_ARGS=(--stage-hook-specs "$V54_STAGE_HOOK_SPECS")
  STAGE_HOOK_LABEL="USER_OVERRIDE"
fi

cat <<EOM
============================================================
[v54.2 sample-inline online v35.1 S+C+D+D_out multipolicy]
============================================================
FORMAL_DIR=$FORMAL_DIR
SEEDS=$SEEDS_CSV
SUBSET=$BASE_SUBSET
DETECTOR_ARTIFACT=$DETECTOR_ARTIFACT
CAL_ROOT=$CAL_ROOT
CLEAN_PRED_TRAJS=$CLEAN_PRED_TRAJS
REGISTRY_CSV=$REGISTRY_CSV
Target mode: SCDOUT_internal_online
Scope: S + C + D + D_out internal neural exp-bit faults
Excluded: A / E0 / E1 / E2 / Evaluator / output-level perturbation
Feature timing: collected during current sample inference; detector score written before next sample.
Detector feature policy: exact v35.1 feature columns only for current model.
Extra feature policy: extra online diagnostics are saved separately for future detector improvement.
Posthoc labels: harmful/severe plus v31-compatible heading/speed/speed_alter labels; labels are evaluation-only, never deployable features.
Record audit: feature inventory + target pool manifest + sampled target summary + sample completeness audit + v31-compatible multi-policy label audit.
Global-random scope weights: $SCOPE_WEIGHTS
Stage monitor mode: $STAGE_HOOK_LABEL; max_summary_calls=${V53_STAGE_SUMMARY_MAX_SUMMARY_CALLS}
Max new tokens: ${V53_MAX_NEW_TOKENS}
Positive min/max: ${POSITIVE_MIN}/${POSITIVE_MAX} per seed
Fault: strict single-fault event, exp-bit, bit count 1/2/3=6/3/1, op flip/stuck0/stuck1=1/1/1
============================================================
EOM

for seed_raw in "${SEEDS[@]}"; do
  seed="$(echo "$seed_raw" | xargs)"
  [[ -z "$seed" ]] && continue
  echo "[v54 online detector smoke/formal] running seed $seed"
  SEED_DIR="$FORMAL_DIR/$SEED_ROOT/seed${seed}"
  mkdir -p "$SEED_DIR"
  torchrun --nproc_per_node=1 \
    /root/autodl-tmp/DriveVLA/drivevla/inference_drivevla_subset1200_v54_online_v35_1_scdout_record_audit.py \
    --num-workers 0 \
    --batch-size 1 \
    --fp16 \
    --attn-implementation sdpa \
    --model-path /root/autodl-tmp/DriveVLA/ckpts/OpenDriveVLA-0.5B \
    --subset-ids "$BASE_SUBSET" \
    --output "$SEED_DIR/plan_conv.json" \
    --mode experiment \
    --target-mode SCDOUT_internal_online \
    --experiment-seed "$seed" \
    --positive-min "$POSITIVE_MIN" \
    --positive-max "$POSITIVE_MAX" \
    --cardinality-weights 1:6,2:3,3:1 \
    --bit-op-weights flip:1,stuck0:1,stuck1:1 \
    --fault-event-count-weights 1:1 \
    --mixed-fault-scope-weights "$SCOPE_WEIGHTS" \
    --strict-single-fault \
    --s1-target-specs "$S_TARGET_SPECS" \
    --s1-target-policy weighted \
    --s1-primary-target "$S_PRIMARY" \
    --s1-primary-weight 1 \
    --s1-broad-target-weights "$S_TARGET_WEIGHTS" \
    --c-target-specs "$C_TARGET_SPECS" \
    --c-target-weights "$C_TARGET_WEIGHTS" \
    --d-target-specs "$D_TARGETS" \
    --d-target-weights "$D_WEIGHTS" \
    "${STAGE_HOOK_ARGS[@]}" \
    --max-new-tokens "$V53_MAX_NEW_TOKENS" \
    --can-bus-mlp-hook-suffix vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp \
    --reset-model-state-between-samples \
    --calibration-json "$CAL_JSON" \
    --clean-ref "$CLEAN_REF" \
    --output-sanity-json "$OUTPUT_SANITY_JSON" \
    --v53-inline-detector \
    --v53-detector-artifact "$DETECTOR_ARTIFACT" \
    --v53-inline-strict \
    2>&1 | tee "$FORMAL_DIR/logs/seed${seed}.log"

  INLINE_JSONL="$SEED_DIR/results/v53_inline_detector_records.jsonl"
  INLINE_SUMMARY="$SEED_DIR/results/v53_inline_detector_summary.json"
  if [[ ! -s "$INLINE_JSONL" || ! -s "$INLINE_SUMMARY" ]]; then
    echo "[ERROR] v54 sample-inline detector outputs missing for seed $seed" >&2
    echo "[ERROR] expected: $INLINE_JSONL and $INLINE_SUMMARY" >&2
    exit 4
  fi
  python -c "from pathlib import Path; p=Path('$INLINE_JSONL'); print('[v54 inline detector check]', p, 'rows=', sum(1 for _ in p.open('r', encoding='utf-8')))"
done

python -u /root/autodl-tmp/DriveVLA/tools_scan/check_v54_scdout_internal_effectiveness.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --output-json "$FORMAL_DIR/v54_scdout_internal_effectiveness_summary.json" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/check_effectiveness.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/export_v54_online_record_audit_matrices.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --detector-artifact "$DETECTOR_ARTIFACT" \
  --resources-dir "$FORMAL_DIR/resources" \
  --c-target-specs "$C_TARGET_SPECS" \
  --output-dir "$FORMAL_DIR/v54_online_exports" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/export_v54_online_record_audit_matrices.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/build_v54_2_multipolicy_posthoc_labels.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --export-dir "$FORMAL_DIR/v54_online_exports" \
  --clean-pred-trajs "$CLEAN_PRED_TRAJS" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/build_v54_2_multipolicy_posthoc_labels.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/audit_v54_inline_decision_parity.py \
  --feature-matrix "$FORMAL_DIR/v54_online_exports/v54_online_deployable_feature_matrix.csv" \
  --score-cache "$FORMAL_DIR/v54_online_exports/v54_detector_score_cache.csv" \
  --detector-artifact "$DETECTOR_ARTIFACT" \
  --output-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" \
  --score-atol "${V54_INLINE_SCORE_ATOL:-1e-9}" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/audit_v54_inline_decision_parity.log"

python -u /root/autodl-tmp/DriveVLA/tools_scan/check_v54_online_smoke_outputs.py \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$EXPECTED_ROWS" \
  --export-dir "$FORMAL_DIR/v54_online_exports" \
  --parity-dir "$FORMAL_DIR/v54_inline_decision_parity_audit" \
  --output-json "$FORMAL_DIR/v54_online_smoke_requirements_check.json" \
  --strict 2>&1 | tee "$FORMAL_DIR/logs/check_v54_online_smoke_outputs.log"

echo "[v59.3] skip intermediate v54 pack; v59 audit/compact pack will run next"
