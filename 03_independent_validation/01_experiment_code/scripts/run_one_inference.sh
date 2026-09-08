#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <subset_json> <expected_rows> <seed> <positive_min> <positive_max> <run_dir>" >&2
  exit 2
fi
SUBSET=$1; EXPECTED=$2; SEED=$3; POSMIN=$4; POSMAX=$5; RUN_DIR=$6
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
cd "$REPO"
source activate drivevla
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/drivevla:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29701}
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export NUSCENES_DATAROOT=$REPO/data/nuscenes NUSCENES_VERSION=v1.0-trainval NUM_WORKERS=0
export SAMPLE_ISOLATION=1 RESET_MODEL_STATE_BETWEEN_SAMPLES=1 CLEAR_CUDA_CACHE_BETWEEN_SAMPLES=0
export V53_STAGE_SUMMARY_MAX_SUMMARY_CALLS=-1 V53_MAX_NEW_TOKENS=192

test -s "$SUBSET" || { echo "[ERROR] missing subset: $SUBSET" >&2; exit 3; }
test -s "$ROOT/frozen_policy/frozen_policy.json" || { echo '[ERROR] frozen policy missing' >&2; exit 4; }
V35=$ROOT/inputs/v35_1_exported_selected_detectors_for_reuse.tar.gz
CAL=$REPO/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration
CAL_JSON=$CAL/results/detector_thresholds.json
OUTPUT_SANITY=$CAL/output_sanity_thresholds.json
REGISTRY=$REPO/output/OpenDriveVLA-0.5B/20260507_FullS_v36_6_required_scan_full_leaf/v36_swide_scan_per_target_effectiveness.csv
for p in "$V35" "$CAL_JSON" "$OUTPUT_SANITY" "$REGISTRY"; do test -s "$p" || { echo "[ERROR] missing $p" >&2; exit 5; }; done

COMPAT_PY=$ROOT/sklearn122_env/bin/python
COMPAT_SERVER=$PKG/tools/v35_sklearn122_server.py
if [[ ! -x "$COMPAT_PY" ]]; then
  echo "[ERROR] sklearn-1.2.2 compatibility environment is missing." >&2
  echo "Run: bash $PKG/scripts/00b_prepare_v35_sklearn122_env.sh" >&2
  exit 6
fi
test -s "$COMPAT_SERVER" || { echo "[ERROR] missing compatibility server: $COMPAT_SERVER" >&2; exit 7; }
export R1C1_V35_COMPAT_PYTHON="$COMPAT_PY"
export R1C1_V35_COMPAT_SERVER="$COMPAT_SERVER"
COMPAT_VER=$("$COMPAT_PY" - <<'PY2'
import sklearn; print(sklearn.__version__)
PY2
)
[[ "$COMPAT_VER" == "1.2.2" ]] || { echo "[ERROR] expected compat sklearn 1.2.2, got $COMPAT_VER" >&2; exit 8; }
echo "v35.1 scoring runtime: isolated scikit-learn $COMPAT_VER"
echo "final Table5 model runtime: main DriveVLA scikit-learn $(python - <<'PY2'
import sklearn; print(sklearn.__version__)
PY2
)"

# The full modified inference implementation is copied as a new version; no patching of historical code.
cp -f "$PKG/drivevla/inference_drivevla_r1c1_table5_broader4819_v1.py" "$REPO/drivevla/inference_drivevla_r1c1_table5_broader4819_v1.py"
mkdir -p "$RUN_DIR" "$RUN_DIR/resources"
python -u "$PKG/tools/build_v49_s_full_specs_from_registry.py" \
  --registry-csv "$REGISTRY" --output-dir "$RUN_DIR/resources" --exclude-logical-substrings query_interact \
  2>&1 | tee "$RUN_DIR/build_s_full_specs.log"
S_SPECS=$(cat "$RUN_DIR/resources/v49_s_full_target_specs.txt")
S_PRIMARY=$(cat "$RUN_DIR/resources/v49_s_primary_target_name.txt")
C_SPECS='c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2'
python -u "$PKG/tools/build_v50_global_random_weights.py" \
  --s-target-specs-file "$RUN_DIR/resources/v49_s_full_target_specs.txt" \
  --d-target-specs-file "$PKG/resources/v49_d_full_targets.txt" \
  --c-target-specs "$C_SPECS" --s0-target-count 7 --output-dir "$RUN_DIR/resources" \
  2>&1 | tee "$RUN_DIR/build_global_random_weights.log"
S_WEIGHTS=$(cat "$RUN_DIR/resources/v50_s_full_target_weights_global_random.txt")
C_WEIGHTS=$(cat "$RUN_DIR/resources/v50_c_target_weights_global_random.txt")
D_WEIGHTS=$(cat "$RUN_DIR/resources/v50_d_target_weights_global_random.txt")
SCOPE_WEIGHTS=$(cat "$RUN_DIR/resources/v50_global_random_scope_weights.txt")
D_TARGETS=$(cat "$PKG/resources/v49_d_full_targets.txt")

cat <<EOF
[R1C1 held-out inference]
subset=$SUBSET
expected_rows=$EXPECTED
seed=$SEED
positive_min/max=$POSMIN/$POSMAX
policy=$ROOT/frozen_policy/frozen_policy.json
CLEAN REFERENCE: NOT PASSED TO INFERENCE
GT ANALYSIS METRICS: DISABLED DURING ONLINE SCORING
final Table5 alarm timing: after current sample forward, before next sample
EOF

torchrun --nproc_per_node=1 "$REPO/drivevla/inference_drivevla_r1c1_table5_broader4819_v1.py" \
  --num-workers 0 --batch-size 1 --fp16 --attn-implementation sdpa \
  --model-path "$REPO/ckpts/OpenDriveVLA-0.5B" --subset-ids "$SUBSET" \
  --output "$RUN_DIR/plan_conv.json" --mode experiment --target-mode SCDOUT_internal_online \
  --experiment-seed "$SEED" --positive-min "$POSMIN" --positive-max "$POSMAX" \
  --cardinality-weights 1:6,2:3,3:1 --bit-op-weights flip:1,stuck0:1,stuck1:1 \
  --fault-event-count-weights 1:1 --mixed-fault-scope-weights "$SCOPE_WEIGHTS" --strict-single-fault \
  --s1-target-specs "$S_SPECS" --s1-target-policy weighted --s1-primary-target "$S_PRIMARY" --s1-primary-weight 1 --s1-broad-target-weights "$S_WEIGHTS" \
  --c-target-specs "$C_SPECS" --c-target-weights "$C_WEIGHTS" \
  --d-target-specs "$D_TARGETS" --d-target-weights "$D_WEIGHTS" \
  --stage-hook-strict --max-new-tokens 192 \
  --can-bus-mlp-hook-suffix vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp \
  --reset-model-state-between-samples --calibration-json "$CAL_JSON" --output-sanity-json "$OUTPUT_SANITY" \
  --v53-inline-detector --v53-detector-artifact "$V35" --v53-inline-strict --v53-disable-inline-analysis-metrics \
  --r1c1-table5-policy-dir "$ROOT/frozen_policy" \
  2>&1 | tee "$RUN_DIR/inference.log"

python -u "$PKG/tools/check_online_run.py" --run-dir "$RUN_DIR" --expected-rows "$EXPECTED" --policy-dir "$ROOT/frozen_policy" --output-dir "$RUN_DIR/reviewer_online_export" \
  2>&1 | tee "$RUN_DIR/check_online_run.log"
echo "[PASS] inference complete: $RUN_DIR"
