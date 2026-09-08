#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${V73_16_REPO_ROOT:-/root/autodl-tmp/DriveVLA}
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/drivevla:$PKG_DIR:${PYTHONPATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29740}
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export NUSCENES_DATAROOT=${NUSCENES_DATAROOT:-$REPO_ROOT/data/nuscenes}
export NUSCENES_VERSION=${NUSCENES_VERSION:-v1.0-trainval}
export NUM_WORKERS=${NUM_WORKERS:-0}
export SAMPLE_ISOLATION=1
export RESET_MODEL_STATE_BETWEEN_SAMPLES=1
export CLEAR_CUDA_CACHE_BETWEEN_SAMPLES=0
export V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS=${V73_16_STAGE_SUMMARY_MAX_SUMMARY_CALLS:--1}
export V50_MAX_NEW_TOKENS=${V73_16_MAX_NEW_TOKENS:-192}

mkdir -p "$REPO_ROOT/drivevla" "$REPO_ROOT/pack_output_analysis" "$REPO_ROOT/code_pack"
cp -f "$PKG_DIR/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" "$REPO_ROOT/drivevla/"
cp -f "$PKG_DIR/drivevla/v73_16_data_contract_filter.py" "$REPO_ROOT/drivevla/"

MODEL_PATH=${V73_16_MODEL_PATH:-$REPO_ROOT/ckpts/OpenDriveVLA-0.5B}
OFFICIAL_IDS=${V73_16_OFFICIAL_FULL_SAMPLE_IDS:-$PKG_DIR/resources/official_full_sample_ids_6019.json}
SAMPLE_COUNT=${V73_16_SAMPLE_COUNT:-100}
SEEDS_CSV=${V73_16_SEEDS:-957001}
RUN_TAG=${V73_16_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_v73_16_standalone_no_detector}
FORMAL_DIR=${V73_16_FORMAL_DIR:-$REPO_ROOT/output/OpenDriveVLA-0.5B/$RUN_TAG}
PACK_PATH=${V73_16_PACK_PATH:-$REPO_ROOT/pack_output_analysis/v73_16_standalone_no_detector_${RUN_TAG}_for_return.tar.gz}
STRICT_AUDIT=${V73_16_STRICT_AUDIT:-1}
FORCE_CLEAN_PASS=${V73_16_FORCE_CLEAN_PASS:-0}
CLEAN_DIR=${V73_16_CLEAN_DIR:-$REPO_ROOT/output/OpenDriveVLA-0.5B/v73_16_clean_reference_official${SAMPLE_COUNT}_maxtok${V50_MAX_NEW_TOKENS}}
RUN_IDS_JSON=${V73_16_RUN_SAMPLE_IDS:-$FORMAL_DIR/resources/run_sample_ids_${SAMPLE_COUNT}.json}

mkdir -p "$FORMAL_DIR/logs" "$FORMAL_DIR/resources" "$CLEAN_DIR"
cp -f "$PKG_DIR/resources/"* "$FORMAL_DIR/resources/" 2>/dev/null || true

python -u "$PKG_DIR/tools_scan/make_v73_16_run_sample_ids.py" \
  --official-full-sample-ids "$OFFICIAL_IDS" \
  --count "$SAMPLE_COUNT" \
  --output-json "$RUN_IDS_JSON" \
  2>&1 | tee "$FORMAL_DIR/logs/make_run_sample_ids.log"

python -u "$PKG_DIR/tools_scan/preflight_v73_16_standalone_resources.py" \
  --repo-root "$REPO_ROOT" \
  --package-dir "$PKG_DIR" \
  --run-sample-ids "$RUN_IDS_JSON" \
  --expected-rows "$SAMPLE_COUNT" \
  --model-path "$MODEL_PATH" \
  --output-json "$FORMAL_DIR/logs/v73_16_preflight_standalone_resources.json" \
  2>&1 | tee "$FORMAL_DIR/logs/v73_16_preflight_standalone_resources.log"

S_TARGET_SPECS_FILE="$PKG_DIR/resources/v49_s_full_target_specs.txt"
S_PRIMARY_FILE="$PKG_DIR/resources/v49_s_primary_target_name.txt"
S_TARGET_WEIGHTS_FILE="$PKG_DIR/resources/v50_s_full_target_weights_global_random.txt"
C_TARGET_WEIGHTS_FILE="$PKG_DIR/resources/v50_c_target_weights_global_random.txt"
D_TARGETS_FILE="$PKG_DIR/resources/v49_d_full_targets.txt"
D_WEIGHTS_FILE="$PKG_DIR/resources/v50_d_target_weights_global_random.txt"
SCOPE_WEIGHTS_FILE="$PKG_DIR/resources/v50_global_random_scope_weights.txt"
GLOBAL_MANIFEST="$PKG_DIR/resources/v50_internal_scd_global_random_scope_manifest.json"
STAGE_HOOK_SPECS_FILE="$PKG_DIR/resources/v73_16_standalone_required_stage_hook_specs.txt"

for p in "$RUN_IDS_JSON" "$MODEL_PATH" "$STAGE_HOOK_SPECS_FILE" "$S_TARGET_SPECS_FILE" "$S_PRIMARY_FILE" "$S_TARGET_WEIGHTS_FILE" "$C_TARGET_WEIGHTS_FILE" "$D_TARGETS_FILE" "$D_WEIGHTS_FILE" "$SCOPE_WEIGHTS_FILE" "$GLOBAL_MANIFEST"; do
  test -e "$p" || { echo "[ERROR] missing required path: $p" >&2; exit 1; }
done

S_TARGET_SPECS="$(cat "$S_TARGET_SPECS_FILE")"
S_PRIMARY="$(cat "$S_PRIMARY_FILE")"
S_TARGET_WEIGHTS="$(cat "$S_TARGET_WEIGHTS_FILE")"
C_TARGET_SPECS="c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2"
C_TARGET_WEIGHTS="$(cat "$C_TARGET_WEIGHTS_FILE")"
D_TARGETS="$(cat "$D_TARGETS_FILE")"
D_WEIGHTS="$(cat "$D_WEIGHTS_FILE")"
SCOPE_WEIGHTS=${V73_16_SCOPE_WEIGHTS:-$(cat "$SCOPE_WEIGHTS_FILE")}
STAGE_HOOK_SPECS="$(cat "$STAGE_HOOK_SPECS_FILE")"
SEED_ROOT=A1_internal_neural_scdout_v73_16_standalone_singlefault

read POSITIVE_MIN POSITIVE_MAX < <(python - <<PY2
import math
n=int('$SAMPLE_COUNT')
print(int(math.ceil(n*0.80)), int(math.floor(n*0.85)))
PY2
)

cat <<EOM
============================================================
[v73.16 standalone no-detector inference]
============================================================
FORMAL_DIR=$FORMAL_DIR
CLEAN_DIR=$CLEAN_DIR
SEEDS=$SEEDS_CSV
SAMPLE_COUNT=$SAMPLE_COUNT
RUN_IDS_JSON=$RUN_IDS_JSON
positive_count_range=$POSITIVE_MIN..$POSITIVE_MAX (per seed; actual count is seed-random in this range)
MODEL_PATH=$MODEL_PATH
STAGE_HOOK_SPECS_FILE=$STAGE_HOOK_SPECS_FILE
V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS=$V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS
V50_MAX_NEW_TOKENS=$V50_MAX_NEW_TOKENS
STRICT_AUDIT=$STRICT_AUDIT
PACK_PATH=$PACK_PATH
Policy: standalone clean pass + standalone fault pass. No v35/v54/v59/v72 historical outputs are read. Clean reference is cached/reused by V73_16_CLEAN_DIR when available.
Detector: disabled; no inline detector artifacts, no score/alarm/threshold/hgbmax outputs.
Outputs: 2399 runtime/source features, 1502 model-ready features, posthoc L2/collision 1s/2s/3s/avg raw ingredients.
Training: not run by this script. Use scripts/run_v73_16_train_selected_features.sh explicitly.
============================================================
EOM

CLEAN_METRICS="$CLEAN_DIR/per_sample_planning_metrics.jsonl"
CLEAN_PREDS="$CLEAN_DIR/pred_trajs_dict.json"
CLEAN_SANITY="$CLEAN_DIR/output_sanity_thresholds.json"
CLEAN_CAL_JSON="$CLEAN_DIR/results/feature_calibration_stats.json"
LEGACY_CLEAN_CAL_JSON="$CLEAN_DIR/results/detector_thresholds.json"
if [[ "$FORCE_CLEAN_PASS" == "1" || ! -s "$CLEAN_METRICS" || ! -s "$CLEAN_PREDS" || ! -s "$CLEAN_SANITY" || ( ! -s "$CLEAN_CAL_JSON" && ! -s "$LEGACY_CLEAN_CAL_JSON" ) ]]; then
  echo "[v73.16] running clean pass to build local clean reference"
  rm -rf "$CLEAN_DIR"
  mkdir -p "$CLEAN_DIR"
  torchrun --nproc_per_node=1 \
    "$REPO_ROOT/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" \
    --num-workers 0 \
    --batch-size 1 \
    --fp16 \
    --attn-implementation sdpa \
    --model-path "$MODEL_PATH" \
    --subset-ids "$RUN_IDS_JSON" \
    --output "$CLEAN_DIR/plan_conv.json" \
    --mode calibrate \
    --target-mode SCDOUT_internal_online \
    --stage-hook-specs "$STAGE_HOOK_SPECS" \
    --v63-decode-token-features \
    --v63-decode-hook-suffix lm_head \
    --v63-decode-max-calls 64 \
    --v63-decode-topk 10 \
    --max-new-tokens "$V50_MAX_NEW_TOKENS" \
    --can-bus-mlp-hook-suffix vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp \
    --reset-model-state-between-samples \
    --v73-15-standalone-feature-run \
    2>&1 | tee "$FORMAL_DIR/logs/clean_pass.log"

  python -u "$PKG_DIR/tools_scan/materialize_v73_16_clean_reference.py" \
    --result-dir "$CLEAN_DIR" \
    --inference-module-path "$REPO_ROOT/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" \
    --expected-rows "$SAMPLE_COUNT" \
    --summary-json "$FORMAL_DIR/logs/v73_16_clean_reference_summary.json" \
    --strict \
    2>&1 | tee "$FORMAL_DIR/logs/materialize_clean_reference.log"
else
  echo "[v73.16] reusing existing clean reference: $CLEAN_DIR"
fi

# v73.16 uses feature_calibration_stats.json. If a reused clean cache was produced by an older
# package, migrate the legacy detector_thresholds.json name once without treating it as a detector artifact.
if [[ ! -s "$CLEAN_CAL_JSON" && -s "$LEGACY_CLEAN_CAL_JSON" ]]; then
  mkdir -p "$(dirname "$CLEAN_CAL_JSON")"
  cp -f "$LEGACY_CLEAN_CAL_JSON" "$CLEAN_CAL_JSON"
fi
if [[ -s "$CLEAN_CAL_JSON" && ! -s "$LEGACY_CLEAN_CAL_JSON" ]]; then
  # Compatibility alias for any imported helper that still expects the historical filename.
  cp -f "$CLEAN_CAL_JSON" "$LEGACY_CLEAN_CAL_JSON"
fi

for p in "$CLEAN_METRICS" "$CLEAN_PREDS" "$CLEAN_SANITY" "$CLEAN_CAL_JSON"; do
  test -s "$p" || { echo "[ERROR] missing clean-pass output: $p" >&2; exit 2; }
done

# Always validate and document the clean reference actually used, including reused cache.
python -u "$PKG_DIR/tools_scan/materialize_v73_16_clean_reference.py" \
  --result-dir "$CLEAN_DIR" \
  --inference-module-path "$REPO_ROOT/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" \
  --expected-rows "$SAMPLE_COUNT" \
  --summary-json "$FORMAL_DIR/logs/v73_16_clean_reference_summary.json" \
  --strict \
  2>&1 | tee "$FORMAL_DIR/logs/materialize_clean_reference_final_validation.log"

python - <<PY_CLEAN_MANIFEST
import json, os, time
from pathlib import Path
clean_dir=Path(os.environ.get('CLEAN_DIR', '$CLEAN_DIR'))
files={
  'pred_trajs_dict_json': clean_dir/'pred_trajs_dict.json',
  'per_sample_planning_metrics_jsonl': clean_dir/'per_sample_planning_metrics.jsonl',
  'output_sanity_thresholds_json': clean_dir/'output_sanity_thresholds.json',
  'planning_metrics_json': clean_dir/'planning_metrics.json',
  'feature_calibration_stats_json': clean_dir/'results'/'feature_calibration_stats.json',
  'legacy_calibration_alias_json': clean_dir/'results'/'detector_thresholds.json',
}
out={
  'version':'v73_16_clean_reference_manifest',
  'clean_dir': str(clean_dir),
  'force_clean_pass': '$FORCE_CLEAN_PASS',
  'sample_count': int('$SAMPLE_COUNT'),
  'max_new_tokens': int('$V50_MAX_NEW_TOKENS'),
  'created_at_unix': time.time(),
  'files': {k:{'path':str(v),'exists':v.exists(),'size_bytes':(v.stat().st_size if v.exists() else None)} for k,v in files.items()},
}
Path('$FORMAL_DIR/logs/v73_16_clean_reference_manifest.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('[v73.16 clean manifest]', json.dumps(out, indent=2))
PY_CLEAN_MANIFEST

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
for seed_raw in "${SEEDS[@]}"; do
  seed="$(echo "$seed_raw" | xargs)"
  [[ -z "$seed" ]] && continue
  echo "[v73.16 fault pass] running seed $seed"
  SEED_DIR="$FORMAL_DIR/$SEED_ROOT/seed${seed}"
  rm -rf "$SEED_DIR"
  mkdir -p "$SEED_DIR"
  torchrun --nproc_per_node=1 \
    "$REPO_ROOT/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" \
    --num-workers 0 \
    --batch-size 1 \
    --fp16 \
    --attn-implementation sdpa \
    --model-path "$MODEL_PATH" \
    --subset-ids "$RUN_IDS_JSON" \
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
    --stage-hook-specs "$STAGE_HOOK_SPECS" \
    --v63-decode-token-features \
    --v63-decode-hook-suffix lm_head \
    --v63-decode-max-calls 64 \
    --v63-decode-topk 10 \
    --max-new-tokens "$V50_MAX_NEW_TOKENS" \
    --can-bus-mlp-hook-suffix vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp \
    --reset-model-state-between-samples \
    --calibration-json "$CLEAN_CAL_JSON" \
    --clean-ref "$CLEAN_METRICS" \
    --output-sanity-json "$CLEAN_SANITY" \
    --v73-15-standalone-feature-run \
    2>&1 | tee "$FORMAL_DIR/logs/seed${seed}.log"

  if compgen -G "$SEED_DIR/results/v53_inline_detector_records*" >/dev/null; then
    echo "[ERROR] v53 inline detector files exist in standalone no-detector run: $SEED_DIR/results" >&2
    exit 4
  fi
  if compgen -G "$SEED_DIR/results/detector_records*" >/dev/null || [[ -f "$SEED_DIR/results/detector_report.json" ]]; then
    echo "[ERROR] legacy detector output files exist in standalone no-detector run: $SEED_DIR/results" >&2
    exit 4
  fi
  test -s "$SEED_DIR/results/v73_16_feature_records.jsonl" || { echo "[ERROR] missing v73_16_feature_records.jsonl for seed $seed" >&2; exit 5; }
  python -c "from pathlib import Path; p=Path('$SEED_DIR/results/v73_16_feature_records.jsonl'); print('[v73.16 feature_records check]', p, 'rows=', sum(1 for _ in p.open('r', encoding='utf-8')))"
done

python -u "$PKG_DIR/tools_scan/summarize_v73_16_fault_plan.py" \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --sample-count "$SAMPLE_COUNT" \
  --output-json "$FORMAL_DIR/v73_16_fault_plan_manifest.json" \
  2>&1 | tee "$FORMAL_DIR/logs/summarize_fault_plan.log"

python -u "$PKG_DIR/tools_scan/export_v73_16_standalone_matrices.py" \
  --campaign-dir "$FORMAL_DIR" \
  --seed-root "$SEED_ROOT" \
  --seeds "$SEEDS_CSV" \
  --expected-rows "$SAMPLE_COUNT" \
  --resources-dir "$PKG_DIR/resources" \
  --output-dir "$FORMAL_DIR/v73_16_standalone_exports" \
  --inference-module-path "$REPO_ROOT/drivevla/inference_drivevla_official6019_v73_16_standalone_no_detector.py" \
  --clean-ref "$CLEAN_METRICS" \
  --output-sanity-json "$CLEAN_SANITY" \
  --clean-pred-trajs "$CLEAN_PREDS" \
  2>&1 | tee "$FORMAL_DIR/logs/export_v73_16_standalone_matrices.log"

# Model training is intentionally not part of the inference/data-generation pipeline.
# Use scripts/run_v73_16_train_selected_features.sh explicitly when training is needed.

AUDIT_STATUS=0
if [[ "$STRICT_AUDIT" == "1" ]]; then
  python - <<PY3 || AUDIT_STATUS=$?
import json, sys
from pathlib import Path
p=Path('$FORMAL_DIR/v73_16_standalone_exports/v73_16_no_detector_export_summary.json')
d=json.loads(p.read_text())
print('[v73.16 strict audit] pass=', d.get('pass'))
if not d.get('pass'):
    print(json.dumps({k:d.get(k) for k in ['rows','expected_rows_total','missing_feature_count','all_nan_feature_count','forbidden_feature_count','underlying_raw_missing_cell_count_report_only','underlying_raw_source_column_count','underlying_raw_source_all_nan_column_count','analysis_missing_columns','analysis_all_nan_columns','availability','outputs']}, indent=2))
    sys.exit(7)
PY3
fi

bash "$PKG_DIR/scripts/pack_v73_16_standalone_outputs.sh" "$FORMAL_DIR" "$PACK_PATH"

if [[ "$AUDIT_STATUS" != "0" ]]; then
  echo "[ERROR] strict audit failed, but outputs were packed for diagnosis: $PACK_PATH" >&2
  exit "$AUDIT_STATUS"
fi
