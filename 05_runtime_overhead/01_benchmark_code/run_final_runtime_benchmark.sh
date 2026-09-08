#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/DriveVLA
PACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN="$REPO/output/OpenDriveVLA-0.5B/R1C4_threshold_sensitivity_5seeds"
SEED_ROOT=A1_internal_neural_scdout_v54_2_online_singlefault
MODEL_DIR="$REPO/pack_output_analysis/R2C6_sync_profile_models"
OUT="$REPO/output/OpenDriveVLA-0.5B/R2C6_synchronization_profile"
RETURN="$REPO/pack_output_analysis/R2C6_synchronization_overhead_for_return.tar.gz"
CAL="$REPO/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration"

cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla

# Same single-process inference environment as the successful v59 campaign.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/drivevla:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29676}
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export NUSCENES_DATAROOT="$REPO/data/nuscenes"
export NUSCENES_VERSION=v1.0-trainval
export NUM_WORKERS=0
export SAMPLE_ISOLATION=1
export RESET_MODEL_STATE_BETWEEN_SAMPLES=1
export CLEAR_CUDA_CACHE_BETWEEN_SAMPLES=0
export OMP_NUM_THREADS=1

bash "$PACK/precheck.sh"

mkdir -p "$REPO/drivevla" "$REPO/pack_output_analysis"

# Copy the exact successful inference source byte-for-byte to a unique import path.
cp -f \
 "$PACK/exact_source/inference_drivevla_subset1200_v59_online_targeted_feature_audit.py" \
 "$REPO/drivevla/r1c6_final_exact_runtime_source.py"

echo "===== OFFLINE FINAL RUNTIME ARTIFACT PREPARATION (NOT TIMED) ====="
rm -rf "$MODEL_DIR"
mkdir -p "$MODEL_DIR"
python "$PACK/00_prepare_final_runtime_artifacts.py" \
  --campaign-dir "$CAMPAIGN" \
  --seed-root "$SEED_ROOT" \
  --contract-dir "$PACK/contract" \
  --dependency-dir "$PACK/dependency" \
  --output-dir "$MODEL_DIR"

echo "===== FINAL COMMENT-6 LEAN RUNTIME BENCHMARK ====="
rm -rf "$OUT"
mkdir -p "$OUT"

torchrun --nproc_per_node=1 \
 "$PACK/01_final_lean_runtime_benchmark.py" \
 --runtime-source "$REPO/drivevla/r1c6_final_exact_runtime_source.py" \
 --project-root "$REPO" \
 --model-path "$REPO/ckpts/OpenDriveVLA-0.5B" \
 --subset-ids "$REPO/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json" \
 --detector-artifact "$REPO/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz" \
 --detector-thresholds "$CAL/detector_thresholds.json" \
 --output-sanity-thresholds "$CAL/output_sanity_thresholds.json" \
 --runtime-model-dir "$MODEL_DIR" \
 --contract-dir "$PACK/contract" \
 --dependency-dir "$PACK/dependency" \
 --output-dir "$OUT" \
 --warmup 20 \
 --measured 100 \
 --num-workers 0 \
 --max-new-tokens 192 \
 --attn-implementation sdpa \
 --fp16

python "$PACK/02_make_comment6_summary.py" --result-dir "$OUT"
python "$PACK/03_make_synchronization_summary.py" --result-dir "$OUT"

tar -czf "$RETURN" \
  -C "$REPO" \
  output/OpenDriveVLA-0.5B/R2C6_synchronization_profile \
  pack_output_analysis/R2C6_sync_profile_models/runtime_model_metadata.json

echo "===== R1C6 COMMENT 6 FINAL DONE ====="
ls -lh "$RETURN"
