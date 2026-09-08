#!/usr/bin/env bash
set -euo pipefail
ROOT=${UNIAD2_ROOT:-/root/autodl-tmp/UniAD2}
export V4492_NUM_THREADS=${V4492_NUM_THREADS:-8}
DATA_ROOT=${V4492_DATA_ROOT:-$ROOT/outputs/direct_fault_repair_extended_features_v4_3_1}
CONTRACT=${V4492_FROZEN_CONTRACT_JSON:-$ROOT/outputs/detector_repair_aware_full_exact_contract_19seeds_v4_4_6_2_20260619_120008/frozen_exact_feature_contract_v4_4_6_2.json}
TRAIN_SEEDS=${V4492_TRAIN_SEEDS:-959101,959102,959103,959104,959105,959113,959114,959115,959116,959117,959118,959119,959120,959121,959122}
VAL_SEEDS=${V4492_VAL_SEEDS:-959106,959123}
TEST_SEEDS=${V4492_TEST_SEEDS:-959107,959124}
OUT=${V4492_OUT_DIR:-$ROOT/outputs/detector_all_eligible_c1score_c2_weight_sweep_v4_4_9_2_$(date +%Y%m%d_%H%M%S)}
LOG=${V4492_LOG:-$ROOT/logs/detector_all_eligible_c1score_c2_weight_sweep_v4_4_9_2_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$ROOT/logs" "$(dirname "$OUT")"; cd "$ROOT"
python -u "$ROOT/code_pack/uniad2_v4_4_9_2_all_eligible_c1score_c2_weight_sweep_pack/tools_uniad/train_all_eligible_c1score_c2_weight_sweep_v4_4_9_2.py"  --data-root "$DATA_ROOT" --frozen-contract-json "$CONTRACT" --out-dir "$OUT"  --train-seeds "$TRAIN_SEEDS" --val-seeds "$VAL_SEEDS" --test-seeds "$TEST_SEEDS"  --c1-positive-weight "${V4492_C1_POSITIVE_WEIGHT:-1}"  --c2-positive-weights "${V4492_C2_POSITIVE_WEIGHTS:-1,2,3,5,10}"  --c2-hard-negative-count "${V4492_C2_HARD_NEGATIVE_COUNT:-8000}"  --c2-random-negative-count "${V4492_C2_RANDOM_NEGATIVE_COUNT:-8000}"  --c1-max-iter "${V4492_C1_MAX_ITER:-220}" --c2-max-iter "${V4492_C2_MAX_ITER:-180}"  --learning-rate "${V4492_LEARNING_RATE:-0.06}" --max-leaf-nodes "${V4492_MAX_LEAF_NODES:-31}"  --min-samples-leaf "${V4492_MIN_SAMPLES_LEAF:-20}" --l2 "${V4492_L2:-1.0}" 2>&1 | tee "$LOG"
echo "TRAIN_OUT=$OUT"; echo "LOG=$LOG"
