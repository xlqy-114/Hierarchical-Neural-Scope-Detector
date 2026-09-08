#!/usr/bin/env bash
set -euo pipefail
ROOT=${UNIAD2_ROOT:-/root/autodl-tmp/UniAD2}
export V4492_NUM_THREADS=${V4492_NUM_THREADS:-8}
DATA_ROOT=${V4492_DATA_ROOT:-$ROOT/outputs/direct_fault_repair_extended_features_v4_3_1}
CONTRACT=${V4492_FROZEN_CONTRACT_JSON:-$ROOT/outputs/detector_repair_aware_full_exact_contract_19seeds_v4_4_6_2_20260619_120008/frozen_exact_feature_contract_v4_4_6_2.json}
OUT=${V4492_PREFLIGHT_OUT:-$ROOT/outputs/preflight_all_eligible_c1score_c2_v4_4_9_2_$(date +%Y%m%d_%H%M%S)}
LOG=${V4492_PREFLIGHT_LOG:-$ROOT/logs/preflight_all_eligible_c1score_c2_v4_4_9_2_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$ROOT/logs" "$(dirname "$OUT")"; cd "$ROOT"
python -u "$ROOT/code_pack/uniad2_v4_4_9_2_all_eligible_c1score_c2_weight_sweep_pack/tools_uniad/train_all_eligible_c1score_c2_weight_sweep_v4_4_9_2.py"  --data-root "$DATA_ROOT" --frozen-contract-json "$CONTRACT" --out-dir "$OUT" --preflight-only 2>&1 | tee "$LOG"
echo "PREFLIGHT_OUT=$OUT"; echo "LOG=$LOG"
