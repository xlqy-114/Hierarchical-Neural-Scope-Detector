#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common_v4_4_6.sh"

: "${V446_SEEDS:?Set V446_SEEDS, for example 959101,959102,...}"
C2_MODES=${V446_C2_ROUTE_MODES:-recall98,quantile45}
FINITE_MODE=${V446_FINITE_BRANCH_FEATURE_MODE:-exclude_nonfinite}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${V446_OUT_DIR:-$ROOT/outputs/detector_repair_aware_custom_v4_4_6_2_$STAMP}
LOG_DIR=${V446_LOG_DIR:-$ROOT/logs}
CONTRACT=${V446_FEATURE_CONTRACT_JSON:-$(find_latest_v431_contract || true)}
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG="$LOG_DIR/v4_4_6_2_repair_aware_custom_$STAMP.log"

ARGS=(
  --data-root "$DATA_ROOT"
  --out-dir "$OUT_DIR"
  --seeds "$V446_SEEDS"
  --c2-route-modes "$C2_MODES"
  --finite-branch-feature-mode "$FINITE_MODE"
  --c2-finite-neg-weight "${V446_C2_FINITE_NEG_WEIGHT:-2.0}"
  --c2-nonfinite-neg-weight "${V446_C2_NONFINITE_NEG_WEIGHT:-2.0}"
  --c2-min-pool-rows-finite "${V446_C2_MIN_POOL_ROWS_FINITE:-200}"
  --c2-min-pool-rows-nonfinite "${V446_C2_MIN_POOL_ROWS_NONFINITE:-100}"
  --c2-min-inner-branch-positives "${V446_C2_MIN_INNER_BRANCH_POSITIVES:-3}"
  --c2-min-total-branch-positives "${V446_C2_MIN_TOTAL_BRANCH_POSITIVES:-8}"
  --random-seed "${V446_RANDOM_SEED:-773003}"
  --policy-grid-max "${V446_POLICY_GRID_MAX:-100}"
  --far-budgets "${V446_FAR_BUDGETS:-0.02}"
  --recall-targets "${V446_RECALL_TARGETS:-0.90,0.95,0.99}"
  --repair-budget-max-default "${V446_REPAIR_BUDGET_MAX_DEFAULT:-4}"
  --excluded-activation-features-file "$EXCLUDED_ACTIVATION_FEATURES_FILE"
  --expected-extended-source-feature-count "$EXPECTED_EXTENDED_SOURCE_FEATURE_COUNT"
  --expected-runtime-summary-feature-count "$EXPECTED_RUNTIME_SUMMARY_FEATURE_COUNT"
  --expected-activation-feature-count "$EXPECTED_ACTIVATION_FEATURE_COUNT"
  --save-models
)
if [[ -n "$CONTRACT" ]]; then
  ARGS+=(--feature-contract-json "$CONTRACT")
fi

python -u "$PKG_DIR/tools_uniad/train_repair_aware_branch_c1c2_v4_4_6.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"
