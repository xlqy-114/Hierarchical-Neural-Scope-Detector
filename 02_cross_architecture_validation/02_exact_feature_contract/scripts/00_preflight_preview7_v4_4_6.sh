#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common_v4_4_6.sh"

SEEDS=${V446_SEEDS:-959101,959102,959103,959104,959105,959106,959107}
OUT_DIR=${V446_PREFLIGHT_OUT_DIR:-$ROOT/outputs/detector_repair_aware_preview7_v4_4_6_2_preflight}
CONTRACT=${V446_FEATURE_CONTRACT_JSON:-$(find_latest_v431_contract || true)}
mkdir -p "$OUT_DIR"

ARGS=(
  --data-root "$DATA_ROOT"
  --out-dir "$OUT_DIR"
  --seeds "$SEEDS"
  --finite-branch-feature-mode "${V446_FINITE_BRANCH_FEATURE_MODE:-exclude_nonfinite}"
  --repair-budget-max-default "${V446_REPAIR_BUDGET_MAX_DEFAULT:-4}"
  --excluded-activation-features-file "$EXCLUDED_ACTIVATION_FEATURES_FILE"
  --expected-extended-source-feature-count "$EXPECTED_EXTENDED_SOURCE_FEATURE_COUNT"
  --expected-runtime-summary-feature-count "$EXPECTED_RUNTIME_SUMMARY_FEATURE_COUNT"
  --expected-activation-feature-count "$EXPECTED_ACTIVATION_FEATURE_COUNT"
  --preflight-only
)
if [[ -n "$CONTRACT" ]]; then
  ARGS+=(--feature-contract-json "$CONTRACT")
fi

python -u "$PKG_DIR/tools_uniad/train_repair_aware_branch_c1c2_v4_4_6.py" "${ARGS[@]}"
echo "PREVIEW7_REPAIR_AWARE_EXACT_CONTRACT_C1C2_PREFLIGHT_OK"
