#!/usr/bin/env bash
set -euo pipefail

ROOT=${UNIAD2_ROOT:-/root/autodl-tmp/UniAD2}
PKG_DIR=${V446_PKG_DIR:-$ROOT/code_pack/uniad2_v4_4_6_2_repair_aware_full_exact_contract_pack}
DATA_ROOT=${V446_DATA_ROOT:-$ROOT/outputs/direct_fault_repair_extended_features_v4_3_1}
ENV_PATH=${UNIAD2_ENV:-$ROOT/envs/uniad2}

cd "$ROOT"
source activate "$ENV_PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$PKG_DIR/tools_uniad:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${V446_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${V446_MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${V446_OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${V446_NUMEXPR_NUM_THREADS:-4}

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PKG_DIR/tools_uniad/train_repair_aware_branch_c1c2_v4_4_6.py" ]]; then
  echo "ERROR: package not installed at $PKG_DIR" >&2
  exit 2
fi

find_latest_v431_contract() {
  find "$ROOT/outputs/feature_contract_v4_3_1" -type f -name 'feature_contract_v4_3_1.json' 2>/dev/null \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

EXCLUDED_ACTIVATION_FEATURES_FILE=${V4462_EXCLUDED_ACTIVATION_FEATURES_FILE:-$PKG_DIR/config/excluded_unstable_activation_features_v4_4_6_2.txt}
EXPECTED_EXTENDED_SOURCE_FEATURE_COUNT=${V4462_EXPECTED_EXTENDED_SOURCE_FEATURE_COUNT:-4780}
EXPECTED_RUNTIME_SUMMARY_FEATURE_COUNT=${V4462_EXPECTED_RUNTIME_SUMMARY_FEATURE_COUNT:-669}
EXPECTED_ACTIVATION_FEATURE_COUNT=${V4462_EXPECTED_ACTIVATION_FEATURE_COUNT:-5449}
if [[ ! -f "$EXCLUDED_ACTIVATION_FEATURES_FILE" ]]; then
  echo "ERROR: exclusion file does not exist: $EXCLUDED_ACTIVATION_FEATURES_FILE" >&2
  exit 2
fi
