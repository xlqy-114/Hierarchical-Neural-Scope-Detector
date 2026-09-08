#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${V73_16_REPO_ROOT:-/root/autodl-tmp/DriveVLA}
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/drivevla:$PKG_DIR:${PYTHONPATH:-}"

: "${V73_16_MODEL_READY_CSV:?Set V73_16_MODEL_READY_CSV to v73_16_model_ready_matrix_with_labels.csv or v73_16_model_ready_plus_aux_matrix_with_labels.csv}"
OUT_DIR=${V73_16_TRAIN_OUT_DIR:-$REPO_ROOT/pack_output_analysis/v73_16_selected_feature_exact_v73_8e_training_$(date +%Y%m%d_%H%M%S)}
LABELS=${V73_16_TRAIN_LABELS:-harmful_l2_2s,severe_l2_2s}
C1_MODELS=${V73_16_C1_MODELS:-hgb,extra_trees}
C2_ROUTE_MODES=${V73_16_C2_ROUTE_MODES:-recall99,recall98,quantile45}
C2_NEG_WEIGHT=${V73_16_C2_NEG_WEIGHT:-2.0}
QUICK_ROWS=${V73_16_TRAIN_QUICK_ROWS:-0}
INCLUDE_AUX=${V73_16_TRAIN_INCLUDE_AUX:-0}
FEATURE_CONTRACT_CSV=${V73_16_FEATURE_CONTRACT_CSV:-$PKG_DIR/resources/v73_9_hgb_online_feature_contract.csv}

EXTRA_ARGS=()
if [[ "$INCLUDE_AUX" == "1" ]]; then
  EXTRA_ARGS+=(--include-aux)
fi
if [[ "$QUICK_ROWS" != "0" ]]; then
  EXTRA_ARGS+=(--quick-rows "$QUICK_ROWS")
fi

mkdir -p "$OUT_DIR"
cat <<EOF
============================================================
[v73.16 selected-feature training - exact v73.8E method]
============================================================
MODEL_READY_CSV=$V73_16_MODEL_READY_CSV
FEATURE_CONTRACT_CSV=$FEATURE_CONTRACT_CSV
OUT_DIR=$OUT_DIR
LABELS=$LABELS
C1_MODELS=$C1_MODELS
C2_ROUTE_MODES=$C2_ROUTE_MODES
C2_NEG_WEIGHT=$C2_NEG_WEIGHT
INCLUDE_AUX=$INCLUDE_AUX
QUICK_ROWS=$QUICK_ROWS
This script is not called by inference. It must be run explicitly.
Method: same as v73.8E except the feature list is the selected standalone contract.
============================================================
EOF

python -u "$PKG_DIR/tools_train/train_v73_16_selected_features_exact_v73_8e_method.py" \
  --model-ready-csv "$V73_16_MODEL_READY_CSV" \
  --feature-contract-csv "$FEATURE_CONTRACT_CSV" \
  --output-dir "$OUT_DIR" \
  --labels "$LABELS" \
  --c1-models "$C1_MODELS" \
  --c2-route-modes "$C2_ROUTE_MODES" \
  --c2-neg-weight "$C2_NEG_WEIGHT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/v73_16_selected_feature_exact_v73_8e_training.log"

# Optional split-used extraction. Disabled by default. This is post-training only and is not part of inference or normal training.
if [[ "${V73_16_RUN_SPLIT_USED_EXTRACTION:-0}" == "1" ]]; then
  python -u "$PKG_DIR/tools_train/extract_v73_16_splitused_contract_from_saved_models.py" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/v73_16_splitused_extraction.log"
fi

tar -czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
ls -lh "$OUT_DIR.tar.gz"
