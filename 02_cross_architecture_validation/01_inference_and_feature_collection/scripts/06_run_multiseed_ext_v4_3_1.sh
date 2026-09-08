#!/usr/bin/env bash
set -euo pipefail
SEEDS=${SEEDS:-"958101 958102 958103 958104 958105 958106 958107 958108"}
PKG=/root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack
for S in $SEEDS; do
  echo "===== v4.3.1 EXT FULL SEED=$S ====="
  SEED=$S bash "$PKG/scripts/05_run_full6019_ext_seed_v4_3_1.sh"
  OUT_ROOT=/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_extended_features_v4_3_1 bash "$PKG/scripts/03_audit_latest_v4_3_1.sh"
done
