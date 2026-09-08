#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack/scripts/_env_check_v4_3_1.sh
PKG=/root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_extended_features_v4_3_1_smoke}
python "$PKG/tools_uniad/audit_extended_dataset_v4_3_1.py" --out-root "$OUT_ROOT"
