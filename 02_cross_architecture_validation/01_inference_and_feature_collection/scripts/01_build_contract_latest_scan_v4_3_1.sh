#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack/scripts/_env_check_v4_3_1.sh
PKG=/root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack
SCAN_DIR=${SCAN_DIR:-}
TAG=${TAG:-latest}
if [[ -n "$SCAN_DIR" ]]; then
  python "$PKG/tools_uniad/build_feature_contract_v4_3_1.py" --scan-dir "$SCAN_DIR" --tag "$TAG"
else
  python "$PKG/tools_uniad/build_feature_contract_v4_3_1.py" --tag "$TAG"
fi
