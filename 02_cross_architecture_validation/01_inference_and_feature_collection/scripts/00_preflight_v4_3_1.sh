#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack/scripts/_env_check_v4_3_1.sh
PKG=/root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack
python - <<'PY'
from pathlib import Path
paths=[
('/root/autodl-tmp/UniAD2/code/UniAD','UniAD repo'),
('/root/autodl-tmp/UniAD2/code/UniAD/ckpts/uniad_base_e2e.pth','checkpoint'),
('/root/autodl-tmp/UniAD2/outputs/clean_calibration_v3_5_3/seeds_958201_958204/clean_hook_thresholds_all_seeds_v3_5_3.csv','clean thresholds'),
('/root/autodl-tmp/UniAD2/outputs/uniad2_hook_registry_v3_5_1/uniad2_hook_registry_v3_5_1.csv','hook registry'),
('/root/autodl-tmp/UniAD2/outputs/feature_source_scan_v4_3_0','v4.3.0 scan root'),
]
for p,name in paths:
    print(name, p, 'OK' if Path(p).exists() else 'MISSING')
print('latest scan dirs:', len([p for p in Path('/root/autodl-tmp/UniAD2/outputs/feature_source_scan_v4_3_0').glob('*') if p.is_dir()]))
PY
python -m py_compile "$PKG/tools_uniad/build_feature_contract_v4_3_1.py"
python -m py_compile "$PKG/tools_uniad/run_uniad_direct_fault_repair_extended_features_v4_3_1.py"
python -m py_compile "$PKG/tools_uniad/audit_extended_dataset_v4_3_1.py"
echo PRELIGHT_OK
