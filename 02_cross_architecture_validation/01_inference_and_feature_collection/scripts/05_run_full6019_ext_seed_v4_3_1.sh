#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack/scripts/_env_check_v4_3_1.sh
PKG=/root/autodl-tmp/UniAD2/code_pack/uniad2_v4_3_1_extended_feature_material_pack
mkdir -p /root/autodl-tmp/UniAD2/logs
SEED=${SEED:-958101}
LOG=/root/autodl-tmp/UniAD2/logs/v4_3_1_ext_full6019_seed${SEED}_$(date +%Y%m%d_%H%M%S).log
CONTRACT=${CONTRACT:-/root/autodl-tmp/UniAD2/outputs/feature_contract_v4_3_1/latest/feature_contract_v4_3_1.json}
REUSE_PLAN_DIR=${REUSE_PLAN_DIR:-/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_detector_dataset_v4_0_1/all_prefix_6019_seed${SEED}_hybrid_m10p0_all_typeaware_hi16384_lo4096_rpbudgeted_multi_K4_hcmodule_aware_r0p5_gate}
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_extended_features_v4_3_1}
echo LOG=$LOG
python "$PKG/tools_uniad/run_uniad_direct_fault_repair_extended_features_v4_3_1.py" \
  --out-root "$OUT_ROOT" --seed "$SEED" --selected-mode all_prefix --case-limit 6019 \
  --method hybrid --margin 10.0 --guard-groups all --sampling-strategy typeaware \
  --high-risk-sample-values 16384 --low-risk-sample-values 4096 \
  --repair-policy budgeted_multi --max-repair-count-per-sample 4 \
  --high-contam-strategy module_aware --high-contam-ratio 0.5 --enable-acceptance-gate \
  --feature-contract "$CONTRACT" --reuse-plan-dir "$REUSE_PLAN_DIR" --progress-every 200 2>&1 | tee "$LOG"
