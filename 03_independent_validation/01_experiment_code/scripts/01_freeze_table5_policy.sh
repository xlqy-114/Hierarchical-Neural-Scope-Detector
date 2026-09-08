#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
cd "$REPO"
source activate drivevla
mkdir -p "$ROOT/logs"
if [[ -s "$ROOT/frozen_policy/FROZEN_BEFORE_TEST.txt" && -s "$ROOT/frozen_policy/frozen_policy.json" ]]; then
  echo '[PASS] Existing frozen policy detected and preserved. Refusing to refit/overwrite.'
  cat "$ROOT/frozen_policy/frozen_policy.json"
  exit 0
fi
python -u "$PKG/tools/freeze_table5_policy.py" --root "$ROOT" 2>&1 | tee "$ROOT/logs/01_freeze_table5_policy.log"
echo '[PASS] Table-5 policy frozen. Do not modify frozen_policy after this point.'
