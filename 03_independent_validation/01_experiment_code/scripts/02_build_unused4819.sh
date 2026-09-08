#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
cd "$REPO"
source activate drivevla
test -s "$ROOT/frozen_policy/FROZEN_BEFORE_TEST.txt" || { echo '[ERROR] Freeze policy first.' >&2; exit 22; }
python -u "$PKG/tools/build_unused4819.py" --root "$ROOT" --repo-root "$REPO" 2>&1 | tee "$ROOT/logs/02_build_unused4819.log"
echo '[PASS] unused4819 and smoke12 lists frozen.'
