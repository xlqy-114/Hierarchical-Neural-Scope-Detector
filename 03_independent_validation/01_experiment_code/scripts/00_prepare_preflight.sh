#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
cd "$REPO"
source activate drivevla
mkdir -p "$ROOT"/{inputs,work,frozen_policy,subsets,output,logs} "$REPO/pack_output_analysis"
python -u "$PKG/tools/preflight_r1c1.py" --root "$ROOT" --repo-root "$REPO" 2>&1 | tee "$ROOT/logs/00_preflight.log"
echo '[PASS] R1C1 preflight complete.'
