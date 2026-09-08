#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/DriveVLA
EXP=$ROOT/reviewer_exp/R1C1_broader4819
CODE=$ROOT/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_6
mkdir -p "$EXP/logs"
python -u "$CODE/tools/reevaluate_historical_table5B.py" \
  --repo "$ROOT" \
  --run-dir "$EXP/output/formal_seed970001" \
  --contract-zip "$EXP/inputs/Table5_REPRODUCIBILITY_CONTRACT_EXACT_REPLAY_20260829.zip" \
  2>&1 | tee "$EXP/logs/05b_reevaluate_historical_table5B.log"
