#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
RUN=$ROOT/output/formal_seed970001
cd "$REPO"
source activate drivevla
test -s "$RUN/results/v53_inline_detector_records.jsonl" || { echo '[ERROR] Formal online records missing.' >&2; exit 40; }
# This is the FIRST stage that reads the full-6019 clean reference / GT-derived planning metrics.
# It cannot change the frozen model, threshold, feature schema, veto, or online alarms.
python -u "$PKG/tools/evaluate_formal.py" --root "$ROOT" --repo-root "$REPO" --run-dir "$RUN" --expected-rows 4819 \
  2>&1 | tee "$ROOT/logs/05_evaluate_formal.log"
