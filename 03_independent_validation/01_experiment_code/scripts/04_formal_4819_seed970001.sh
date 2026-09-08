#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
SUBSET=$ROOT/subsets/reviewer_R1C1_unused4819_ids.json
RUN=$ROOT/output/formal_seed970001
if [[ -e "$RUN/plan_conv.json" || -d "$RUN/results" ]]; then
  echo "[ERROR] Formal output already exists: $RUN" >&2
  echo "Refusing to overwrite a held-out test. Inspect/resume separately rather than rerunning over it." >&2
  exit 30
fi
# Historical Table-5 runs injected 800--900 of 1200 rows per seed (66.67%--75%).
# Scale that pre-existing range to all 4819 held-out rows: ceil(4819*800/1200)=3213; floor(4819*900/1200)=3614.
bash "$PKG/scripts/run_one_inference.sh" "$SUBSET" 4819 970001 3213 3614 "$RUN"
