#!/usr/bin/env bash
set -euo pipefail
ROOT=${UNIAD2_ROOT:-/root/autodl-tmp/UniAD2}
export V4492_NUM_THREADS=${V4492_NUM_THREADS:-1}
cd "$ROOT/code_pack/uniad2_v4_4_9_2_all_eligible_c1score_c2_weight_sweep_pack"
python -m py_compile tools_uniad/train_all_eligible_c1score_c2_weight_sweep_v4_4_9_2.py tests/test_synthetic_pipeline_v4_4_9_2.py
python -u tests/test_synthetic_pipeline_v4_4_9_2.py
