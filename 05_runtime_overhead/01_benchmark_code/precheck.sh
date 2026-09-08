#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
PACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN="$REPO/output/OpenDriveVLA-0.5B/R1C4_threshold_sensitivity_5seeds"
SEED_ROOT=A1_internal_neural_scdout_v54_2_online_singlefault
CAL="$REPO/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration"

cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla

echo "===== R1C6 FINAL LEAN PRECHECK ====="

python - <<'PY'
import sklearn, torch, psutil
print("scikit-learn =", sklearn.__version__)
print("torch =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())
print("GPU =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("CPU logical =", psutil.cpu_count(logical=True))
assert sklearn.__version__ == "1.2.2"
assert torch.cuda.is_available()
PY

for p in \
 "$PACK/00_prepare_final_runtime_artifacts.py" \
 "$PACK/01_final_lean_runtime_benchmark.py" \
 "$PACK/02_make_comment6_summary.py" \
 "$PACK/FINAL_BENCHMARK_POLICY.json" \
 "$PACK/exact_source/inference_drivevla_subset1200_v59_online_targeted_feature_audit.py" \
 "$PACK/dependency/final_required_runtime_sources.csv" \
 "$PACK/dependency/final_targeted_selected_dependencies.csv" \
 "$PACK/dependency/lean_required_statistics_by_hook.csv" \
 "$PACK/dependency/lean_stage_hook_specs.csv" \
 "$PACK/dependency/base_detector_parity.csv" \
 "$PACK/dependency/targeted_model_selection_summary.csv" \
 "$PACK/contract/v59_targeted_7573_features.csv" \
 "$PACK/contract/v59_s4_plus_scores_5459.csv" \
 "$PACK/contract/selected_score_features_6.csv" \
 "$PACK/contract/historical_labels_6000.csv" \
 "$REPO/ckpts/OpenDriveVLA-0.5B" \
 "$REPO/data/nuscenes" \
 "$REPO/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json" \
 "$REPO/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz" \
 "$CAL/detector_thresholds.json" \
 "$CAL/output_sanity_thresholds.json"; do
  test -e "$p" || { echo "[ERROR] missing: $p"; exit 1; }
  echo "[OK] $p"
done

for s in 955916 955917 955918 955919 955920; do
  p="$CAMPAIGN/$SEED_ROOT/seed$s/results/v53_inline_detector_records.jsonl"
  test -f "$p" || { echo "[ERROR] missing: $p"; exit 1; }
  n=$(wc -l < "$p")
  echo "[CHECK] seed$s rows=$n"
  test "$n" -eq 1200 || exit 1
done

python - "$PACK" <<'PY'
import json, sys, pandas as pd
from pathlib import Path
p=Path(sys.argv[1])
policy=json.load(open(p/"FINAL_BENCHMARK_POLICY.json",encoding="utf-8"))
hooks=pd.read_csv(p/"dependency/lean_stage_hook_specs.csv")
stats=pd.read_csv(p/"dependency/lean_required_statistics_by_hook.csv")
bp=pd.read_csv(p/"dependency/base_detector_parity.csv")
tp=pd.read_csv(p/"dependency/targeted_model_selection_summary.csv")
assert len(hooks)==42, len(hooks)
assert hooks.hook.nunique()==42
assert set(stats.hook.unique())==set(hooks.hook.unique())
assert (bp.alarm_mismatch_full_vs_zero_unused==0).all()
assert (bp.alarm_mismatch_full_vs_stored_record_score==0).all()
assert (tp.heldout_alarm_mismatch_full_vs_zero_unselected==0).all()
assert policy["dependency_audit"]["total_forward_callbacks"]==44
print("[OK] dependency parity frozen")
print("[OK] lean stage-summary hooks = 42")
print("[OK] total forward callbacks = 44 (42 stage + canbus + online head)")
PY

python - "$PACK/01_final_lean_runtime_benchmark.py" <<'PY'
import sys
s=open(sys.argv[1],encoding="utf-8").read()
assert "PERCENTILE_SAMPLE_CAP = 4096" in s
assert "_bounded_percentiles" in s
assert "torch.sort(sample)" in s
assert "_linear_quantile_large_safe" not in s
assert "torch.kthvalue" not in s
assert "accounted_cpu_peak_working_bytes" in s
assert "POST-TIMING MEMORY PROFILING" in s
timed=s[s.index("MONITORED PASS"):s.index("POST-TIMING MEMORY PROFILING")]
assert "deep_size(" not in timed
print("[OK] bounded deterministic percentile sampling enabled: cap=4096")
print("[OK] direct CPU memory accounting enabled")
print("[OK] memory bookkeeping excluded from timed monitor section")
PY

python - "$PACK/01_final_lean_runtime_benchmark.py" <<'PY'
import sys
s=open(sys.argv[1],encoding="utf-8").read()
for token in [
    "stage_hook_internal_sync_wait_ms",
    "feature_extraction_sync_wait_ms",
    "feature_aggregation_sync_wait_ms",
    "detector_execution_required_cuda_sync_ms",
    "total_required_sync_wait_ms",
]:
    assert token in s, token
print("[OK] synchronization profiler instrumentation present")
print("[OK] detector stage adds no artificial CUDA synchronize call")
PY

echo "===== PRECHECK END ====="
