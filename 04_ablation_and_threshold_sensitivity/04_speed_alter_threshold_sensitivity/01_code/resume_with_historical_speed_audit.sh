#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/DriveVLA
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export OMP_NUM_THREADS=1

PACK_DIR="${PACK_DIR:-/root/autodl-tmp/DriveVLA/code_pack/R1C4_speed_sensitivity_final_runpack}"
CAMPAIGN="${CAMPAIGN:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/R1C4_threshold_sensitivity_5seeds}"
SEED_ROOT="${SEED_ROOT:-A1_internal_neural_scdout_v54_2_online_singlefault}"
CLEAN_TRAJ="${CLEAN_TRAJ:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration/pred_trajs_dict.json}"
ANALYSIS="${ANALYSIS:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/R1C4_threshold_sensitivity_analysis}"
CANDIDATE_VALUES="$ANALYSIS/speed_metric_recovery/speed_metric_candidate_values.csv"
HISTDEF="$ANALYSIS/speed_metric_recovery/historical_speed_definition_confirmation.json"
SPEED_OUT="$ANALYSIS/speed_sensitivity"

if [ ! -f "$CANDIDATE_VALUES" ]; then
  echo "[ERROR] candidate values not found: $CANDIDATE_VALUES"
  echo "The fixed3 speed-metric recovery must have completed first."
  exit 1
fi

if [ ! -f "$ANALYSIS/harmful_severe/harmful_severe_threshold_sensitivity_summary.csv" ]; then
  echo "[ERROR] completed harmful/severe result is missing."
  exit 1
fi

python "$PACK_DIR/05_confirm_historical_speed_definition.py" \
  --audit-csv "$PACK_DIR/historical_v31_heading_speed_label_audit.csv" \
  --clean-trajs "$CLEAN_TRAJ" \
  --output "$HISTDEF"

python "$PACK_DIR/06_run_speed_sensitivity.py" \
  --campaign-dir "$CAMPAIGN" \
  --seed-root "$SEED_ROOT" \
  --candidate-values "$CANDIDATE_VALUES" \
  --historical-definition-json "$HISTDEF" \
  --output-dir "$SPEED_OUT" \
  --clean-grid "0.30,0.35,0.40,0.45,0.50" \
  --gt-grid "0.10,0.15,0.20,0.25,0.30" \
  --require-sklearn-1-2-2

tar -czf \
/root/autodl-tmp/DriveVLA/pack_output_analysis/R1C4_threshold_sensitivity_complete_for_return.tar.gz \
-C /root/autodl-tmp/DriveVLA \
output/OpenDriveVLA-0.5B/R1C4_threshold_sensitivity_analysis \
R1C4_threshold_sensitivity_analysis.log

echo
echo "[DONE]"
ls -lh \
/root/autodl-tmp/DriveVLA/pack_output_analysis/R1C4_threshold_sensitivity_complete_for_return.tar.gz
