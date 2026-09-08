#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/DriveVLA
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source activate drivevla

# Hard isolation: never inherit old v50/v51/v52/v53 settings accidentally.
unset FORMAL_DIR V50_SEEDS V50_SUBSET_IDS V50_POSITIVE_MIN V50_POSITIVE_MAX V50_EXPECTED_ROWS V51_SEEDS V51_SUBSET_IDS V51_POSITIVE_MIN V51_POSITIVE_MAX V51_EXPECTED_ROWS V52_SEEDS V52_SUBSET_IDS V52_POSITIVE_MIN V52_POSITIVE_MAX V52_EXPECTED_ROWS V52_FORMAL_DIR V53_FORMAL_DIR V54_FORMAL_DIR

BASE_SUBSET=/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json
SMOKE_SUBSET=/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_100_for_v54_online_record_audit/subset_ids.json
mkdir -p "$(dirname "$SMOKE_SUBSET")"
python - <<'PY'
import json
from pathlib import Path
base = Path('/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json')
out = Path('/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_100_for_v54_online_record_audit/subset_ids.json')
ids = json.loads(base.read_text(encoding='utf-8'))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ids[:100], indent=2), encoding='utf-8')
print(f'[v54 smoke subset] wrote {out} rows={len(ids[:100])}')
PY

export V54_SEEDS=954001
export V54_SUBSET_IDS="$SMOKE_SUBSET"
export V54_POSITIVE_MIN=70
export V54_POSITIVE_MAX=80
export V54_EXPECTED_ROWS=100
export V54_FORMAL_DIR=${V54_SMOKE_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v54_sample_inline_online_v35_1_scdout_record_audit_smoke_1seed100}
export MASTER_PORT=${MASTER_PORT:-29654}
# Smoke keeps full legacy stage-summary content because exact v35.1 feature coverage and record inventory must be tested.
export V54_STAGE_SUMMARY_MAX_SUMMARY_CALLS=-1
export V54_PACK_PATH=/root/autodl-tmp/DriveVLA/pack_output_analysis/v54_sample_inline_online_v35_1_scdout_record_audit_smoke_1seed100_for_return.tar.gz

bash "$PKG_DIR/scripts/run_v54_online_detector_smoke_or_formal.sh"
