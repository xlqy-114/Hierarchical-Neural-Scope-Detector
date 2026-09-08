#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DriveVLA
source activate drivevla
BASE_SUBSET=${V59_BASE_SUBSET:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json}
SMOKE_SUBSET=${V59_SMOKE_SUBSET:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/subsets/dev_stress_100_for_v59_targeted_feature_smoke/subset_ids.json}
export BASE_SUBSET
export SMOKE_SUBSET
mkdir -p "$(dirname "$SMOKE_SUBSET")"
python - <<'PYSMOKE'
import json, os
from pathlib import Path
base=Path(os.environ['BASE_SUBSET'])
out=Path(os.environ['SMOKE_SUBSET'])
ids=json.loads(base.read_text(encoding='utf-8'))
out.write_text(json.dumps(ids[:100], indent=2), encoding='utf-8')
print('wrote', out, 'n=', len(ids[:100]))
PYSMOKE
export V59_SEEDS=${V59_SEEDS:-955901}
export V59_SUBSET_IDS="$SMOKE_SUBSET"
export V59_POSITIVE_MIN=${V59_POSITIVE_MIN:-70}
export V59_POSITIVE_MAX=${V59_POSITIVE_MAX:-80}
export V59_EXPECTED_ROWS=100
export V59_FORMAL_DIR=${V59_FORMAL_DIR:-/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260518_v59_3_targeted_feature_smoke_1seed100}
export V59_PACK_PATH=${V59_PACK_PATH:-/root/autodl-tmp/DriveVLA/pack_output_analysis/v59_3_targeted_feature_smoke_1seed100_for_return.tar.gz}
export V59_STAGE_SUMMARY_MAX_SUMMARY_CALLS=${V59_STAGE_SUMMARY_MAX_SUMMARY_CALLS:-1}
bash /root/autodl-tmp/DriveVLA/code_pack/v59_3_targeted_feature_smoke_pack/scripts/run_v59_targeted_feature_smoke_or_formal.sh
