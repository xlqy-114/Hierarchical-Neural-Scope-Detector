#!/usr/bin/env bash
set -euo pipefail
REPO=/root/autodl-tmp/DriveVLA
ROOT=$REPO/reviewer_exp/R1C1_broader4819
PKG=$REPO/code_pack/R1C1_Table5_FrozenPolicy_Broader4819_v1_5
ENV=$ROOT/sklearn122_env
ART=$ROOT/inputs/v35_1_exported_selected_detectors_for_reuse.tar.gz
DETS='lineA_old_extra_trees_high_recall_online_all_no_temporal,lineA_old_hgb_online_all_no_temporal,lineA_speed_alter_hgb_online_compact'

cd "$REPO"
source activate drivevla

echo "[compat] Main DriveVLA sklearn: $(python - <<'PY'
import sklearn; print(sklearn.__version__)
PY
)"

test -s "$ART" || { echo "[ERROR] missing v35.1 artifact: $ART" >&2; exit 10; }

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "[compat] Creating isolated sklearn-1.2.2 environment at $ENV"
  python -m venv "$ENV"
fi

# Do not modify the DriveVLA environment.  The old v35.1 joblib artifacts were
# serialized by scikit-learn 1.2.2, while the current DriveVLA environment uses 1.7.2.
# Pin a small isolated compatibility environment for those three historical models only.
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install \
  'numpy==1.24.4' \
  'scipy==1.10.1' \
  'joblib==1.2.0' \
  'threadpoolctl==3.1.0' \
  'scikit-learn==1.2.2'

echo "[compat] Verifying isolated environment and exact historical artifacts..."
"$ENV/bin/python" "$PKG/tools/v35_sklearn122_server.py" \
  --artifact "$ART" --detectors "$DETS" --self-test \
  | tee "$ROOT/v35_sklearn122_selftest.json"

"$ENV/bin/python" - <<'PY'
import sklearn, numpy, scipy, joblib
assert sklearn.__version__ == '1.2.2', sklearn.__version__
print('[PASS] isolated versions:', 'sklearn='+sklearn.__version__, 'numpy='+numpy.__version__, 'scipy='+scipy.__version__, 'joblib='+joblib.__version__)
PY

echo '[PASS] v35.1 compatibility environment ready. Main DriveVLA environment was not modified.'
