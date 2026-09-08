#!/usr/bin/env bash
set -euo pipefail
: "${V73_16_SEEDS:?Please set V73_16_SEEDS, e.g. export V73_16_SEEDS=957001,957002,957003}"
export V73_16_SAMPLE_COUNT=${V73_16_SAMPLE_COUNT:-6019}
export V73_16_RUN_TAG=${V73_16_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_v73_16_formal6019}
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v73_16_standalone_no_detector.sh"
