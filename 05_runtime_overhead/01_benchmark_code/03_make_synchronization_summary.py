#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--result-dir", required=True)
args = ap.parse_args()

r = Path(args.result_dir)
lat = pd.read_csv(r / "runtime_latency_summary.csv")
hw = json.load(open(r / "hardware_protocol_and_percentages.json", encoding="utf-8"))

def row(name):
    x = lat[lat.component == name].iloc[0]
    return {
        "mean_ms_per_sample": float(x["mean"]),
        "median_ms_per_sample": float(x["median"]),
        "p95_ms_per_sample": float(x["p95"]),
        "std_ms_per_sample": float(x["std"]),
    }

summary = {
    "feature_extraction_latency": row("Feature extraction"),
    "detector_inference_latency": row("Detector inference"),
    "synchronization": {
        "feature_extraction": row("Feature-extraction synchronization"),
        "feature_aggregation": row("Feature-aggregation synchronization"),
        "stage_hook_internal": row("Stage-hook internal synchronization"),
        "hook_readout_boundary": row("Hook-readout boundary synchronization"),
        "detector_execution_required_cuda_sync": row(
            "Detector-execution required CUDA synchronization"
        ),
        "total_required_sync": row("Total required synchronization"),
    },
    "interpretation": {
        "detector_execution": (
            "Detector scoring is CPU-side after online evidence has been "
            "materialized on the host, so it requires no additional CUDA "
            "synchronization step."
        ),
        "profiling_scope": hw["synchronization_profiling"],
    },
}

(r / "synchronization_overhead_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

table = pd.DataFrame([
    {"Item": "Feature extraction latency", **row("Feature extraction")},
    {"Item": "Detector inference latency", **row("Detector inference")},
    {"Item": "Feature-extraction synchronization", **row("Feature-extraction synchronization")},
    {"Item": "Stage-hook internal synchronization", **row("Stage-hook internal synchronization")},
    {"Item": "Total required synchronization", **row("Total required synchronization")},
])
table.to_csv(r / "synchronization_overhead_table.csv", index=False)

print(json.dumps(summary, indent=2))
print("\nSynchronization table:")
print(table.to_string(index=False))
