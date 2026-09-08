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
mem = pd.read_csv(r / "runtime_memory_summary.csv")
hw = json.load(open(r / "hardware_protocol_and_percentages.json", encoding="utf-8"))
eq = json.load(open(r / "output_equivalence.json", encoding="utf-8"))

def lrow(name):
    x = lat[lat["component"] == name].iloc[0]
    return {
        "mean_ms": float(x["mean"]),
        "median_ms": float(x["median"]),
        "p95_ms": float(x["p95"]),
        "std_ms": float(x["std"]),
    }

def mmib(component, metric):
    d = mem[(mem.component == component) & (mem.metric == metric)]
    return None if d.empty else float(d.iloc[0].mib)

table = []
memory_map = {
    "Hooks": (
        mmib("Hooks", "working_state_bytes"),
        "captured scalar hook state; GPU transient peak is reported separately",
    ),
    "Feature extraction": (
        mmib("Feature extraction", "representative_working_buffer_bytes"),
        "representative current-sample extraction/runtime-row working buffer",
    ),
    "Feature aggregation": (
        mmib("Feature aggregation", "representative_working_buffer_bytes"),
        "base detector inputs + compact 256-D targeted input + consistency vector",
    ),
    "Detector inference": (
        mmib("Detector inference", "resident_detector_accounted_bytes"),
        "directly accounted resident memory of the three base/veto artifacts, compact targeted model, and consistency state",
    ),
    "Total online monitor": (
        mmib("Full online monitor", "accounted_cpu_peak_working_bytes"),
        "persistent process RSS increment after loading monitor artifacts and hooks",
    ),
}
for component in [
    "Hooks", "Feature extraction", "Feature aggregation",
    "Detector inference", "Total online monitor",
]:
    x = lrow(component)
    table.append({
        "component": component,
        **x,
        "cpu_memory_mib": memory_map[component][0],
        "memory_definition": memory_map[component][1],
    })

pd.DataFrame(table).to_csv(r / "comment6_component_table.csv", index=False)

relative = lat[lat.component == "Relative latency overhead"].iloc[0]
summary = {
    "benchmark_protocol": {
        "gpu": hw["gpu"],
        "batch_size": hw["batch_size"],
        "warmup_samples": hw["warmup_samples"],
        "measured_samples": hw["measured_samples"],
        "forward_callbacks": hw["total_forward_callbacks"],
        "deployment_configuration": hw["deployment_configuration"],
    },
    "component_latency_and_memory": table,
    "overall": {
        "mean_relative_latency_overhead_percent": float(relative["mean"]),
        "median_relative_latency_overhead_percent": float(relative["median"]),
        "p95_relative_latency_overhead_percent": float(relative["p95"]),
        "persistent_cpu_memory_increase_mib": mmib(
            "Full online monitor", "persistent_cpu_rss_increment_bytes"
        ),
        "accounted_cpu_memory_increase_percent": hw[
            "accounted_cpu_memory_increase_percent"
        ],
        "incremental_gpu_peak_mean_mib": mmib(
            "Full online monitor", "paired_incremental_gpu_peak_mean_bytes"
        ),
        "incremental_gpu_peak_p95_mib": mmib(
            "Full online monitor", "paired_incremental_gpu_peak_p95_bytes"
        ),
        "incremental_gpu_peak_mean_percent": hw[
            "paired_incremental_gpu_peak_mean_percent_of_baseline_peak"
        ],
        "cpu_core_equivalent_increase_percentage_points": hw[
            "cpu_core_equivalent_increase_percentage_points"
        ],
        "cpu_logical_capacity_increase_percentage_points": hw[
            "cpu_capacity_increase_percentage_points"
        ],
    },
    "trajectory_parity": eq,
    "latency_exclusions": hw["latency_scope"],
    "deployment_percentile_optimization": {
        "p95_p99_sample_cap": 4096,
        "sampling": "deterministic evenly distributed midpoint sampling",
        "small_tensors_exact": True,
        "random_sampling": False,
    },
}
(r / "comment6_reviewer_ready_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
print("\nComment-6 component table:")
print(pd.DataFrame(table).to_string(index=False))
