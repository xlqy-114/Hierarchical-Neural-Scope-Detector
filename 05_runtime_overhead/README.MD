# Runtime Overhead and Synchronization Profiling

This directory contains the code, runtime contract, dependency specification, and processed benchmark results used to evaluate the deployment overhead of the proposed online detector.

## Benchmark Protocol

The final benchmark uses:

- GPU: NVIDIA H20
- Batch size: 1
- Warm-up samples: 20
- Measured samples: 100
- Stage-summary hooks: 42
- Total forward callbacks: 44

The same ordered samples are used for the baseline and monitored runs.

The final benchmark configuration is defined by:

`01_benchmark_code/FINAL_BENCHMARK_POLICY.json`

## Directory Structure

- `01_benchmark_code/`  
  Runtime benchmark, synchronization profiler, and runtime-source implementation.

- `02_runtime_contract/`  
  Feature and detector contracts required by the benchmark.

- `03_runtime_dependency_specification/`  
  Runtime dependency and hook specifications.

- `04_reported_results/`  
  Sample-level benchmark data and reported latency, memory, synchronization, and output-equivalence results.

## Reported Results

The mean online-monitor latency is:

- Hooks: 796.6 ms
- Feature extraction: 9.6 ms
- Feature aggregation: 70.8 ms
- Detector inference: 85.2 ms
- Total online monitor: 962.1 ms

The mean relative latency overhead is 82.4%.

The directly accounted CPU monitor memory is approximately 11.3 MiB, and the mean incremental GPU peak is approximately 701.5 MiB.

The required synchronization overhead is approximately 227.7 ms per sample.

Trajectory-output equivalence was verified on all 120 paired baseline/monitored samples, with 120/120 exact matches.

## Dependencies

Detector artifacts and calibration files released under the earlier reproducibility modules are reused here.

OpenDriveVLA source code, pretrained checkpoints, nuScenes data, and official evaluation resources are external dependencies and are not redistributed in this repository.