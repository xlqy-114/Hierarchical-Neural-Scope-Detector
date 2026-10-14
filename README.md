# Hierarchical Neural-Scope Detector

This repository contains the experiment code, processed data, detector artifacts, feature specifications, and supporting results for **Hierarchical Neural-Scope Detector for Silent Fault Detection in Vision-Language-Action-Based Autonomous Driving**.

The materials cover detector development and evaluation on OpenDriveVLA, cross-architecture validation on UniAD2, independent validation, component ablation, threshold sensitivity, and runtime overhead measurements.

## Repository Contents

### 01. Detector Specification (`01_detector_specification/`)

Contains the original OpenDriveVLA detector implementation, detector models, feature definitions, training configurations, and reproducibility records.

- **`01_original_experiment_code/`** — Scripts for fault-injected OpenDriveVLA inference, online feature collection, detector evaluation, and consistency/veto analysis.
- **`02_component_models/`** — Serialized detector models, model metadata, decision thresholds, and exact input-feature specifications.
- **`03_feature_specification/`** — Ordered feature lists, feature-group summaries, numerical transformation counts, and feature-audit materials.
- **`04_training_and_policy_specification/`** — Detector training and policy-analysis code, consistency settings, and reported operating-point data.
- **`05_reproducibility_verification/`** — Experiment manifests, feature-audit summaries, and comparisons between replayed and reported results.

### 02. Cross-Architecture Validation (`02_cross_architecture_validation/`)

Contains the UniAD2 validation implementation and its reported evaluation data.

- **`01_inference_and_feature_collection/`** — UniAD2 fault-injection inference and runtime feature-collection scripts.
- **`02_exact_feature_contract/`** — Feature-contract construction, model-input processing, and repair-telemetry utilities.
- **`03_fixed_split_training/`** — Detector training, validation-based policy selection, and held-out evaluation code.
- **`04_reproducibility_config/`** — Fixed feature contracts, dataset-split settings, and experiment configurations.
- **`05_reported_results/`** — Held-out sample predictions, evaluation metrics, selected policies, and population summaries.
- **`06_naninf_outcome_audit/`** — Fault-injection records and summaries for the internal NaN/Inf outcome analysis.

### 03. Independent Validation (`03_independent_validation/`)

Contains the frozen-policy validation on the independent OpenDriveVLA held-out set.

- **`01_experiment_code/`** — Scripts for constructing the held-out split, freezing the detector policy, running fault-injected inference, and evaluating the results.
- **`02_reproducibility_inputs/`** — Development inputs, detector artifacts, and reproducibility-contract materials.
- **`03_inference_dependencies/`** — Calibration files and fault-target registry data used by the inference pipeline.
- **`04_frozen_policy/`** — The frozen detector model and its feature specification.
- **`05_heldout_split/`** — Held-out sample identifiers and split-audit data.
- **`06_reported_results/`** — Sample-level predictions, aggregate metrics, and results grouped by fault operator.

### 04. Ablation and Threshold Sensitivity (`04_ablation_and_threshold_sensitivity/`)

Contains the component-ablation, statistical-support, and threshold-sensitivity datasets and analysis code.

- **`01_component_ablation/`** — Component-level prediction data, ablation summaries, selected policies, and associated analysis code.
- **`02_additional_18seed_analysis/`** — Code and processed sample-level results for the additional 18-seed evaluation.
- **`03_harmful_severe_threshold_sensitivity/`** — Continuous label metrics, OOF predictions, seed-wise summaries, and aggregate results under different harmful and severely harmful thresholds.
- **`04_speed_alter_threshold_sensitivity/`** — Speed-metric reconstruction and verification code, continuous metrics, OOF predictions, and speed-threshold sensitivity results.

### 05. Runtime Overhead (`05_runtime_overhead/`)

Contains the runtime instrumentation, benchmark implementation, dependency specifications, and measured overhead data.

- **`01_benchmark_code/`** — Runtime benchmark, feature-collection and synchronization profiling, and result-summary scripts.
- **`02_runtime_contract/`** — Feature lists and detector-input contracts used in runtime measurements.
- **`03_runtime_dependency_specification/`** — Hook specifications, required feature dependencies, and runtime-source audits.
- **`04_reported_results/`** — Sample-level timing records, component latency, end-to-end latency, memory usage, synchronization measurements, and output-equivalence results.

## Correspondence to the Manuscript and Supplementary Information

| Reported material | Repository directories |
| --- | --- |
| Main manuscript: detector configurations and Table 5 | `01_detector_specification/` |
| Main manuscript: Appendix A (UniAD2 validation) | `02_cross_architecture_validation/` |
| Main manuscript: Appendix B (independent OpenDriveVLA validation) | `03_independent_validation/` |
| Main manuscript: Appendix C (controlled component ablation) | `04_ablation_and_threshold_sensitivity/01_component_ablation/` |
| Supplementary Information: Note 1 (detector specifications) | `01_detector_specification/` |
| Supplementary Information: Note 2 (extended statistical analysis) | `04_ablation_and_threshold_sensitivity/01_component_ablation/`, `04_ablation_and_threshold_sensitivity/02_additional_18seed_analysis/`, and `04_ablation_and_threshold_sensitivity/04_speed_alter_threshold_sensitivity/` |
| Supplementary Information: Note 3 (operating-point comparison and threshold sensitivity) | `01_detector_specification/04_training_and_policy_specification/` and `04_ablation_and_threshold_sensitivity/` |
| Supplementary Information: Note 4 (runtime overhead) | `05_runtime_overhead/` |

## File Formats and Data Sources

- **CSV** — Feature lists, sample-level predictions, continuous evaluation quantities, seed-wise metrics, aggregate results, and runtime measurements.
- **JSON / JSONL** — Experiment configurations, sample identifiers, feature and policy contracts, fault records, and evaluation summaries.
- **Joblib** — Serialized detector models.
- **ZIP / TAR.GZ** — Packaged reproducibility inputs and detector artifacts.
- **Python / Shell** — Inference, data processing, detector training, evaluation, audit, and profiling scripts.

The experiments use OpenDriveVLA, UniAD2, and the nuScenes dataset. The repository provides experiment-specific code and processed materials; the official model implementations, pretrained checkpoints, original dataset files, and official evaluation resources are obtained from their respective sources.

Python dependencies for the released analysis and evaluation utilities are listed in [`requirements.txt`](requirements.txt). Additional experiment-specific information is provided in the README files and configuration records within the corresponding directories.
