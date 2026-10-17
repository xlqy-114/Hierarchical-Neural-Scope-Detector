# Replication Guide

**Manuscript:** Hierarchical Neural-Scope Detector for Silent Fault Detection in Vision-Language-Action-Based Autonomous Driving  
**Journal:** Communications in Transportation Research (COMMTR)  
**Manuscript ID:** COMMTR-2026-0138
**Public code and data repository:** https://github.com/xlqy-114/Hierarchical-Neural-Scope-Detector

## 1. Scope of the replication package

This package contains the original experiment scripts, processed datasets, sample-level predictions, detector artifacts, feature specifications, evaluation configurations, and aggregate results associated with the manuscript and its Supplementary Information. It supports two levels of replication:

1. **Verification from released results:** use saved sample-level records to recalculate the reported detection metrics, component comparisons, threshold-sensitivity results, and runtime summaries.
2. **Re-execution of model experiments:** use the released inference, feature-collection, training, and evaluation scripts with the OpenDriveVLA/UniAD2 implementations, model checkpoints, nuScenes data, and corresponding experiment configurations.

The package includes processed experiment outputs, selected reproducibility inputs, and model artifacts. The nuScenes dataset and official OpenDriveVLA/UniAD2 implementations and pretrained checkpoints are available from their respective projects.

## 2. Directory structure and reported results

Paths below are relative to the repository root (the folder containing this guide and `requirements.txt`).

| Repository directory | Materials provided | Main manuscript / Supplementary Information |
| --- | --- | --- |
| `01_detector_specification/` | Original OpenDriveVLA detector scripts, serialized component models, ordered feature lists, training/policy settings, and OOF result-replay records | Main manuscript Table 5; Supplementary Information Note 1 (Tables S1–S3); operating-point data for Note 3 |
| `02_cross_architecture_validation/` | UniAD2 inference and feature-collection code, fixed feature contract, train/validation/test configuration, predictions, metrics, and NaN/Inf audit records | Main manuscript Appendix A (Tables A1–A3) |
| `03_independent_validation/` | Independent OpenDriveVLA inference/evaluation code, frozen detector policy, exact held-out sample identifiers, predictions, and fault-operator results | Main manuscript Appendix B (Tables B1–B2) |
| `04_ablation_and_threshold_sensitivity/` | Component-ablation data and code, additional 18-seed OOF analysis, harmful/severe threshold sensitivity, and speed-alteration threshold sensitivity | Main manuscript Appendix C (Table C1); Supplementary Information Notes 2–3 (Tables S4, S6, S7) |
| `05_runtime_overhead/` | Benchmark scripts, feature/runtime contracts, runtime samples, component latency, memory and synchronization summaries | Supplementary Information Note 4 (Tables S8–S9) |

The root `README.md` and the module-level README files describe the contents of these directories in more detail.

## 3. Software and input data

### 3.1 Checking saved results

Python 3.10 is recommended. The root `requirements.txt` lists the released analysis dependencies, including NumPy, pandas, SciPy, scikit-learn 1.2.2, joblib, and psutil, as well as PyTorch 2.1.2 for runtime-related utilities.

For an environment that will run the released Python utilities, install the dependencies from the **repository root**:

```bash
python -m pip install -r requirements.txt
```

The published CSV, JSON, and JSONL files can be inspected directly. Most aggregate tables can be verified using pandas.

### 3.2 Re-running inference or training

Full model-level replication additionally requires:

- The official **OpenDriveVLA** or **UniAD2** model implementation and the appropriate pretrained checkpoints.
- The **nuScenes** dataset and corresponding annotations/evaluation resources.
- Compatible PyTorch, CUDA, and GPU software environments for the selected model.
- Experiment-specific input directories and configurations described by the scripts, manifest/configuration files, and README files within `01`–`05`.

The OpenDriveVLA and UniAD2 inference environments follow their respective model implementations. The root `requirements.txt` specifies the dependencies used by the released analysis utilities.

### 3.3 Dataset and output formats

- `.csv`: ordered feature lists, sample-level predictions, labels for retrospective evaluation, held-out metrics, ablation results, threshold sweeps, and runtime measurements.
- `.json` / `.jsonl`: sample identifiers, inference/audit records, model and feature contracts, experiment configurations, and summary statistics.
- `.joblib`: exported scikit-learn detector models.
- `.zip` / `.tar.gz`: archived inputs and detector reproducibility artifacts.
- `.py` / `.sh`: inference, fault-injection, feature collection, analysis, training, and benchmark code.

Evaluation labels, clean-reference quantities, ground-truth outcomes, and fault metadata in the published analysis datasets support offline training, auditing, and evaluation. The deployed detector uses the runtime feature specifications documented in the package.

## 4. Verification of reported results

For the saved confusion-matrix counts, the reported detection rates are calculated as

- **Recall (%)** = `100 × TP / (TP + FN)`.
- **False-alarm rate, FAR (%)** = `100 × FP / (FP + TN)`.

Some CSV files store `recall` and `far` as fractions between 0 and 1; others use percentage-valued columns (for example, `recall_percent` and `far_percent`). Apply the corresponding percentage conversion when checking the manuscript tables. Confidence intervals reported in Supplementary Information Table S4 use pooled 95% Wilson score intervals.

### 4.1 Main manuscript: Table 5 and Supplementary Information: Note 1

| Purpose | Source file |
| --- | --- |
| Detector operating-point results for the three evaluation targets | `01_detector_specification/04_training_and_policy_specification/table5_reported_operating_points.csv` |
| Five-seed feature/model/consistency evaluations | `01_detector_specification/04_training_and_policy_specification/v61_feature_model_oof_results_by_scope.csv` |
| Feature schemas and feature-pool dimensions (Table S1) | `01_detector_specification/03_feature_specification/v59_targeted_7573_features.csv`, `old_exact_2605_features.csv`, and `v59_transform_counts.json` in the same directory |
| Consistency thresholds and training/policy settings (Table S2) | `01_detector_specification/04_training_and_policy_specification/v61_consistency_thresholds.csv` and `v61_summary.json`; model and experiment code in `01` |
| Offline target definitions (Table S3) | Supplementary Information Note 1 and the corresponding label-construction/evaluation code |
| Historical Table 5 result replay | `01_detector_specification/05_reproducibility_verification/table5_replay_vs_historical.csv` and `table5_exact_replay_summary.json` |

`table5_exact_replay_summary.json` reports 54 matching historical result rows and zero replay mismatches from the saved OOF signals and operating-point results.

### 4.2 Main manuscript: Appendix A — UniAD2 cross-architecture validation

The final held-out split consists of 12,038 samples. Use:

- `02_cross_architecture_validation/05_reported_results/fixed_split_test_metrics_v4_4_9_2.csv` — the validation-selected held-out metrics for Table A1 and learned-branch results in Table A2.
- `02_cross_architecture_validation/05_reported_results/population_summary_v4_4_9_2.csv` — counts for internal non-finite cases, deterministic validity alarms, learned-eligible cases, and positives.
- `02_cross_architecture_validation/05_reported_results/fixed_split_test_predictions_v4_4_9_2.csv` — per-sample held-out evaluation data.
- `02_cross_architecture_validation/04_reproducibility_config/` — the fixed feature contract and train/validation/test configuration.
- `02_cross_architecture_validation/06_naninf_outcome_audit/` — the separate earlier UniAD2 fault-injection audit underpinning Table A3, including 6,019-sample JSONL records and analysis summaries.

For Tables A1–A2, select the metrics row with `criterion = best_system_recall_at_far_le_0.002`, `selected_model_variant = c2_pw5p0`, and `selected_positive_weight = 5.0`. The reported held-out system counts are TP/FN/FP/TN = **1,386 / 22 / 21 / 10,609**, and the learned branch detects **59 of 81** eligible positive cases.

Table A3 uses the earlier v3.5.1 UniAD2 fault-injection audit, covering 6,019 samples. Tables A1–A2 use the final 19-seed experiment and its 12,038-sample held-out test set.

### 4.3 Main manuscript: Appendix B — independent OpenDriveVLA validation

Use the files in `03_independent_validation/`:

- `05_heldout_split/reviewer_R1C1_unused4819_ids.json` — the exact 4,819 held-out sample IDs.
- `04_frozen_policy/` — the frozen model and its feature specification.
- `06_reported_results/r1c1_formal_metrics_historical_table5B.csv` — aggregate results for Table B1.
- `06_reported_results/r1c1_formal_per_sample_historical_table5B.csv` — individual predictions for recalculation of Table B1.
- `06_reported_results/fault_operator_stratified_metrics_historical_table5B.csv` — operator-wise results for Table B2.
- `06_reported_results/historical_table5B_policy_manifest.json` — the evaluated historical Table 5 policy and threshold.

The reported severe-harmful counts are TP/FN/FP/TN = **140 / 2 / 17 / 4,660**, corresponding to 98.59% Recall and 0.363% FAR. The detector policy and operating thresholds were frozen before the independent evaluation.

### 4.4 Main manuscript: Appendix C — controlled component ablation

The final Table C1 is calculated from the same 6,000 OOF sample records in:

`04_ablation_and_threshold_sensitivity/01_component_ablation/component_ablation_sample_level.csv`

Run this **verified** command from the repository root:

```bash
python 04_ablation_and_threshold_sensitivity/01_component_ablation/table_d2.py
```

The script outputs all five rows of the component ablation and the strictly paired veto ON/OFF comparison. Its historical filename and printed heading use **“Table D2”**, corresponding to **Table C1** in the current manuscript. The third and fourth rows show the paired veto OFF/ON settings. The fifth row reports a separately selected recall-matched policy.

Additional summaries and policy definitions are in `component_ablation_policy.json`, `recovery_stage_component_ablation.csv`, and `recall_matched_veto_comparison.csv` in the same directory.

### 4.5 Supplementary Information: Note 2 — five-seed statistics and 18-seed analysis

**Table S4** combines saved results from two experimental outputs:

- Harmful and severely harmful: `04_ablation_and_threshold_sensitivity/01_component_ablation/component_ablation_sample_level.csv`; use each target's `<target>__recall_matched_veto_prediction` against its corresponding binary evaluation label, grouped by `seed`.
- Speed alteration: `04_ablation_and_threshold_sensitivity/04_speed_alter_threshold_sensitivity/03_reported_results/speed_threshold_sensitivity_seedwise.csv`; select `varied_parameter = fault_vs_clean_speed_threshold`, `speed_fault_vs_clean_threshold = 0.40`, and `speed_gt_degradation_threshold = 0.20`.

Calculate TP/FN/FP/TN separately for seeds `955916`–`955920`; pool the counts across all five seeds for the final row. The published Table S4 pooled operating points are **98.56% / 0.138%** (harmful), **99.49% / 0.310%** (severely harmful), and **98.09% / 0.121%** (speed alteration), reported as Recall / FAR.

For Table S4, the speed-alteration column corresponds to the nominal-threshold speed-sensitivity results specified above. The component-ablation directory also contains results for a separately selected speed-alteration veto policy.

For the **18-seed** supporting evaluation, use:

- `04_ablation_and_threshold_sensitivity/02_additional_18seed_analysis/02_reported_results/veto_analysis_sample_level.csv` — sample-level records.
- `.../recall_constrained_99_veto_comparison.csv` — reported comparison subject to the Recall ≥ 99% requirement.
- `.../additional_veto_analysis_summary.json` — aggregate outcomes for 18 seeds / 108,342 sample-level records.
- `.../sample_level_data_dictionary.json` — column meanings.

The `target_label` column supplies the binary evaluation outcomes used to calculate retrospective metrics.

### 4.6 Supplementary Information: Note 3 — operating points and threshold sensitivity

| Table | Source file |
| --- | --- |
| **S5**, representative operating-point comparison | `01_detector_specification/04_training_and_policy_specification/table5_reported_operating_points.csv` |
| **S6**, harmful/severely harmful L2 threshold sensitivity | `04_ablation_and_threshold_sensitivity/03_harmful_severe_threshold_sensitivity/harmful_severe_threshold_sensitivity_summary.csv` |
| **S7**, speed-alteration threshold sensitivity | `04_ablation_and_threshold_sensitivity/04_speed_alter_threshold_sensitivity/03_reported_results/speed_threshold_sensitivity_summary.csv` |

Seed-wise outputs and OOF prediction tables are also available in the respective threshold-sensitivity directories. The CSV columns identify the threshold settings and evaluation policies for each result. These analyses vary the offline evaluation definitions.

### 4.7 Supplementary Information: Note 4 — computational overhead

Use `05_runtime_overhead/04_reported_results/`:

- `comment6_component_table.csv` — per-component monitoring latency and accounted CPU memory, **Table S8**.
- `runtime_latency_summary.csv` — baseline/monitored latency, relative overhead, and synchronization statistics, **Table S9** and related text.
- `runtime_memory_summary.csv` — detailed CPU and GPU memory quantities.
- `synchronization_overhead_table.csv` and `synchronization_overhead_summary.json` — measured synchronization contributions.
- `runtime_sample_level.csv` — recorded per-sample runtime data.
- `hardware_protocol_and_percentages.json` and `output_equivalence.json` — benchmark configuration and trajectory-output parity records.

The following **verified** commands rebuild the runtime summary reports from the released measurements. Run them from the repository root:

```bash
python 05_runtime_overhead/01_benchmark_code/02_make_comment6_summary.py \
  --result-dir 05_runtime_overhead/04_reported_results

python 05_runtime_overhead/01_benchmark_code/03_make_synchronization_summary.py \
  --result-dir 05_runtime_overhead/04_reported_results
```

These commands aggregate the recorded measurements. The GPU benchmark implementation is in `05_runtime_overhead/01_benchmark_code/`, and its settings are captured in `FINAL_BENCHMARK_POLICY.json`.

## 5. Additional replication resources

The result tables, sample-level records, frozen detector policies, and experiment configuration files provide the materials for reproducing the reported numerical summaries. The module-specific configuration files identify the model settings, fault-injection seeds, data splits, feature contracts, and evaluation policies.

The large JSONL files in `02_cross_architecture_validation/06_naninf_outcome_audit/` are stored using **Git LFS**. Download the repository with LFS objects included in the archive, or retrieve the corresponding files using `git lfs pull`.

Further descriptions of the experimental materials are available in the root `README.md` and the README files in directories `02`–`05`.
