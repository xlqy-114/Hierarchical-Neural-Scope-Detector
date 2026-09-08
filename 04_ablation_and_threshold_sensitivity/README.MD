# Ablation and Threshold Sensitivity

This directory contains the component-ablation, additional 18-seed analysis, and threshold-sensitivity results.

## Directory Structure

- `01_component_ablation/`  
  Five-seed, 6,000-sample OOF component ablation for the base detector, targeted evidence, consistency signal, and conservative veto.

- `02_additional_18seed_analysis/`  
  Supporting 18-seed analysis, including the corresponding inference/training implementation and processed sample-level results.

- `03_harmful_severe_threshold_sensitivity/`  
  Harmful and severe-harmful threshold sensitivity using the same five-seed OOF protocol.

- `04_speed_alter_threshold_sensitivity/`  
  Speed-alter metric recovery, historical-definition confirmation, and speed-threshold sensitivity analysis.

## Evaluation Protocol

The component ablation and threshold-sensitivity analyses use the same 6,000 samples from five inference seeds (`955916`–`955920`) under a leave-one-seed-out OOF protocol.

The harmful L2 thresholds are:

`0.25, 0.40, 0.50, 0.60, 0.75`

The severe-harmful L2 thresholds are:

`0.75, 0.90, 1.00, 1.10, 1.25`

The speed-alter analysis varies the fault-vs-clean speed threshold and the reference-relative speed-degradation threshold around their nominal values of `0.40` and `0.20`.

The additional 18-seed analysis contains 108,342 sample-level predictions and provides supporting evidence for the conservative veto under the Recall >= 99% requirement.

## Reproducibility

Processed sample-level, seed-wise, and aggregate results required to reproduce the reported ablation and sensitivity tables are included in this directory.

The underlying five-seed detector and OOF implementation is released separately under `01_detector_specification`.