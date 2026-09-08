# Independent Validation on the 4,819-Sample Held-Out Set

This directory contains the code, frozen policy, experiment configuration, held-out split, and reported results for the independent validation of the Table 5 severe-harmful detector.

## Experiment Protocol

The original development subset contains 1,200 OpenDriveVLA samples. The independent validation uses the remaining 4,819 samples from the official 6,019-sample set.

- Development samples: 1,200
- Held-out validation samples: 4,819
- Fault seed: `970001`
- Target: `severe_harmful`
- No held-out sample is used for detector fitting or operating-point selection.

The detector policy and required component models are frozen before evaluation on the 4,819-sample set.

## Directory Structure

- `01_experiment_code/`  
  Code for policy freezing, held-out split construction, fault-injected inference, and evaluation.

- `02_reproducibility_inputs/`  
  Table 5 reproducibility contract, development inputs, detector artifacts, and the original 1,200-sample ID list.

- `03_inference_dependencies/`  
  Calibration files and target-registry artifacts required by the formal inference pipeline.

- `04_frozen_policy/`  
  Frozen Table 5 severe-harmful detector model and exact feature specification.

- `05_heldout_split/`  
  Exact 4,819 held-out sample IDs and split audit.

- `06_reported_results/`  
  Sample-level predictions, aggregate metrics, policy manifest, and fault-operator-stratified results.

## Reported Operating Point

The reported result uses the historical Table 5 severe-harmful operating point:

- Feature-model threshold: `0.2055820757066283`
- Final policy: `(base alarm OR feature-model alarm) AND NOT veto`

The canonical v1.6 evaluation fields are the columns prefixed with:

`historical_table5B_`

These fields correspond to the results reported in the paper and supplementary material.

## Held-Out Results

On the 4,819-sample independent validation set:

- Severe-harmful cases: 142
- TP / FN / FP / TN: 140 / 2 / 17 / 4,660
- Recall: 98.59%
- FAR: 0.363%

The released sample-level results can be used to independently recompute these aggregate metrics and the reported fault-operator-stratified results.

## External Dependencies

OpenDriveVLA source code, pretrained checkpoints, nuScenes data, and official evaluation resources are external dependencies and are not redistributed in this repository.