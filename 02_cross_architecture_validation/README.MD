# Cross-Architecture Validation on UniAD2

This directory contains the code, configuration, feature contract, and reported results for the UniAD2 cross-architecture validation experiment, together with a supplementary analysis of internal NaN/Inf events.

## Experiment Protocol

The main experiment uses 19 full inference seeds with a fixed 15/2/2 train/validation/test split.

- Training seeds:  
  `959101, 959102, 959103, 959104, 959105, 959113, 959114, 959115, 959116, 959117, 959118, 959119, 959120, 959121, 959122`

- Validation seeds:  
  `959106, 959123`

- Test seeds:  
  `959107, 959124`

The complete campaign contains 114,361 sample-level evaluations. Model fitting is performed on the training split, while model/operating-point selection uses the validation split. The test split is not used for model or threshold selection.

## Directory Structure

- `01_inference_and_feature_collection/`  
  UniAD2 inference and runtime feature-collection code.

- `02_exact_feature_contract/`  
  Code used to construct and audit the exact UniAD2 feature contract.

- `03_fixed_split_training/`  
  Final fixed-split detector training, validation selection, and held-out test evaluation code.

- `04_reproducibility_config/`  
  Exact feature contract and configuration used by the final experiment.

- `05_reported_results/`  
  Validation-selected policies, held-out test predictions, aggregate metrics, and population summary.

- `06_naninf_outcome_audit/`  
  Original records from an earlier UniAD2 fault-injection experiment (v3.5.1, seed `958101`) supporting the internal NaN/Inf outcome analysis in Table B3.

## Reproduction Notes

The historical scripts retain their original filenames and default paths. For reproducing the reported 19-seed experiment, use the seed split listed above rather than historical script defaults.

The final training code should use:

`04_reproducibility_config/fixed_exact_feature_contract_v4_4_9_2.json`

as the frozen feature contract.

The operating point reported in the paper corresponds to:

- criterion: `best_system_recall_at_far_le_0.002`
- model variant: `c2_pw5p0`
- positive weight: `5.0`
- validation-selected threshold: `0.11698264754652367`

On the held-out test split, this policy gives:

- Samples: 12,038
- Positive cases: 1,408
- TP / FN / FP / TN: 1,386 / 22 / 21 / 10,609
- Composite system-alarm recall: 98.44%
- FAR: 0.198%
- Learned-branch recall on eligible trajectory-discrepancy cases: 59 / 81 (72.84%)

The supplementary v3.5.1 analysis in `06_naninf_outcome_audit/` is separate from the 19-seed evaluation. Among 900 effectively fault-injected samples with observed internal NaN/Inf events, 483 had non-finite planning outputs and 10 additional samples had finite outputs with maximum point-wise trajectory deviation exceeding 1.0 m relative to the fault-free output. Thus, 493 / 900 (54.78%) met the final-output abnormality criterion in Table B3.

Columns related to fault metadata, labels, or evaluation status in the released prediction files are retained for offline analysis and evaluation only. They are not deployment-time detector input features.

UniAD2 source code, pretrained checkpoints, and the required dataset are external dependencies and are not redistributed in this repository.