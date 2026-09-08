# Hierarchical Neural-Scope Detector

Code, processed data, and supplementary reproducibility materials for the **Hierarchical Neural-Scope Detector for Silent Fault Detection in Vision-Language-Action-Based Autonomous Driving**.

## Repository Structure

- `01_detector_specification/`  
  Original detector experiment code, component models, exact feature specifications, training/policy definitions, and Table 5 reproducibility materials.

- `02_cross_architecture_validation/`  
  Cross-architecture validation on UniAD2, including inference/feature-collection code, exact feature contracts, fixed-split training, and held-out test results.

- `03_independent_validation/`  
  Frozen-policy evaluation on the independent 4,819-sample OpenDriveVLA set, including the exact held-out split, experiment code, frozen detector policy, and sample-level results.

- `04_ablation_and_threshold_sensitivity/`  
  Five-seed component ablation, additional 18-seed supporting analysis, and harmful, severe-harmful, and speed-alter threshold-sensitivity results.

- `05_runtime_overhead/`  
  Runtime benchmark and synchronization-profiling code, runtime contracts, and processed latency, memory, synchronization, and output-equivalence results.

Detailed reproduction notes and experiment-specific settings are provided in the README files within the corresponding directories.

## Dependencies

Python dependencies are listed in `requirements.txt`.

The experiments additionally rely on the official OpenDriveVLA and UniAD implementations, pretrained checkpoints, nuScenes data, and their associated evaluation resources. These external resources are not redistributed in this repository.
