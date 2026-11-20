# Research Evaluation Report

## Overview
This report uses the preserved experimental outputs already generated in the project under `analysis_outputs/` and creates a separate evaluation package at `Research_Evaluation_Results/` without modifying the source implementations or their saved results.

## Contribution summary
The four preserved contribution families are:
- C1: Agronomic Relationship-Based Feature Engineering
- C2: Crop-Conditioned Fertilizer Recommendation
- C3: Gated Residual Network with Class-Balanced Learning
- C4: Uncertainty-Aware Adaptive Fusion

## Best observed results from the saved outputs
- Highest mean accuracy: C1 (0.9385)
- Highest mean macro F1: C1 (0.9456)
- Highest mean weighted F1: C1 (0.9381)

## Ablation interpretation
The ablation tables in `analysis_outputs/ablation_study_results.csv` show the full pipeline and component-removal variants. The reported ablations are reused directly; no additional synthetic experiments were created. The strongest empirical pattern in the saved results is that removing agronomic structure (C1) causes a substantial drop in performance for the fertilizer task, while the crop-conditioning and gating mechanisms improve the downstream fused model in the dataset-specific settings reported by the project.

## Important note on unavailable visual/statistical artifacts
No epoch-wise training curves were synthesized because the project does not preserve per-epoch histories for all four contributions. Likewise, paired per-sample statistical tests were not possible from the exported outputs alone. The evaluation therefore records the applicable status rather than fabricating unavailable results.

## Generated outputs
- `Research_Evaluation_Results/Contribution_1/Quantitative_Metrics/`
- `Research_Evaluation_Results/Contribution_2/Quantitative_Metrics/`
- `Research_Evaluation_Results/Contribution_3/Quantitative_Metrics/`
- `Research_Evaluation_Results/Contribution_4/Quantitative_Metrics/`
- `Research_Evaluation_Results/Comparative_Visualizations/`
- `Research_Evaluation_Results/Combined_Ablation_Analysis/`
- `Research_Evaluation_Results/Statistical_Analysis/`
- `Research_Evaluation_Results/Quantitative_Reports/`
