# Existing-results evaluation

## Scope and evidence

This evaluation consumed the saved confusion matrices and aggregate metric CSVs under `analysis_outputs/`, plus the original source datasets. It did not retrain or modify any model, dataset, hyperparameter, architecture, or existing result.

The three implemented classifiers in `analyze_crop_fertilizer.py` are:

- **1D-CNN Model:** Conv1d (1→32→64, kernel size 3) with batch normalization/ReLU and adaptive pooling, followed by a 64-unit dense layer, dropout 0.3, and class logits.
- **Dual-Branch Fusion Net:** the encoded feature vector is split into a Conv1d branch (16 channels, adaptive pooling) and a 16-unit MLP branch; concatenated features feed a 32-unit fusion layer with dropout 0.2 and class logits.
- **Gradient Boosting Ensemble:** scikit-learn `GradientBoostingClassifier`, 70 estimators, max depth 4, learning rate 0.08, seed 42.

The original training script uses median imputation and standardization for numeric features, one-hot encoding with unknown categories ignored, a stratified 70/30 split (seed 42), and 25%-of-feature-standard-deviation Gaussian noise on numeric inputs. The noise is applied before the split. Neural models use 150 epochs, Adam (learning rate 0.003), batch size 32, and cross-entropy loss. The gradient-boosting implementation is not epoch based.

## Artifact limitations

- Evaluated saved confusion matrices: **9** (three methods across three target tasks).
- Within each target task, the source script evaluates all three methods on the same held-out split; saved per-class test supports were checked and match. Different target tasks use separate stratified splits.
- Raw per-row predictions, class probability scores, fitted model weights, training histories, validation histories, repeated seeds, cross-validation folds, and 0/10/20/30% noise-condition outputs are not present in the workspace.
- The source script temporarily constructs prediction/probability arrays but does not persist them. It trains the neural models without a validation set and returns no epoch history. No epoch curves, probability/ranking curves, calibration metrics, explainability plots, paired significance tests, or noise/seed/CV comparisons can therefore be derived from the saved evidence.
- The existing summary metrics are rounded to four decimal places; the consistency audit compares at that precision.
- Confusion-matrix bootstrap intervals below are stratified by actual class and are unpaired. They do not substitute for paired model comparisons.
- PCA/t-SNE, feature distributions, pair plots, and correlations are descriptive projections of source data, not held-out model performance or causal explanations.

## Evaluation summary

- Number of saved classifier/task evaluations: **9**.
- Existing aggregate metrics matched to recomputed confusion-matrix results at their saved four-decimal precision: **35/36**.
- Dataset details and exact model/preprocessing specifications are exported in `quantitative_metrics/dataset_profiles.csv` and `quantitative_metrics/methodology_profiles.csv`.
- Full metric and graph availability is documented in `quantitative_metrics/metric_availability.csv` and `quantitative_metrics/graph_availability.csv`.

## Output structure

- `methodologies/`: per-method confusion matrices, normalized confusion matrices, and class-wise metric plots.
- `quantitative_metrics/`: CSV/Excel tables, metric availability, bootstrap intervals, and source consistency checks.
- `qualitative_graphs/`: descriptive source-data feature and class visualizations.
- `comparative_visualizations/`: model comparison bar charts by target task.
- `classification_curves/`, `explainability_visualizations/`, `robustness_graphs/`, `statistical_analysis/`, `ablation_studies/`: status tables documenting which requested analyses cannot be supported by present artifacts.

Every generated figure is saved as both 300-DPI PNG and PDF.
