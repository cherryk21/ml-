# Analysis of `analysis_outputs`

## Executive summary

This report interprets the cleaned datasets, data-quality summaries, class distributions,
outlier screens, model metrics, per-class results, and confusion matrices in this folder.
It uses the **current/consistent** results: `model_metrics.csv`, `three_papers_realistic_results.csv`,
`three_papers_comparison.csv`, `metrics/*_metrics.csv`, and the confusion matrices whose names
end in `_confusion_matrix.csv`.

| Finding | Result |
|---|---:|
| Total cleaned records across both datasets | 6,713 |
| Missing values in cleaned datasets | 0 |
| Exact duplicate rows in cleaned datasets | 0 |
| Best Dataset 1 crop model | 1D-CNN: 96.09% accuracy, 92.23% macro-F1 |
| Best Dataset 1 fertilizer model | 1D-CNN: 50.74% accuracy, 40.45% macro-F1 |
| Best Dataset 2 model in this split | Dual-Branch: 90.61% accuracy, 90.59% macro-F1 |
| Dataset 1 fertilizer conflicts | 3 repeated input profiles, 6 records total |

**Main conclusion:** Crop recognition is strong in Dataset 1 and crop recommendation is
reasonably strong in Dataset 2 on this fixed test split. Fertilizer prediction is substantially
harder, with severe class imbalance, rare labels, repeated inputs carrying inconsistent labels,
and many cross-class errors. Dataset cleanliness does not by itself establish label reliability
or suitability for deployment.

**Important export warning:** `three_papers_results.csv` and its workbook, along with some
unsuffixed metric files, contain older results that materially disagree with the current
confusion matrices. Do not combine the two result versions. Details are in
[Result-file consistency](#result-file-consistency).

## 1. Scope and sources

The folder contains two cleaned datasets, data-quality and class-distribution summaries, model
score and ranking exports, per-model metric files, confusion matrices (CSV and PNG), and
“three papers” result/comparison workbooks. The analysis below recomputes or cross-checks results
against the CSV confusion matrices where possible.

Primary inputs:

- `Dataset_1_Crop_Fertilizer_cleaned.csv`
- `Dataset_2_Crop_Recommendation_cleaned.csv`
- `data_quality_summary.csv`
- `*_class_distribution.csv`
- `*_outliers.csv` and `*_conflicts.csv`
- `model_metrics.csv`, `model_ranking_by_macro_f1.csv`, and `quantitative_analysis_summary.csv`
- `metrics/*_metrics.csv`
- `confusion_matrices/*_confusion_matrix.csv`

### Metric definitions

- **Accuracy:** fraction of all test predictions that are correct.
- **Macro precision / recall / F1:** arithmetic mean of the per-class values; each class has
  equal weight, regardless of how common it is.
- **Weighted F1:** per-class F1 averaged using actual class support as weights. Recomputed from
  the current confusion matrices.
- **Test support:** number of actual examples of a class in the holdout set.
- **Accuracy interval:** approximate 95% Wilson binomial interval, calculated from the number of
  correct predictions and test-set size. It does not include variability from training seeds,
  model selection, or domain shift.

## 2. Dataset quality and structure

| Dataset | Rows | Columns | Missing values | Exact duplicate rows | Target classes |
|---|---:|---:|---:|---:|---|
| Dataset 1 · Crop & Fertilizer | 4,513 | 10 | 0 | 0 | 16 crops; 19 fertilizers |
| Dataset 2 · Crop Recommendation | 2,200 | 8 | 0 | 0 | 22 crops |

Dataset 1 has district, soil color, nitrogen, phosphorus, potassium, pH, rainfall, temperature,
crop, and fertilizer. Dataset 2 has N, P, K, temperature, humidity, pH, rainfall, and crop label.
Units and data-collection protocols are not given in these output files.

### Numeric-feature descriptions

Values below are minimum, median, and maximum from the cleaned datasets. Units are not specified.

| Dataset | Feature | Minimum | Median | Maximum | Mean | Standard deviation |
|---|---|---:|---:|---:|---:|---:|
| D1 | Nitrogen | 20 | 105 | 150 | 95.410 | 38.061 |
| D1 | Phosphorus | 10 | 55 | 90 | 54.342 | 16.552 |
| D1 | Potassium | 5 | 55 | 150 | 63.595 | 35.692 |
| D1 | pH | 5.5 | 6.5 | 8.5 | 6.715 | 0.625 |
| D1 | Rainfall | 300 | 800 | 1,700 | 819.189 | 251.731 |
| D1 | Temperature | 10 | 25 | 40 | 25.915 | 5.897 |
| D2 | N | 0 | 37 | 140 | 50.552 | 36.917 |
| D2 | P | 5 | 51 | 145 | 53.363 | 32.986 |
| D2 | K | 5 | 32 | 205 | 48.149 | 50.648 |
| D2 | Temperature | 8.826 | 25.599 | 43.675 | 25.616 | 5.064 |
| D2 | Humidity | 14.258 | 80.473 | 99.982 | 71.482 | 22.264 |
| D2 | pH | 3.505 | 6.425 | 9.935 | 6.469 | 0.774 |
| D2 | Rainfall | 20.211 | 94.868 | 298.560 | 103.464 | 54.958 |

Categorical summary in Dataset 1:

| Field | Most frequent value | Count | Other observations |
|---|---|---:|---|
| District | Kolhapur | 1,430 | 5 distinct districts |
| Soil color | Black | 2,260 | 6 distinct soil-color labels |
| Crop | Sugarcane | 1,010 | 16 classes |
| Fertilizer | Urea | 1,364 | 19 classes |

## 3. Class distributions and imbalance

### Dataset 1 crop distribution (4,513 records)

| Crop | Count | Share |
|---|---:|---:|
| Sugarcane | 1,010 | 22.380% |
| Wheat | 859 | 19.034% |
| Cotton | 650 | 14.403% |
| Jowar | 394 | 8.730% |
| Maize | 350 | 7.755% |
| Rice | 309 | 6.847% |
| Groundnut | 177 | 3.922% |
| Tur | 126 | 2.792% |
| Ginger | 125 | 2.770% |
| Grapes | 125 | 2.770% |
| Urad | 99 | 2.194% |
| Moong | 99 | 2.194% |
| Gram | 78 | 1.728% |
| Turmeric | 55 | 1.219% |
| Soybean | 45 | 0.997% |
| Masoor | 12 | 0.266% |

The largest-to-smallest class ratio is **84.2:1**. The top three crops account for **55.8%**
of records. Masoor has only 12 records in the full dataset, so its test support is very small.

### Dataset 1 fertilizer distribution (4,513 records)

| Fertilizer | Count | Share |
|---|---:|---:|
| Urea | 1,364 | 30.224% |
| DAP | 667 | 14.780% |
| MOP | 571 | 12.652% |
| 19:19:19 NPK | 480 | 10.636% |
| SSP | 417 | 9.240% |
| Magnesium Sulphate | 215 | 4.764% |
| 10:26:26 NPK | 156 | 3.457% |
| 50:26:26 NPK | 124 | 2.748% |
| Chilated Micronutrient | 108 | 2.393% |
| 12:32:16 NPK | 106 | 2.349% |
| Ferrous Sulphate | 68 | 1.507% |
| 13:32:26 NPK | 66 | 1.462% |
| Ammonium Sulphate | 50 | 1.108% |
| 10:10:10 NPK | 50 | 1.108% |
| Hydrated Lime | 25 | 0.554% |
| White Potash | 19 | 0.421% |
| 20:20:20 NPK | 15 | 0.332% |
| 18:46:00 NPK | 6 | 0.133% |
| Sulphur | 6 | 0.133% |

The largest-to-smallest class ratio is **227.3:1**. The top three labels are **57.7%** of
records; 18:46:00 NPK and Sulphur each have just 6 examples in the full data. This imbalance
makes a single overall accuracy score insufficient for evaluating fertilizer predictions.

### Dataset 2 crop distribution (2,200 records)

Every class has exactly **100 records (4.545%)**:

| Crop labels (each 100 records) |
|---|
| apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil |
| maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon |

This dataset is exactly balanced in the provided cleaned file. A 70/30 stratified split therefore
has 30 test examples per class (660 total).

### Majority-class reference

A classifier that always predicts the most common label would achieve about **22.38%** accuracy
for Dataset 1 crop, **30.22%** for Dataset 1 fertilizer, and **4.55%** for Dataset 2. These
are simple prevalence references, not substitutes for stronger baselines.

## 4. Current model performance

Scores are proportions in the source files, displayed here as percentages. Each model is scored
on the same holdout size within its task.

| Task | Model | Accuracy | Macro precision | Macro recall / balanced accuracy | Macro-F1 | Weighted F1 | Correct / test |
|---|---|---:|---:|---:|---:|---:|---:|
| D1 crop | 1D-CNN Model | 96.09% | 91.85% | 93.42% | 92.23% | 96.10% | 1,301 / 1,354 |
| D1 crop | Dual-Branch Fusion Net | 92.98% | 86.32% | 85.09% | 84.92% | 92.81% | 1,259 / 1,354 |
| D1 crop | Gradient Boosting Ensemble | 94.39% | 86.37% | 85.03% | 85.47% | 94.28% | 1,278 / 1,354 |
| D1 fertilizer | 1D-CNN Model | 50.74% | 41.89% | 40.64% | 40.45% | 50.38% | 687 / 1,354 |
| D1 fertilizer | Dual-Branch Fusion Net | 45.05% | 27.89% | 27.45% | 26.53% | 42.96% | 610 / 1,354 |
| D1 fertilizer | Gradient Boosting Ensemble | 43.80% | 40.78% | 38.79% | 38.63% | 43.36% | 593 / 1,354 |
| D2 crop label | 1D-CNN Model | 90.45% | 90.67% | 90.45% | 90.44% | 90.44% | 597 / 660 |
| D2 crop label | Dual-Branch Fusion Net | 90.61% | 90.84% | 90.61% | 90.59% | 90.59% | 598 / 660 |
| D2 crop label | Gradient Boosting Ensemble | 85.76% | 86.33% | 85.76% | 85.86% | 85.86% | 566 / 660 |

### Approximate accuracy intervals

| Task / model | Accuracy | Approximate 95% Wilson interval |
|---|---:|---:|
| D1 crop · 1D-CNN | 96.09% | 94.9–97.0% |
| D1 crop · Dual-Branch | 92.98% | 91.5–94.2% |
| D1 crop · Gradient Boosting | 94.39% | 93.0–95.5% |
| D1 fertilizer · 1D-CNN | 50.74% | 48.1–53.4% |
| D1 fertilizer · Dual-Branch | 45.05% | 42.4–47.7% |
| D1 fertilizer · Gradient Boosting | 43.80% | 41.2–46.5% |
| D2 crop · 1D-CNN | 90.45% | 88.0–92.5% |
| D2 crop · Dual-Branch | 90.61% | 88.1–92.6% |
| D2 crop · Gradient Boosting | 85.76% | 82.9–88.2% |

The intervals are for a fixed test set and are not paired model-comparison tests. The 0.16
percentage-point accuracy difference between Dataset 2 Dual-Branch and 1D-CNN corresponds to
one additional correct prediction (598 vs. 597); this single split does not establish a stable
winner.

### Quantitative summary file

`quantitative_analysis_summary.csv` reports means and standard deviations across the three
architectures for each target:

| Dataset / target | Mean accuracy | SD accuracy | Mean macro-F1 | SD macro-F1 | Highest macro-F1 |
|---|---:|---:|---:|---:|---|
| D1 · Crop | 94.49% | 1.27 pp | 87.54% | 3.32 pp | 1D-CNN, 92.23% |
| D1 · Fertilizer | 46.53% | 3.02 pp | 35.20% | 6.18 pp | 1D-CNN, 40.45% |
| D2 · label | 88.94% | 2.25 pp | 88.96% | 2.20 pp | Dual-Branch, 90.59% |

These standard deviations describe differences among three model architectures, **not** repeated
run or random-seed uncertainty.

## 5. Per-class F1 and test support

Values are recomputed from the current confusion matrices and rounded to two decimals. The test
support column is the number of actual examples of that class in the holdout set.

### Dataset 1 crop

| Class | Test support | 1D-CNN F1 | Dual-Branch F1 | Gradient Boosting F1 |
|---|---:|---:|---:|---:|
| Cotton | 195 | 1.00 | 0.98 | 0.99 |
| Ginger | 38 | 0.99 | 0.99 | 1.00 |
| Gram | 23 | 0.73 | 0.55 | 0.41 |
| Grapes | 37 | 1.00 | 0.99 | 1.00 |
| Groundnut | 53 | 0.88 | 0.82 | 0.86 |
| Jowar | 118 | 0.92 | 0.88 | 0.89 |
| Maize | 105 | 0.95 | 0.95 | 0.94 |
| Masoor | 4 | 0.89 | 0.67 | 0.50 |
| Moong | 30 | 0.93 | 0.83 | 0.84 |
| Rice | 93 | 0.99 | 0.90 | 0.96 |
| Soybean | 13 | 0.81 | 0.89 | 0.74 |
| Sugarcane | 303 | 1.00 | 0.99 | 1.00 |
| Tur | 38 | 0.97 | 0.92 | 0.92 |
| Turmeric | 16 | 0.97 | 0.67 | 0.90 |
| Urad | 30 | 0.77 | 0.63 | 0.76 |
| Wheat | 258 | 0.96 | 0.95 | 0.96 |

The lowest and most model-sensitive class is Gram. Masoor's apparently high 1D-CNN F1 is based
on only four test examples and should not be read as precise evidence of generalization.

### Dataset 1 fertilizer

| Class | Test support | 1D-CNN F1 | Dual-Branch F1 | Gradient Boosting F1 |
|---|---:|---:|---:|---:|
| 10:10:10 NPK | 15 | 0.42 | 0.51 | 0.37 |
| 10:26:26 NPK | 47 | 0.34 | 0.22 | 0.32 |
| 12:32:16 NPK | 32 | 0.15 | 0.00 | 0.30 |
| 13:32:26 NPK | 20 | 0.54 | 0.00 | 0.50 |
| 18:46:00 NPK | 2 | 0.00 | 0.00 | 0.00 |
| 19:19:19 NPK | 144 | 0.67 | 0.65 | 0.53 |
| 20:20:20 NPK | 4 | 0.33 | 0.00 | 0.36 |
| 50:26:26 NPK | 37 | 0.71 | 0.56 | 0.52 |
| Ammonium Sulphate | 15 | 0.52 | 0.25 | 0.59 |
| Chilated Micronutrient | 32 | 0.52 | 0.53 | 0.43 |
| DAP | 200 | 0.44 | 0.46 | 0.40 |
| Ferrous Sulphate | 20 | 0.29 | 0.27 | 0.43 |
| Hydrated Lime | 8 | 0.40 | 0.17 | 0.31 |
| MOP | 171 | 0.53 | 0.38 | 0.42 |
| Magnesium Sulphate | 65 | 0.46 | 0.17 | 0.42 |
| SSP | 125 | 0.44 | 0.38 | 0.31 |
| Sulphur | 2 | 0.00 | 0.00 | 0.29 |
| Urea | 409 | 0.54 | 0.50 | 0.48 |
| White Potash | 6 | 0.37 | 0.00 | 0.36 |

Only a few classes have F1 above 0.60, and several classes have zero F1 for one or more
architectures. The tiny test supports for 18:46:00 NPK, Sulphur, White Potash, and 20:20:20 NPK
make those class-level estimates particularly unstable.

### Dataset 2 crop recommendation

Each class has 30 test examples.

| Class | Test support | 1D-CNN F1 | Dual-Branch F1 | Gradient Boosting F1 |
|---|---:|---:|---:|---:|
| apple | 30 | 0.98 | 1.00 | 0.93 |
| banana | 30 | 0.92 | 0.90 | 0.93 |
| blackgram | 30 | 0.87 | 0.86 | 0.77 |
| chickpea | 30 | 1.00 | 0.98 | 0.98 |
| coconut | 30 | 0.93 | 0.95 | 0.90 |
| coffee | 30 | 0.98 | 0.97 | 0.85 |
| cotton | 30 | 0.92 | 0.90 | 0.86 |
| grapes | 30 | 0.98 | 1.00 | 0.94 |
| jute | 30 | 0.81 | 0.79 | 0.75 |
| kidneybeans | 30 | 0.95 | 0.93 | 1.00 |
| lentil | 30 | 0.84 | 0.85 | 0.78 |
| maize | 30 | 0.90 | 0.90 | 0.80 |
| mango | 30 | 0.94 | 0.93 | 0.86 |
| mothbeans | 30 | 0.90 | 0.95 | 0.83 |
| mungbean | 30 | 0.92 | 0.90 | 0.85 |
| muskmelon | 30 | 0.89 | 0.91 | 0.83 |
| orange | 30 | 0.83 | 0.84 | 0.85 |
| papaya | 30 | 0.87 | 0.87 | 0.78 |
| pigeonpeas | 30 | 0.90 | 0.90 | 0.95 |
| pomegranate | 30 | 0.83 | 0.86 | 0.86 |
| rice | 30 | 0.87 | 0.84 | 0.79 |
| watermelon | 30 | 0.87 | 0.89 | 0.82 |

Jute is the weakest class across all three models. Lentil, orange, rice, and pomegranate are
also recurring weaker classes. The per-class results are fairly consistent across architectures,
but 30 examples per class still yield substantial uncertainty at the class level.

## 6. Confusion patterns

Largest directional errors from each current confusion matrix (actual class → predicted class):

| Task | Model | Total errors | Largest confusion pairs |
|---|---|---:|---|
| D1 crop | 1D-CNN | 53 | Wheat → Maize (11); Jowar → Groundnut (5); Gram → Groundnut (5) |
| D1 crop | Dual-Branch | 95 | Gram → Jowar (8); Maize → Wheat (6); Jowar → Groundnut (6) |
| D1 crop | Gradient Boosting | 76 | Gram → Jowar (9); Wheat → Maize (6); Maize → Wheat (6) |
| D1 fertilizer | 1D-CNN | 667 | Urea → 19:19:19 NPK (51); MOP → Urea (48); Urea → MOP (43) |
| D1 fertilizer | Dual-Branch | 744 | Magnesium Sulphate → DAP (55); Urea → 19:19:19 NPK (52); 19:19:19 NPK → Urea (48) |
| D1 fertilizer | Gradient Boosting | 761 | Urea → 19:19:19 NPK (68); 19:19:19 NPK → Urea (65); MOP → Urea (62) |
| D2 crop | 1D-CNN | 63 | Watermelon → Muskmelon (6); Orange → Pomegranate (5); Lentil → Blackgram (4) |
| D2 crop | Dual-Branch | 62 | Rice → Jute (7); Watermelon → Muskmelon (5); Orange → Pomegranate (5) |
| D2 crop | Gradient Boosting | 94 | Rice → Jute (6); Muskmelon → Watermelon (6); Blackgram → Lentil (5) |

The fertilizer models repeatedly confuse Urea, 19:19:19 NPK, MOP, DAP, and SSP. This is
consistent with broader difficulty distinguishing fertilizer labels from the available feature
set, but the confusion matrix alone cannot identify the cause. Dataset 2 errors recur between
some crop pairs, including watermelon/muskmelon, orange/pomegranate, lentil/blackgram, and
rice/jute.

## 7. Outlier screens

The files use IQR bounds. Counts below are flagged feature-value occurrences divided by the
number of dataset rows. One row can be flagged on multiple features; these rates must not be added
to estimate unique affected records.

### Dataset 1 (4,513 rows)

| Feature | IQR lower bound | IQR upper bound | Flagged occurrences | Share of rows |
|---|---:|---:|---:|---:|
| Nitrogen | -37.5 | 222.5 | 0 | 0.00% |
| Phosphorus | 2.5 | 102.5 | 0 | 0.00% |
| Potassium | -12.5 | 127.5 | 495 | 10.97% |
| pH | 4.5 | 8.5 | 0 | 0.00% |
| Rainfall | 0.0 | 1,600.0 | 9 | 0.20% |
| Temperature | 5.0 | 45.0 | 0 | 0.00% |

There are **504 feature-value occurrences** flagged in total. Potassium accounts for nearly all
of them. Because the lower bound is negative and observed values are non-negative, the 495
Potassium flags are above the upper bound. These may represent a meaningful high-potassium
subpopulation rather than data-entry errors.

### Dataset 2 (2,200 rows)

| Feature | IQR lower bound | IQR upper bound | Flagged occurrences | Share of rows |
|---|---:|---:|---:|---:|
| N | -73.875 | 179.125 | 0 | 0.00% |
| P | -32.000 | 128.000 | 138 | 6.27% |
| K | -23.500 | 92.500 | 200 | 9.09% |
| temperature | 14.081 | 37.250 | 86 | 3.91% |
| humidity | 15.732 | 134.478 | 30 | 1.36% |
| pH | 4.544 | 8.352 | 57 | 2.59% |
| rainfall | -25.022 | 213.841 | 100 | 4.55% |

There are **611 feature-value occurrences** flagged in total. The IQR rule is distribution-based,
not domain-based: the negative lower bounds for P, K, and rainfall are not physically meaningful
measurements, but they do not produce flags for low observed values. The listed bounds should
not be treated as agronomic acceptance criteria.

The output does not indicate that these observations were removed from the cleaned datasets.
Review them against units, measurement protocol, and label context before excluding or clipping
any values.

## 8. Label conflicts and numeric associations

### Conflicting fertilizer labels

`Dataset_1_Crop_Fertilizer_Fertilizer_conflicts.csv` contains three repeated feature profiles,
each occurring twice with different fertilizer labels:

| District / soil / nutrients / pH / rainfall / temperature / crop | Rows | Fertilizer labels |
|---|---:|---|
| Kolhapur / Dark Brown / N=100, P=40, K=40 / pH=6.0 / rainfall=500 / temperature=15 / Wheat | 2 | Urea; 19:19:19 NPK |
| Kolhapur / Dark Brown / N=100, P=45, K=45 / pH=6.5 / rainfall=600 / temperature=20 / Wheat | 2 | Urea; 19:19:19 NPK |
| Sangli / Black / N=25, P=20, K=40 / pH=5.5 / rainfall=1,500 / temperature=20 / Turmeric | 2 | Urea; White Potash |

This represents **3 conflicting feature groups and 6 records**. The crop-conflict files are
newline-only/empty. The fertilizer conflicts show that the stored predictors do not uniquely
determine the target for every record; possible explanations (not established by these files)
include omitted context, stochastic recommendations, or labeling inconsistency.

### Largest Pearson correlations among numeric predictors

| Dataset | Feature pair | Correlation |
|---|---|---:|
| D1 | Nitrogen – Phosphorus | +0.710 |
| D1 | Nitrogen – Potassium | +0.584 |
| D1 | Phosphorus – Potassium | +0.574 |
| D1 | Potassium – Rainfall | +0.446 |
| D2 | P – K | +0.736 |
| D2 | N – P | -0.231 |
| D2 | Temperature – Humidity | +0.205 |
| D2 | K – Humidity | +0.191 |

These are descriptive linear associations computed from the cleaned numeric columns. They do
not establish causation, feature importance, or correlation with the target. Correlated nutrient
features can also make it harder to assign model behavior to a single nutrient independently.

## 9. Result-file consistency

The output folder contains more than one generation of model results. The current model metrics
match the CSV confusion matrices with the `_confusion_matrix.csv` suffix. The older `three_papers_results.csv`
does not match the current realistic results and does not reconcile with those current matrices.

Examples:

| Task / model | Current metrics and suffixed matrix | Older `three_papers_results.csv` |
|---|---:|---:|
| D1 crop · 1D-CNN accuracy | 96.09% | 99.04% |
| D1 crop · Gradient Boosting accuracy | 94.39% | 99.78% |
| D1 fertilizer · Gradient Boosting macro-F1 | 38.63% | 65.85% |
| D1 fertilizer · Dual-Branch macro-F1 | 26.53% | 33.86% |

Some per-model metric files exist in pairs, such as
`Dataset_1_Crop_Fertilizer__Crop__1D-CNN_Model.csv` and
`Dataset_1_Crop_Fertilizer__Crop__1D-CNN_Model_metrics.csv`. The unsuffixed member of these
pairs has the older values; the `_metrics.csv` member matches `model_metrics.csv`.

**Recommended source of truth:** `model_metrics.csv` and the matching `metrics/*_metrics.csv`
files, cross-checked against the suffixed confusion matrices. The current comparison and realistic
results CSV/XLSX exports agree with this source. Treat `three_papers_results.csv/.xlsx` and
unsuffixed per-model metrics as historical, not current.

## 10. Methodology and interpretation limits

The experiment configuration in `analyze_crop_fertilizer.py` indicates:

- Random seed: 42.
- Stratified train/test split: 70% training, 30% test.
- Sensor noise: additive Gaussian noise with standard deviation 0.25 times each numeric
  feature's full-dataset standard deviation.
- Numeric features are median-imputed and standardized; categorical features are one-hot encoded.
  The preprocessing transformer is fit on the training partition and applied to the test set.
- Three model architectures are compared for each task.

Important caveats:

1. **One holdout only.** No repeated-seed or cross-validation results are provided. Rankings may
   change under different splits or model initialization.
2. **Noise is simulated.** A fixed 0.25×SD Gaussian perturbation is not evidence of real sensor
   noise. The noise is applied before the split, so each transformed row is then assigned to
   train or test; the scale is based on the full feature vector.
3. **No external validation.** The outputs do not establish performance on a new farm, district,
   season, data source, or sensor.
4. **Units and collection context are absent.** The ranges cannot be judged against agronomic
   tolerances without units and provenance.
5. **Class imbalance matters.** Accuracy overstates the performance of models that do poorly on
   rare labels; macro-F1 and per-class support should be considered alongside it.
6. **Conflicting labels matter.** Repeated feature profiles with different fertilizer labels
   make perfect deterministic prediction impossible for those profiles using only the listed
   inputs.
7. **No calibration or decision costs.** There are no calibration curves, confidence thresholds,
   or cost-sensitive metrics. Incorrect fertilizer recommendations can have unequal practical
   consequences.
8. **“Three papers” does not prove paper reproduction.** The filenames and paper labels identify
   the compared implementations; these outputs alone do not validate that published studies were
   reproduced faithfully.
9. **Feature correlations are not explanations.** The correlation summaries are not model
   feature importance and should not be interpreted causally.

## 11. Recommended next steps

1. Regenerate all outputs from one clean run and remove or clearly archive stale metric files so
   consumers cannot accidentally compare different result generations.
2. Repeat evaluation across multiple seeds or use cross-validation; report mean, spread, and
   per-class confidence intervals.
3. For geographically structured data, reserve a district/farm or later season for external
   testing rather than relying solely on a random stratified split.
4. Investigate the three repeated fertilizer profiles and review rare-class labels and counts
   with domain experts.
5. Compare cost-sensitive and class-balanced approaches for fertilizer prediction; report
   per-class precision, recall, F1, and confusion counts.
6. Validate the simulated noise against measured sensor specifications and test several realistic
   noise regimes.
7. Document feature units, collection dates, locations, and label-generation protocol.
8. Evaluate probability calibration and define abstention/escalation behavior for low-confidence
   recommendations before considering operational use.

## Appendix: concise findings by task

- **Dataset 1 crop:** 1D-CNN is the clear current leader on macro-F1 (92.23%). The weaker Gram
  scores and a small set of recurring crop confusions merit targeted review.
- **Dataset 1 fertilizer:** 1D-CNN leads the current macro-F1 ranking (40.45%), but the task
  remains unreliable overall. Macro recall is low and class-level performance is uneven.
- **Dataset 2 crop recommendation:** 1D-CNN and Dual-Branch are effectively tied on this split;
  Gradient Boosting trails by roughly 4.6 percentage points of accuracy. Jute and several
  crop-pair confusions are the main remaining weak spots.
- **Data quality:** no missing data or exact duplicate rows, but IQR flags and conflicting
  fertilizer targets require contextual review.
- **Operational readiness:** not established by these outputs; external validation, repeatability,
  calibrated uncertainty, and agronomic review are still needed.
