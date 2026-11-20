"""Evaluate saved ML research artifacts without retraining the models."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from analyze_crop_fertilizer import load_and_prep, resolve_dataset_path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "analysis_outputs"
OUTPUT_DIR = ROOT / "Research_Evaluation_Results"
SEED = 42
BOOTSTRAP_SEED = 20261115
BOOTSTRAP_REPLICATES = 1000
MAX_EMBEDDING_SAMPLES = 1200

DATASET_SPECS = (
    {
        "key": "Dataset_1_Crop_Fertilizer",
        "name": "Dataset 1 (Crop & Fertilizer)",
        "file_names": (
            "Crop and fertilizer dataset_3.csv",
            "Crop and fertilizer dataset_2.csv",
            "Crop and fertilizer dataset.csv",
        ),
        "targets": ("Fertilizer", "Crop"),
    },
    {
        "key": "Dataset_2_Crop_Recommendation",
        "name": "Dataset 2 (Crop Recommendation)",
        "file_names": (
            "Crop_recommendation_3.csv",
            "Crop_recommendation_2.csv",
            "Crop_recommendation.csv",
        ),
        "targets": ("label",),
    },
)

MODEL_NAMES = (
    "1D-CNN Model",
    "Dual-Branch Fusion Net",
    "Gradient Boosting Ensemble",
)

PLOT_STYLE = {
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def configure_output() -> None:
    directories = (
        "methodologies",
        "quantitative_metrics",
        "qualitative_graphs",
        "classification_curves",
        "explainability_visualizations",
        "robustness_graphs",
        "statistical_analysis",
        "ablation_studies",
        "comparative_visualizations",
    )
    for directory in directories:
        (OUTPUT_DIR / directory).mkdir(parents=True, exist_ok=True)
    for model in MODEL_NAMES:
        (OUTPUT_DIR / "methodologies" / safe_filename(model)).mkdir(
            parents=True, exist_ok=True
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metrics_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot evaluate an empty confusion matrix.")

    true_positive = np.diag(matrix).astype(float)
    supports = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, supports, out=np.zeros_like(true_positive), where=supports > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    specificity = np.divide(
        total - supports - predicted + true_positive,
        total - supports,
        out=np.zeros_like(true_positive),
        where=(total - supports) > 0,
    )
    observed_agreement = float(true_positive.sum() / total)
    expected_count = float(np.dot(supports, predicted))
    expected_agreement = expected_count / (total * total)
    kappa_denominator = 1.0 - expected_agreement
    kappa = (
        (observed_agreement - expected_agreement) / kappa_denominator
        if kappa_denominator
        else 0.0
    )
    mcc_denominator = np.sqrt(
        (total * total - np.dot(predicted, predicted))
        * (total * total - np.dot(supports, supports))
    )
    mcc_numerator = total * true_positive.sum() - np.dot(supports, predicted)
    mcc = mcc_numerator / mcc_denominator if mcc_denominator else 0.0

    return {
        "Accuracy": observed_agreement,
        "Precision": float(np.mean(precision)),
        "Macro_Precision": float(np.mean(precision)),
        "Weighted_Precision": float(np.average(precision, weights=supports)),
        "Micro_Precision": observed_agreement,
        "Recall": float(np.mean(recall)),
        "Macro_Recall": float(np.mean(recall)),
        "Weighted_Recall": float(np.average(recall, weights=supports)),
        "Micro_Recall": observed_agreement,
        "F1_Score": float(np.mean(f1)),
        "Macro_F1": float(np.mean(f1)),
        "Weighted_F1": float(np.average(f1, weights=supports)),
        "Micro_F1": observed_agreement,
        "Balanced_Accuracy": float(np.mean(recall[supports > 0])),
        "MCC": float(mcc),
        "Cohen_Kappa": float(kappa),
        "Sensitivity": float(np.mean(recall)),
        "Macro_Specificity": float(np.mean(specificity)),
        "Weighted_Specificity": float(np.average(specificity, weights=supports)),
        "Top_1_Accuracy": observed_agreement,
    }


def class_metrics_from_matrix(matrix: np.ndarray, classes: list[str]) -> pd.DataFrame:
    totals = int(matrix.sum())
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    fp = predicted - tp
    fn = support - tp
    tn = totals - tp - fp - fn
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    specificity = np.divide(
        tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0
    )
    return pd.DataFrame(
        {
            "Class": classes,
            "Support": support.astype(int),
            "Precision": precision,
            "Recall_Sensitivity": recall,
            "F1_Score": f1,
            "Specificity": specificity,
        }
    )


def bootstrap_intervals(matrix: np.ndarray, rng: np.random.Generator) -> list[dict]:
    row_sums = matrix.sum(axis=1)
    values = {"Accuracy": [], "Macro_F1": [], "Weighted_F1": [], "Balanced_Accuracy": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = np.zeros_like(matrix)
        for class_index, row_total in enumerate(row_sums):
            if row_total:
                sample[class_index] = rng.multinomial(
                    int(row_total), matrix[class_index] / row_total
                )
        metrics = metrics_from_matrix(sample)
        for metric in values:
            values[metric].append(metrics[metric])
    return [
        {
            "Metric": metric,
            "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
            "Confidence_Level": 0.95,
            "CI_Lower": float(np.quantile(samples, 0.025)),
            "CI_Upper": float(np.quantile(samples, 0.975)),
            "Method": "Stratified nonparametric bootstrap from confusion-matrix rows",
        }
        for metric, samples in values.items()
    ]


def load_saved_confusion_matrices() -> list[dict]:
    matrix_dir = SOURCE_DIR / "confusion_matrices"
    artifacts = []
    for path in sorted(matrix_dir.glob("*_confusion_matrix.csv")):
        parts = path.name.removesuffix("_confusion_matrix.csv").split("__")
        if len(parts) != 3:
            warnings.warn(f"Skipping unrecognized confusion matrix filename: {path.name}")
            continue
        dataset_key, target, model = parts
        model = next(
            (name for name in MODEL_NAMES if safe_filename(name) == model),
            model,
        )
        table = pd.read_csv(path, index_col=0)
        matrix = table.to_numpy(dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected a square confusion matrix in {path}.")
        artifacts.append(
            {
                "Dataset_Key": dataset_key,
                "Dataset": next(
                    (
                        item["name"]
                        for item in DATASET_SPECS
                        if item["key"] == dataset_key
                    ),
                    dataset_key,
                ),
                "Target": target,
                "Model": model,
                "Classes": [str(value) for value in table.index],
                "Matrix": matrix,
                "Source_File": str(path.relative_to(ROOT)),
            }
        )
    if not artifacts:
        raise FileNotFoundError(f"No saved confusion matrices found under {matrix_dir}.")
    return artifacts


def save_confusion_visualizations(artifact: dict) -> None:
    model_dir = OUTPUT_DIR / "methodologies" / safe_filename(artifact["Model"])
    short_name = f'{safe_filename(artifact["Dataset_Key"])}_{safe_filename(artifact["Target"])}'
    matrix = artifact["Matrix"]
    classes = artifact["Classes"]

    for normalized in (False, True):
        displayed = matrix.astype(float)
        if normalized:
            denominators = displayed.sum(axis=1, keepdims=True)
            displayed = np.divide(
                displayed,
                denominators,
                out=np.zeros_like(displayed),
                where=denominators > 0,
            )
        side = min(max(8.0, len(classes) * 0.48), 17.0)
        fig, ax = plt.subplots(figsize=(side, side * 0.88))
        sns.heatmap(
            displayed,
            cmap="Blues",
            annot=len(classes) <= 22,
            fmt=".2f" if normalized else "g",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Row proportion" if normalized else "Sample count"},
            ax=ax,
        )
        ax.set(
            title=(
                f'{artifact["Dataset"]} | {artifact["Target"]} | {artifact["Model"]}'
                + (" — row normalized" if normalized else " — counts")
            ),
            xlabel="Predicted class",
            ylabel="Actual class",
        )
        ax.tick_params(axis="x", labelrotation=90)
        ax.tick_params(axis="y", labelrotation=0)
        fig.tight_layout()
        suffix = "normalized_confusion_matrix" if normalized else "confusion_matrix"
        save_figure(fig, model_dir / f"{short_name}_{suffix}")

    class_frame = class_metrics_from_matrix(matrix, classes)
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.65), 6.2))
    class_frame.set_index("Class")[["Precision", "Recall_Sensitivity", "F1_Score"]].plot(
        kind="bar", ax=ax, color=["#4477AA", "#228833", "#CC6677"], width=0.82
    )
    ax.set(
        title=f'Class-wise performance — {artifact["Dataset"]} | {artifact["Target"]} | {artifact["Model"]}',
        xlabel="Class",
        ylabel="Score",
        ylim=(0, 1.05),
    )
    ax.legend(title="Metric", loc="lower right")
    ax.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    save_figure(fig, model_dir / f"{short_name}_classwise_precision_recall_f1")


def compare_to_saved_summary(artifacts: list[dict], metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Dataset", "Target", "Model", "Metric", "Recomputed", "Saved", "Difference",
        "Matches_At_Saved_Precision",
    ]
    summary_path = SOURCE_DIR / "model_metrics.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=columns)
    saved = pd.read_csv(summary_path)
    checks = []
    for _, row in metrics.iterrows():
        matching = saved[
            (saved["Dataset"] == row["Dataset"])
            & (saved["Target"].astype(str).str.lower() == str(row["Target"]).lower())
            & (saved["Model"] == row["Model"])
        ]
        if matching.empty:
            continue
        saved_row = matching.iloc[0]
        for metric_name, saved_name in (
            ("Accuracy", "Accuracy"),
            ("Macro_Precision", "Macro_Precision"),
            ("Macro_Recall", "Macro_Recall"),
            ("Macro_F1", "Macro_F1"),
        ):
            if saved_name not in saved_row.index:
                continue
            recomputed = float(row[metric_name])
            saved_value = float(saved_row[saved_name])
            checks.append(
                {
                    "Dataset": row["Dataset"],
                    "Target": row["Target"],
                    "Model": row["Model"],
                    "Metric": metric_name,
                    "Recomputed": recomputed,
                    "Saved": saved_value,
                    "Difference": recomputed - saved_value,
                    "Matches_At_Saved_Precision": round(recomputed, 4) == round(saved_value, 4),
                }
            )
    return pd.DataFrame(checks, columns=columns)


def save_comparative_plots(metrics: pd.DataFrame) -> None:
    plot_metrics = [
        "Accuracy",
        "Macro_F1",
        "Weighted_F1",
        "Balanced_Accuracy",
        "MCC",
    ]
    for (dataset, target), group in metrics.groupby(["Dataset", "Target"], sort=False):
        fig, ax = plt.subplots(figsize=(10.5, 6))
        plotted = group.set_index("Model")[plot_metrics]
        plotted.plot(kind="bar", ax=ax, width=0.82)
        ax.set(
            title=f"Methodology comparison — {dataset} | {target}",
            xlabel="Methodology",
            ylabel="Score",
            ylim=(-1.05, 1.05),
        )
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", labelrotation=18)
        fig.tight_layout()
        save_figure(
            fig,
            OUTPUT_DIR
            / "comparative_visualizations"
            / f"{safe_filename(dataset)}_{safe_filename(target)}_metric_comparison",
        )

    summary = metrics.groupby("Model", as_index=False)[["Accuracy", "Macro_F1"]].mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    summary.set_index("Model").plot(kind="bar", ax=ax, color=["#4477AA", "#CC6677"])
    ax.set(
        title="Mean saved-result performance across classification tasks",
        xlabel="Methodology",
        ylabel="Mean score across tasks",
        ylim=(0, 1.05),
    )
    ax.legend(title="Metric")
    ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    save_figure(
        fig, OUTPUT_DIR / "comparative_visualizations" / "mean_model_performance"
    )


def exploratory_dataset_plots() -> tuple[list[dict], pd.DataFrame]:
    availability = []
    profiles = []
    for spec in DATASET_SPECS:
        csv_path = resolve_dataset_path(spec["file_names"])
        for target in spec["targets"]:
            X, y = load_and_prep(csv_path, target)
            source = pd.read_csv(csv_path)
            numeric_columns = X.select_dtypes(include=np.number).columns
            categorical_count = len(X.columns) - len(numeric_columns)
            class_counts = y.value_counts()
            profiles.append(
                {
                    "Dataset_Key": spec["key"],
                    "Dataset": spec["name"],
                    "Source_File": csv_path.name,
                    "Target": target,
                    "Rows": len(X),
                    "Predictor_Count": len(X.columns),
                    "Numeric_Predictor_Count": len(numeric_columns),
                    "Categorical_Predictor_Count": categorical_count,
                    "Target_Class_Count": int(y.nunique()),
                    "Missing_Predictor_Cells": int(X.isna().sum().sum()),
                    "Missing_Target_Values": int(source[target].isna().sum())
                    if target in source.columns
                    else 0,
                    "Majority_Class": str(class_counts.index[0]),
                    "Majority_Class_Share": float(class_counts.iloc[0] / len(y)),
                }
            )
            numeric = X.select_dtypes(include=np.number).replace(
                [np.inf, -np.inf], np.nan
            )
            frame = numeric.copy()
            frame["Target"] = y.astype(str)
            label = f'{safe_filename(spec["key"])}_{safe_filename(target)}'
            target_dir = OUTPUT_DIR / "qualitative_graphs" / label
            target_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(max(10, y.nunique() * 0.55), 5.5))
            counts = y.value_counts().sort_values(ascending=False)
            sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4477AA", ax=ax)
            ax.set(
                title=f"Class distribution — {spec['name']} | {target}",
                xlabel="Class",
                ylabel="Number of observations",
            )
            ax.tick_params(axis="x", labelrotation=55)
            fig.tight_layout()
            save_figure(fig, target_dir / "class_distribution")

            if len(numeric.columns):
                columns = list(numeric.columns)
                fig, axes = plt.subplots(
                    len(columns), 1, figsize=(9.5, max(3.2, 2.5 * len(columns)))
                )
                axes = np.atleast_1d(axes)
                for ax, column in zip(axes, columns):
                    sns.histplot(numeric[column].dropna(), bins=30, color="#4477AA", ax=ax)
                    ax.set(title=f"Feature distribution: {column}", xlabel=column, ylabel="Count")
                fig.suptitle(f"Feature distributions — {spec['name']} | {target}")
                fig.tight_layout()
                save_figure(fig, target_dir / "feature_distribution_histograms")

                if len(columns) >= 2:
                    correlation = numeric.corr()
                    fig, ax = plt.subplots(
                        figsize=(max(8, len(columns) * 0.8), max(6, len(columns) * 0.65))
                    )
                    sns.heatmap(
                        correlation,
                        cmap="vlag",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        annot=len(columns) <= 12,
                        fmt=".2f",
                        square=True,
                        ax=ax,
                    )
                    ax.set_title(f"Pearson correlation — {spec['name']} | {target}")
                    fig.tight_layout()
                    save_figure(fig, target_dir / "numeric_feature_correlation")

                    fig, axes = plt.subplots(
                        len(columns), 2, figsize=(12, max(4, 2.8 * len(columns)))
                    )
                    for row_index, column in enumerate(columns):
                        sns.boxplot(x=numeric[column], color="#4477AA", ax=axes[row_index, 0])
                        axes[row_index, 0].set(title=f"Box plot: {column}", xlabel=column)
                        sns.violinplot(
                            x=numeric[column],
                            color="#66CCEE",
                            inner="quart",
                            cut=0,
                            ax=axes[row_index, 1],
                        )
                        axes[row_index, 1].set(title=f"Violin plot: {column}", xlabel=column)
                    fig.suptitle(f"Feature spread — {spec['name']} | {target}")
                    fig.tight_layout()
                    save_figure(fig, target_dir / "feature_box_and_violin_plots")

                    pair_columns = columns[: min(4, len(columns))]
                    pair_frame = numeric[pair_columns].copy()
                    pair_frame["Class"] = y.astype(str).to_numpy()
                    if len(pair_frame) > 600:
                        pair_frame = pair_frame.sample(600, random_state=SEED)
                    grid = sns.pairplot(
                        pair_frame,
                        vars=pair_columns,
                        corner=True,
                        diag_kind="hist",
                        plot_kws={"s": 10, "alpha": 0.4, "color": "#4477AA"},
                    )
                    grid.fig.suptitle(
                        f"Pair plot (first {len(pair_columns)} numeric features) — {spec['name']} | {target}",
                        y=1.02,
                    )
                    save_figure(grid.fig, target_dir / "numeric_feature_pair_plot")

                    # These embeddings describe the dataset only; they are not test-set scores.
                    clean_numeric = numeric.fillna(numeric.median()).fillna(0.0)
                    sample_size = min(len(clean_numeric), MAX_EMBEDDING_SAMPLES)
                    sample_indices = (
                        clean_numeric.sample(sample_size, random_state=SEED).index
                        if sample_size < len(clean_numeric)
                        else clean_numeric.index
                    )
                    embedding_input = clean_numeric.loc[sample_indices]
                    embedding_labels = y.loc[sample_indices].astype(str)
                    if len(embedding_input) >= 3:
                        pca_coordinates = PCA(n_components=2).fit_transform(embedding_input)
                        tsne_coordinates = TSNE(
                            n_components=2,
                            random_state=SEED,
                            init="pca",
                            learning_rate="auto",
                            perplexity=min(30.0, (len(embedding_input) - 1) / 3),
                            max_iter=1000,
                        ).fit_transform(embedding_input)
                        for method, coordinates in (
                            ("PCA", pca_coordinates),
                            ("tSNE", tsne_coordinates),
                        ):
                            embedding = pd.DataFrame(
                                {
                                    "Component 1": coordinates[:, 0],
                                    "Component 2": coordinates[:, 1],
                                    "Class": embedding_labels.to_numpy(),
                                }
                            )
                            fig, ax = plt.subplots(figsize=(10, 7))
                            sns.scatterplot(
                                data=embedding,
                                x="Component 1",
                                y="Component 2",
                                hue="Class",
                                palette="tab20",
                                s=24,
                                alpha=0.75,
                                linewidth=0,
                                ax=ax,
                            )
                            ax.set_title(
                                f"{method} visualization (exploratory) — {spec['name']} | {target}"
                            )
                            ax.legend(
                                title="Class",
                                bbox_to_anchor=(1.02, 1),
                                loc="upper left",
                                ncol=1,
                            )
                            fig.tight_layout()
                            save_figure(fig, target_dir / f"{method.lower()}_visualization")
                        availability.extend(
                            [
                                {
                                    "Requested_Output": f"{method} 2D visualization",
                                    "Status": "Generated",
                                    "Reason": "Exploratory projection of source features; not a model evaluation.",
                                    "Path": str(target_dir.relative_to(OUTPUT_DIR)),
                                }
                                for method in ("PCA", "t-SNE")
                            ]
                        )
            else:
                availability.append(
                    {
                        "Requested_Output": f"Numeric feature plots for {spec['name']} | {target}",
                        "Status": "Unavailable",
                        "Reason": "No numeric predictor columns were available.",
                        "Path": "",
                    }
                )
    return availability, pd.DataFrame(profiles)


def methodology_profiles() -> pd.DataFrame:
    preprocessing = (
        "Numeric: median imputation + StandardScaler; categorical: OneHotEncoder "
        "(handle_unknown='ignore'); stratified 70/30 split, random_state=42; "
        "numeric inputs receive Gaussian noise with SD=0.25×feature SD before split."
    )
    profiles = {
        "1D-CNN Model": {
            "Algorithm": "PyTorch 1D convolutional neural network",
            "Architecture": (
                "Conv1d 1→32 (k=3,p=1), BatchNorm, ReLU; Conv1d 32→64 "
                "(k=3,p=1), BatchNorm, ReLU; AdaptiveAvgPool1d(4); "
                "Linear 256→64, ReLU, Dropout(0.3), Linear 64→classes."
            ),
            "Training": "150 epochs; Adam lr=0.003, weight_decay=1e-4; batch size 32; cross-entropy.",
        },
        "Dual-Branch Fusion Net": {
            "Algorithm": "PyTorch dual-branch CNN/MLP feature-fusion classifier",
            "Architecture": (
                "Encoded features split at floor(input_dim/2). Branch 1: Conv1d "
                "1→16 (k=2,p=1), ReLU, AdaptiveAvgPool1d(2). Branch 2: Linear "
                "remaining_features→16, ReLU. Concatenate 32+16 features; "
                "Linear 48→32, ReLU, Dropout(0.2), Linear 32→classes."
            ),
            "Training": "150 epochs; Adam lr=0.003, weight_decay=1e-4; batch size 32; cross-entropy.",
        },
        "Gradient Boosting Ensemble": {
            "Algorithm": "scikit-learn GradientBoostingClassifier",
            "Architecture": "Sequential gradient-boosted decision-tree ensemble.",
            "Training": "70 estimators; max_depth=4; learning_rate=0.08; random_state=42.",
        },
    }
    return pd.DataFrame(
        [
            {
                "Methodology": method,
                **details,
                "Shared_Preprocessing_and_Split": preprocessing,
                "Saved_Model_Weights": "Not present",
                "Saved_Training_History": "Not present",
            }
            for method, details in profiles.items()
        ]
    )


def test_sample_alignment(artifacts: list[dict]) -> pd.DataFrame:
    rows = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for artifact in artifacts:
        groups.setdefault((artifact["Dataset"], artifact["Target"]), []).append(artifact)
    for (dataset, target), group in groups.items():
        supports = [item["Matrix"].sum(axis=1) for item in group]
        same_supports = bool(
            supports
            and all(np.array_equal(supports[0], candidate) for candidate in supports[1:])
        )
        rows.append(
            {
                "Dataset": dataset,
                "Target": target,
                "Method_Count": len(group),
                "Test_Sample_Counts": ";".join(
                    str(int(item["Matrix"].sum())) for item in group
                ),
                "Actual_Class_Supports_Match": same_supports,
                "Evidence": (
                    "Original training script passes the same y_test and test-loader to each method in a target suite; saved row-support vectors match."
                    if same_supports
                    else "Saved class supports differ; test-sample identity cannot be verified."
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_availability() -> pd.DataFrame:
    available = {
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "Weighted F1",
        "Micro F1",
        "Balanced Accuracy",
        "MCC",
        "Cohen’s Kappa",
        "Sensitivity",
        "Specificity",
        "Top-1 Accuracy",
    }
    names = (
        "Accuracy", "Precision", "Recall", "F1-score", "Macro F1", "Weighted F1",
        "Micro F1", "Balanced Accuracy", "MCC", "Cohen’s Kappa", "Sensitivity",
        "Specificity", "ROC-AUC", "PR-AUC", "Top-1 Accuracy", "Top-2 Accuracy",
        "Top-3 Accuracy", "MAP@K", "MRR", "Log Loss", "Brier Score", "ECE",
        "Training time", "Inference time", "Memory consumption", "Convergence performance",
        "Generalization gap", "Cross-validation scores", "Mean and SD across random seeds",
        "0% noise robustness", "10% noise robustness", "20% noise robustness",
        "30% noise robustness",
    )
    rows = []
    for name in names:
        is_available = name in available
        rows.append(
            {
                "Metric": name,
                "Status": "Computed from saved confusion matrices" if is_available else "Unavailable",
                "Reason": (
                    "Confusion matrices contain sufficient hard-label counts."
                    if is_available
                    else "The current saved artifacts do not contain the scores, timing, memory, histories, folds, seeds, or noise-condition outputs required."
                ),
            }
        )
    rows.extend(
        [
            {
                "Metric": "Bootstrap 95% confidence intervals (accuracy, macro/weighted F1, balanced accuracy)",
                "Status": "Computed",
                "Reason": "Stratified bootstrap from saved confusion-matrix rows; uncertainty is not paired across methods.",
            },
            {
                "Metric": "Paired t-tests / Wilcoxon / Friedman / McNemar",
                "Status": "Unavailable",
                "Reason": "Per-sample paired predictions or repeated fold/seed scores are not saved.",
            },
        ]
    )
    return pd.DataFrame(rows)


def graph_availability() -> pd.DataFrame:
    available = {
        "Confusion matrix heatmaps",
        "Normalized confusion matrix heatmaps",
        "Class-wise precision, recall and F1 graphs",
        "Class distribution graphs",
        "Feature distribution histograms",
        "Box plots",
        "Violin plots",
        "Pair plots",
        "PCA 2D visualizations",
        "t-SNE 2D visualizations",
        "Correlation heatmaps",
        "Class-wise model comparison bar charts",
        "Model performance comparison bar charts",
        "Fertilizer class distribution graphs",
    }
    requests = (
        "Training loss vs epochs", "Validation loss vs epochs",
        "Training accuracy vs epochs", "Validation accuracy vs epochs",
        "Training vs validation loss", "Training vs validation accuracy",
        "Learning curves", "One-vs-rest / macro / micro ROC curves",
        "Precision-recall / macro / micro curves", "DET curves", "CAP curves",
        "Cumulative gain curves", "Lift curves", "Confusion matrix heatmaps",
        "Normalized confusion matrix heatmaps", "Calibration and reliability diagrams",
        "SHAP summary plots", "SHAP feature importance plots", "SHAP dependence plots",
        "SHAP waterfall plots", "LIME explanation plots", "Feature importance graphs",
        "Feature interaction plots", "Correlation heatmaps", "Feature distribution histograms",
        "Box plots", "Violin plots", "Pair plots", "PCA 2D visualizations",
        "t-SNE 2D visualizations", "UMAP visualizations", "Fertilizer class distribution graphs",
        "Class-wise precision, recall and F1 graphs", "Prediction error distributions",
        "Noise robustness curves", "Accuracy vs noise level", "Macro F1 vs noise level",
        "Performance degradation curves", "Feature sensitivity plots",
        "Prediction confidence distributions", "Prediction entropy distributions",
        "Model performance comparison bar charts", "Model performance comparison box plots",
        "Training time comparisons", "Inference time comparisons",
    )
    rows = []
    for request in requests:
        status = "Generated" if request in available else "Unavailable"
        if status == "Generated":
            reason = "Derived from saved confusion matrices or source-dataset features."
        elif "box plot" in request.lower() and "comparison" in request.lower():
            reason = "Only one saved run per method is available; a performance distribution requires repeated runs."
        elif "umap" in request.lower():
            reason = "UMAP is not installed and no existing UMAP coordinates are saved."
        elif "epoch" in request.lower() or "learning curve" in request.lower():
            reason = "Training histories are not saved; the original training code returns no epoch history."
        elif any(token in request.lower() for token in ("roc", "precision-recall", "det", "cap", "gain", "lift", "calibration", "confidence", "entropy")):
            reason = "Per-sample class probabilities or ranking scores are not saved."
        elif "noise" in request.lower() or "degradation" in request.lower():
            reason = "No predictions from the requested noise conditions are saved."
        elif any(token in request.lower() for token in ("shap", "lime", "importance", "interaction", "sensitivity")):
            reason = "Fitted estimators and per-row explanations are not saved."
        elif "time" in request.lower():
            reason = "Per-method training and inference timing was not saved in the available result tables."
        elif "error distribution" in request.lower():
            reason = "Per-row predictions are not saved; aggregate confusion counts cannot recover individual errors."
        else:
            reason = "The saved artifacts do not contain the scores or repeated results required."
        rows.append({"Requested_Graph": request, "Status": status, "Reason": reason})
    return pd.DataFrame(rows)


def write_report(
    artifacts: list[dict],
    metrics: pd.DataFrame,
    metric_status: pd.DataFrame,
    graph_status: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    if len(consistency):
        matched_line = (
            f"- Existing aggregate metrics matched to recomputed confusion-matrix results at their "
            f"saved four-decimal precision: **{int(consistency['Matches_At_Saved_Precision'].sum())}/{len(consistency)}**."
        )
    else:
        matched_line = "- No existing aggregate metric rows could be joined to the confusion-matrix results."
    lines = [
        "# Existing-results evaluation",
        "",
        "## Scope and evidence",
        "",
        "This evaluation consumed the saved confusion matrices and aggregate metric CSVs under `analysis_outputs/`, plus the original source datasets. It did not retrain or modify any model, dataset, hyperparameter, architecture, or existing result.",
        "",
        "The three implemented classifiers in `analyze_crop_fertilizer.py` are:",
        "",
        "- **1D-CNN Model:** Conv1d (1→32→64, kernel size 3) with batch normalization/ReLU and adaptive pooling, followed by a 64-unit dense layer, dropout 0.3, and class logits.",
        "- **Dual-Branch Fusion Net:** the encoded feature vector is split into a Conv1d branch (16 channels, adaptive pooling) and a 16-unit MLP branch; concatenated features feed a 32-unit fusion layer with dropout 0.2 and class logits.",
        "- **Gradient Boosting Ensemble:** scikit-learn `GradientBoostingClassifier`, 70 estimators, max depth 4, learning rate 0.08, seed 42.",
        "",
        "The original training script uses median imputation and standardization for numeric features, one-hot encoding with unknown categories ignored, a stratified 70/30 split (seed 42), and 25%-of-feature-standard-deviation Gaussian noise on numeric inputs. The noise is applied before the split. Neural models use 150 epochs, Adam (learning rate 0.003), batch size 32, and cross-entropy loss. The gradient-boosting implementation is not epoch based.",
        "",
        "## Artifact limitations",
        "",
        f"- Evaluated saved confusion matrices: **{len(artifacts)}** (three methods across three target tasks).",
        "- Within each target task, the source script evaluates all three methods on the same held-out split; saved per-class test supports were checked and match. Different target tasks use separate stratified splits.",
        "- Raw per-row predictions, class probability scores, fitted model weights, training histories, validation histories, repeated seeds, cross-validation folds, and 0/10/20/30% noise-condition outputs are not present in the workspace.",
        "- The source script temporarily constructs prediction/probability arrays but does not persist them. It trains the neural models without a validation set and returns no epoch history. No epoch curves, probability/ranking curves, calibration metrics, explainability plots, paired significance tests, or noise/seed/CV comparisons can therefore be derived from the saved evidence.",
        "- The existing summary metrics are rounded to four decimal places; the consistency audit compares at that precision.",
        "- Confusion-matrix bootstrap intervals below are stratified by actual class and are unpaired. They do not substitute for paired model comparisons.",
        "- PCA/t-SNE, feature distributions, pair plots, and correlations are descriptive projections of source data, not held-out model performance or causal explanations.",
        "",
        "## Evaluation summary",
        "",
        f"- Number of saved classifier/task evaluations: **{len(metrics)}**.",
        matched_line,
        "- Dataset details and exact model/preprocessing specifications are exported in `quantitative_metrics/dataset_profiles.csv` and `quantitative_metrics/methodology_profiles.csv`.",
        "- Full metric and graph availability is documented in `quantitative_metrics/metric_availability.csv` and `quantitative_metrics/graph_availability.csv`.",
        "",
        "## Output structure",
        "",
        "- `methodologies/`: per-method confusion matrices, normalized confusion matrices, and class-wise metric plots.",
        "- `quantitative_metrics/`: CSV/Excel tables, metric availability, bootstrap intervals, and source consistency checks.",
        "- `qualitative_graphs/`: descriptive source-data feature and class visualizations.",
        "- `comparative_visualizations/`: model comparison bar charts by target task.",
        "- `classification_curves/`, `explainability_visualizations/`, `robustness_graphs/`, `statistical_analysis/`, `ablation_studies/`: status tables documenting which requested analyses cannot be supported by present artifacts.",
        "",
        "Every generated figure is saved as both 300-DPI PNG and PDF.",
    ]
    (OUTPUT_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_output()
    plt.rcParams.update(PLOT_STYLE)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    artifacts = load_saved_confusion_matrices()
    metric_rows = []
    class_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for artifact in artifacts:
        matrix = artifact["Matrix"]
        metrics = metrics_from_matrix(matrix)
        metric_rows.append(
            {
                "Dataset_Key": artifact["Dataset_Key"],
                "Dataset": artifact["Dataset"],
                "Target": artifact["Target"],
                "Model": artifact["Model"],
                "Test_Samples": int(matrix.sum()),
                "Class_Count": len(artifact["Classes"]),
                **metrics,
            }
        )
        class_frame = class_metrics_from_matrix(matrix, artifact["Classes"])
        for column, value in {
            "Dataset_Key": artifact["Dataset_Key"],
            "Dataset": artifact["Dataset"],
            "Target": artifact["Target"],
            "Model": artifact["Model"],
        }.items():
            class_frame[column] = value
        class_rows.extend(class_frame.to_dict(orient="records"))
        for interval in bootstrap_intervals(matrix, rng):
            bootstrap_rows.append(
                {
                    "Dataset": artifact["Dataset"],
                    "Target": artifact["Target"],
                    "Model": artifact["Model"],
                    **interval,
                }
            )
        save_confusion_visualizations(artifact)

    metrics_frame = pd.DataFrame(metric_rows)
    classes_frame = pd.DataFrame(class_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    consistency_frame = compare_to_saved_summary(artifacts, metrics_frame)
    sample_alignment = test_sample_alignment(artifacts)
    metric_status = metric_availability()
    graph_status = graph_availability()
    dataset_graph_status, dataset_profiles = exploratory_dataset_plots()
    dataset_graph_status = pd.DataFrame(dataset_graph_status)
    methods = methodology_profiles()
    statistical_status = pd.DataFrame(
        [
            {
                "Test": test,
                "Status": "Unavailable",
                "Reason": reason,
            }
            for test, reason in (
                ("Paired t-test", "No per-sample paired correctness vectors or repeated runs."),
                ("Wilcoxon signed-rank", "No paired per-sample or per-fold metrics."),
                ("Friedman test", "No repeated fold/seed scores across methods."),
                ("McNemar test", "No paired per-sample predictions for discordant-pair counts."),
            )
        ]
    )
    ablation_status = pd.DataFrame(
        [
            {
                "Analysis": "Ablation study",
                "Status": "Not performed",
                "Reason": "No ablated model runs are saved; creating variants would require new experiments and would not evaluate the existing results.",
            }
        ]
    )

    metrics_frame.to_csv(OUTPUT_DIR / "quantitative_metrics" / "classification_metrics.csv", index=False)
    classes_frame.to_csv(OUTPUT_DIR / "quantitative_metrics" / "classwise_metrics.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT_DIR / "quantitative_metrics" / "bootstrap_95_percent_confidence_intervals.csv", index=False)
    consistency_frame.to_csv(OUTPUT_DIR / "quantitative_metrics" / "saved_metric_consistency.csv", index=False)
    sample_alignment.to_csv(OUTPUT_DIR / "quantitative_metrics" / "test_sample_alignment.csv", index=False)
    methods.to_csv(OUTPUT_DIR / "quantitative_metrics" / "methodology_profiles.csv", index=False)
    dataset_profiles.to_csv(OUTPUT_DIR / "quantitative_metrics" / "dataset_profiles.csv", index=False)
    metric_status.to_csv(OUTPUT_DIR / "quantitative_metrics" / "metric_availability.csv", index=False)
    graph_status.to_csv(OUTPUT_DIR / "quantitative_metrics" / "graph_availability.csv", index=False)
    dataset_graph_status.to_csv(OUTPUT_DIR / "quantitative_metrics" / "dataset_graph_status.csv", index=False)

    statistical_status.to_csv(OUTPUT_DIR / "statistical_analysis" / "test_availability.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT_DIR / "statistical_analysis" / "bootstrap_intervals.csv", index=False)
    ablation_status.to_csv(OUTPUT_DIR / "ablation_studies" / "ablation_status.csv", index=False)
    for model in MODEL_NAMES:
        model_rows = metrics_frame[metrics_frame["Model"] == model]
        class_rows_for_model = classes_frame[classes_frame["Model"] == model]
        method_dir = OUTPUT_DIR / "methodologies" / safe_filename(model)
        model_rows.to_csv(method_dir / "quantitative_metrics.csv", index=False)
        class_rows_for_model.to_csv(method_dir / "classwise_metrics.csv", index=False)
    for directory, status in (
        ("classification_curves", graph_status[graph_status["Requested_Graph"].isin(
            ["One-vs-rest / macro / micro ROC curves", "Precision-recall / macro / micro curves",
             "DET curves", "CAP curves", "Cumulative gain curves", "Lift curves",
             "Training loss vs epochs", "Validation loss vs epochs", "Learning curves"]
        )]),
        ("explainability_visualizations", graph_status[graph_status["Requested_Graph"].isin(
            ["SHAP summary plots", "SHAP feature importance plots", "SHAP dependence plots",
             "SHAP waterfall plots", "LIME explanation plots", "Feature importance graphs",
             "Feature interaction plots", "Feature sensitivity plots"]
        )]),
        ("robustness_graphs", graph_status[graph_status["Requested_Graph"].str.contains(
            "noise|degradation", case=False
        )]),
    ):
        status.to_csv(OUTPUT_DIR / directory / "availability.csv", index=False)

    save_comparative_plots(metrics_frame)

    workbook_tables = {
        "Metrics": metrics_frame,
        "Classwise": classes_frame,
        "Bootstrap_CI": bootstrap_frame,
        "Metric_Status": metric_status,
        "Graph_Status": graph_status,
        "Saved_Consistency": consistency_frame,
        "Statistical_Status": statistical_status,
        "Ablation_Status": ablation_status,
        "Methodology_Profile": methods,
        "Dataset_Profile": dataset_profiles,
        "Sample_Alignment": sample_alignment,
    }
    with pd.ExcelWriter(
        OUTPUT_DIR / "quantitative_metrics" / "evaluation_results.xlsx",
        engine="openpyxl",
    ) as writer:
        for sheet_name, dataframe in workbook_tables.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    width + 2, 48
                )

    write_report(
        artifacts,
        metrics_frame,
        metric_status,
        graph_status,
        consistency_frame,
    )
    figure_rows = [
        {
            "File": str(path.relative_to(OUTPUT_DIR)),
            "Format": path.suffix.removeprefix(".").upper(),
            "Resolution_DPI": 300 if path.suffix == ".png" else "vector",
        }
        for path in sorted(OUTPUT_DIR.rglob("*"))
        if path.suffix in {".png", ".pdf"}
    ]
    pd.DataFrame(figure_rows).to_csv(
        OUTPUT_DIR / "quantitative_metrics" / "figure_index.csv", index=False
    )
    print(f"Evaluated {len(artifacts)} saved confusion matrices without retraining.")
    print(f"Generated outputs: {OUTPUT_DIR}")
    print(f"Generated figures: {len(list(OUTPUT_DIR.rglob('*.png')))} PNG / "
          f"{len(list(OUTPUT_DIR.rglob('*.pdf')))} PDF")


if __name__ == "__main__":
    main()
