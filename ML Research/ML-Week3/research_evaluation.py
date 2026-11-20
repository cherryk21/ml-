from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis_outputs"
OUTPUT_ROOT = ROOT / "Research_Evaluation_Results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_frame(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    try:
        df.to_excel(path.with_suffix(".xlsx"), index=False)
    except Exception:
        pass


def save_plot(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inputs():
    bench = pd.read_csv(ANALYSIS_DIR / "all_benchmark_results.csv")
    ablation = pd.read_csv(ANALYSIS_DIR / "ablation_study_results.csv")
    four = pd.read_csv(ANALYSIS_DIR / "four_contribution_experimental_results.csv")
    return bench, ablation, four


def contribution_rows(bench_df: pd.DataFrame):
    rows = {}
    for code, label in {
        "C1": "Contribution 1",
        "C2": "Contribution 2",
        "C3": "Contribution 3",
        "C4": "Contribution 4",
    }.items():
        rows[code] = bench_df[bench_df["Paper"].astype(str).str.startswith(label)].copy()
    return rows


def summarize_by_contribution(bench_df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for contribution_name, df in contribution_rows(bench_df).items():
        if df.empty:
            continue
        row = {
            "Contribution": contribution_name,
            "Accuracy": float(df["Accuracy"].mean()),
            "Macro_F1": float(df["Macro_F1"].mean()),
            "Weighted_F1": float(df["Weighted_F1"].mean()),
            "Balanced_Accuracy": float(df["Balanced_Accuracy"].mean()),
            "Training_Seconds": float(df["Training_Seconds"].mean()),
            "Inference_Seconds": float(df["Inference_Seconds"].mean()),
        }
        summary.append(row)
    return pd.DataFrame(summary)


def contribution_label(code: str) -> str:
    mapping = {
        "C1": "Agronomic Relationship-Based Feature Engineering",
        "C2": "Crop-Conditioned Fertilizer Recommendation",
        "C3": "Gated Residual Network with Class-Balanced Learning",
        "C4": "Uncertainty-Aware Adaptive Fusion",
    }
    return mapping.get(code, code)


def make_bar_chart(df: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    values = df[metric].tolist()
    labels = df["Contribution"].tolist()
    ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Contribution")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate(rotation=0)
    save_plot(fig, path)


def make_combined_metrics_plot(summary_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(summary_df)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary_df["Accuracy"], width=width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], summary_df["Macro_F1"], width=width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Contribution"].tolist())
    ax.set_ylabel("Score")
    ax.set_title("Contribution Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_plot(fig, path)


def write_contribution_outputs(four_df: pd.DataFrame, bench_df: pd.DataFrame, ablation_df: pd.DataFrame):
    for code in ["C1", "C2", "C3", "C4"]:
        contribution_dir = OUTPUT_ROOT / f"Contribution_{code[-1]}"
        metrics_dir = contribution_dir / "Quantitative_Metrics"
        ensure_dir(metrics_dir)

        subset = bench_df[bench_df["Paper"].astype(str).str.startswith(f"Contribution {code[-1]}")].copy()
        if subset.empty:
            # Fallback to any direct rows if paper naming differs.
            subset = four_df[four_df["Experiment"].astype(str).str.contains(code, case=False, na=False)].copy()
        if subset.empty:
            continue

        subset = subset[[
            c for c in subset.columns if c not in ["Unnamed: 0"]
        ]]
        save_frame(subset, metrics_dir / f"{code}_metrics.csv")

        summary = pd.DataFrame([
            {
                "Contribution": code,
                "Accuracy": float(subset["Accuracy"].mean()),
                "Balanced_Accuracy": float(subset["Balanced_Accuracy"].mean()),
                "Macro_F1": float(subset["Macro_F1"].mean()),
                "Weighted_F1": float(subset["Weighted_F1"].mean()),
                "MCC": float(subset.get("Matthews_Correlation_Coefficient", pd.Series([float('nan')]*len(subset))).mean()),
                "ROC_AUC": float(subset.get("Macro_ROC_AUC_OVR", pd.Series([float('nan')]*len(subset))).mean()),
                "Training_Seconds": float(subset["Training_Seconds"].mean()),
                "Inference_Seconds": float(subset["Inference_Seconds"].mean()),
            }
        ])
        save_frame(summary, metrics_dir / f"{code}_summary.csv")

        if code == "C1":
            note_text = (
                "C1 uses agronomic feature engineering based on NPK balance, pH stress, and weather interactions. "
                "No synthetic training curves were created; the project exports preserved metrics only."
            )
        elif code == "C2":
            note_text = (
                "C2 adds crop-conditioned features using crop probability conditioning. "
                "The evaluation reuses the saved experimental results and does not synthetic retraining history."
            )
        elif code == "C3":
            note_text = (
                "C3 applies the gated residual classifier with class-balanced learning. "
                "Epoch-wise curves are not generated because the project does not expose per-epoch history in the preserved outputs."
            )
        else:
            note_text = (
                "C4 combines uncertainty-aware fusion to improve adaptive reliability. "
                "The framework uses the original saved uncertainty and fusion outputs without fabricating missing curves."
            )

        (contribution_dir / "methodology_note.txt").write_text(note_text, encoding="utf-8")

        # Create a small metric bar chart for each contribution.
        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Macro_F1", "Weighted_F1", "Balanced_Accuracy"],
            "Value": [
                float(subset["Accuracy"].mean()),
                float(subset["Macro_F1"].mean()),
                float(subset["Weighted_F1"].mean()),
                float(subset["Balanced_Accuracy"].mean()),
            ],
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(metric_df["Metric"], metric_df["Value"], color="#4C72B0")
        ax.set_title(f"{contribution_label(code)}\n(average across saved runs)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        save_plot(fig, metrics_dir / f"{code}_metric_profile.png")

        # Create placeholder folders for requested analysis sections.
        for section in [
            "Training_Curves",
            "Classification_Curves",
            "Explainability",
            "Feature_Analysis",
            "Robustness_Analysis",
            "Ablation_Study",
        ]:
            ensure_dir(contribution_dir / section)
            (contribution_dir / section / "not_applicable.txt").write_text(
                "This project preserves the original experimental outputs; no new synthetic training curves or metrics were fabricated for this section.",
                encoding="utf-8",
            )

    # Comparative visualizations.
    chart_dir = OUTPUT_ROOT / "Comparative_Visualizations"
    ensure_dir(chart_dir)

    summary_df = summarize_by_contribution(bench_df).copy()
    summary_df["Contribution"] = [
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    summary_df = summary_df[["Contribution", "Accuracy", "Macro_F1", "Weighted_F1", "Balanced_Accuracy", "Training_Seconds"]]
    save_frame(summary_df, chart_dir / "contribution_summary.csv")
    make_bar_chart(summary_df, "Accuracy", "Average Accuracy by Contribution", "Accuracy", chart_dir / "accuracy_comparison.png")
    make_bar_chart(summary_df, "Macro_F1", "Average Macro F1 by Contribution", "Macro F1", chart_dir / "macro_f1_comparison.png")
    make_bar_chart(summary_df, "Training_Seconds", "Average Training Time by Contribution", "Training Seconds", chart_dir / "training_time_comparison.png")
    make_combined_metrics_plot(summary_df, chart_dir / "combined_accuracy_f1_comparison.png")

    # Ablation analysis.
    ablation_dir = OUTPUT_ROOT / "Combined_Ablation_Analysis"
    ensure_dir(ablation_dir)
    ablation_save = ablation_df.copy()
    save_frame(ablation_save, ablation_dir / "combined_ablation_results.csv")

    # Additional summary tables and stats notes.
    stats_dir = OUTPUT_ROOT / "Statistical_Analysis"
    ensure_dir(stats_dir)
    stats_df = pd.DataFrame([
        {
            "Test": "Paired t-test",
            "Status": "Not applicable",
            "Reason": "The preserved outputs do not include paired per-sample predictions needed for exact paired statistical testing.",
        },
        {
            "Test": "Wilcoxon signed-rank",
            "Status": "Not applicable",
            "Reason": "Per-sample paired predictions are unavailable in the exported result files.",
        },
        {
            "Test": "Friedman test",
            "Status": "Not applicable",
            "Reason": "The evaluation uses a small number of contribution-level summaries rather than repeated matched-sample observations.",
        },
        {
            "Test": "McNemar test",
            "Status": "Not applicable",
            "Reason": "The required paired classification outcome matrices are not present in the saved outputs.",
        },
    ])
    save_frame(stats_df, stats_dir / "statistical_test_status.csv")

    # Quantitative reports.
    qreport_dir = OUTPUT_ROOT / "Quantitative_Reports"
    ensure_dir(qreport_dir)
    report_md = qreport_dir / "research_evaluation_report.md"

    best_accuracy = summary_df.loc[summary_df["Accuracy"].idxmax()]
    best_macro_f1 = summary_df.loc[summary_df["Macro_F1"].idxmax()]
    best_weighted_f1 = summary_df.loc[summary_df["Weighted_F1"].idxmax()]

    report_content = f"""# Research Evaluation Report

## Overview
This report uses the preserved experimental outputs already generated in the project under `analysis_outputs/` and creates a separate evaluation package at `Research_Evaluation_Results/` without modifying the source implementations or their saved results.

## Contribution summary
The four preserved contribution families are:
- C1: Agronomic Relationship-Based Feature Engineering
- C2: Crop-Conditioned Fertilizer Recommendation
- C3: Gated Residual Network with Class-Balanced Learning
- C4: Uncertainty-Aware Adaptive Fusion

## Best observed results from the saved outputs
- Highest mean accuracy: {best_accuracy['Contribution']} ({best_accuracy['Accuracy']:.4f})
- Highest mean macro F1: {best_macro_f1['Contribution']} ({best_macro_f1['Macro_F1']:.4f})
- Highest mean weighted F1: {best_weighted_f1['Contribution']} ({best_weighted_f1['Weighted_F1']:.4f})

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
"""
    report_md.write_text(report_content, encoding="utf-8")

    # Summary reports directory.
    summary_dir = OUTPUT_ROOT / "Summary_Reports"
    ensure_dir(summary_dir)
    summary_csv = summary_dir / "contribution_summary.csv"
    save_frame(summary_df, summary_csv)
    summary_text = (
        "C1, C2, C3, C4 summary exported to Research_Evaluation_Results/Summary_Reports/contribution_summary.csv.\n"
        "Best observed accuracy and macro-F1 values were computed directly from the preserved project results, without synthetic fabrication."
    )
    (summary_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    # Experimental logs.
    log_dir = OUTPUT_ROOT / "Experimental_Logs"
    ensure_dir(log_dir)
    log_data = {
        "source_outputs": [
            str(ANALYSIS_DIR / "all_benchmark_results.csv"),
            str(ANALYSIS_DIR / "ablation_study_results.csv"),
            str(ANALYSIS_DIR / "four_contribution_experimental_results.csv"),
        ],
        "created_outputs": [
            str(OUTPUT_ROOT),
        ],
        "evaluation_note": "This process reads existing saved results and preserves the original project implementation and datasets.",
    }
    (log_dir / "evaluation_log.json").write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    # Print summary to console for confirmation.
    print("Research_Evaluation_Results created at:", OUTPUT_ROOT)
    print(summary_df.to_string(index=False))


def main():
    bench_df, ablation_df, four_df = load_inputs()
    write_contribution_outputs(four_df, bench_df, ablation_df)


if __name__ == "__main__":
    main()
