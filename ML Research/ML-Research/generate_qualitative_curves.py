"""
Generate qualitative analysis curves for 3 research methods.

Produces 4 curve plots, each containing all 3 methods overlaid:
  1. ROC Curve           (Macro-average One-vs-Rest)
  2. Precision-Recall Curve  (Macro-average)
  3. Calibration Curve   (Reliability Diagram)
  4. Recall-Threshold Curve (Macro-average One-vs-Rest)

Each plot is saved as both PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import os, re, sys, time, warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Import helpers from main script ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_crop_fertilizer import (
    load_and_prep,
    resolve_dataset_path,
    preprocess_features,
    apply_real_world_sensor_noise,
    Paper1_1DCNN,
    Paper2_DualBranchFusion,
)

# ── Configuration ────────────────────────────────────────────
SEED = 42
EPOCHS = 150
LEARNING_RATE = 0.003
BATCH_SIZE = 32
REALISTIC_SENSOR_NOISE = 0.25

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Research_Evaluation_Results" / "classification_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SPECS = (
    {
        "key": "Dataset_1_Crop_Fertilizer",
        "name": "Dataset 1 (Crop & Fertilizer)",
        "file_names": (
            "Crop and fertilizer dataset_3.csv",
            "Crop and fertilizer dataset_2.csv",
            "Crop and fertilizer dataset.csv",
        ),
        "targets": ("Crop", "Fertilizer"),  # Both targets
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

MODEL_SPECS = (
    {"name": "1D-CNN Model",            "paper": "Paper 1", "color": "#2196F3", "ls": "-"},
    {"name": "Dual-Branch Fusion Net",   "paper": "Paper 2", "color": "#FF5722", "ls": "--"},
    {"name": "Gradient Boosting Ensemble","paper": "Paper 3", "color": "#4CAF50", "ls": "-."},
)


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Train models & collect probabilities ─────────────────────
def train_pytorch_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]")

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for bx, _ in test_loader:
            out = model(bx)
            preds.append(torch.argmax(out, dim=1).numpy())
            probs.append(torch.softmax(out, dim=1).numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_models_for_dataset(csv_path, target_col):
    """Return dict: model_name -> (y_test, y_pred, y_prob, classes)."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y = load_and_prep(csv_path, target_col)
    X_noisy = apply_real_world_sensor_noise(X, noise_level=REALISTIC_SENSOR_NOISE)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y_enc, test_size=0.30, random_state=SEED, stratify=y_enc,
    )
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    t_X_tr = torch.tensor(X_train_proc, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train, dtype=torch.long)
    t_X_te = torch.tensor(X_test_proc, dtype=torch.float32)
    t_y_te = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(t_X_te, t_y_te), batch_size=64, shuffle=False)

    results = {}

    # Paper 1: 1D-CNN
    print("  Training 1D-CNN Model ...")
    torch.manual_seed(SEED)
    m1 = Paper1_1DCNN(X_train_proc.shape[1], num_classes)
    p1, prob1 = train_pytorch_model(m1, train_loader, test_loader)
    results["1D-CNN Model"] = (y_test, p1, prob1, le.classes_)

    # Paper 2: Dual-Branch Fusion Net
    print("  Training Dual-Branch Fusion Net ...")
    torch.manual_seed(SEED)
    m2 = Paper2_DualBranchFusion(X_train_proc.shape[1], num_classes)
    p2, prob2 = train_pytorch_model(m2, train_loader, test_loader)
    results["Dual-Branch Fusion Net"] = (y_test, p2, prob2, le.classes_)

    # Paper 3: Gradient Boosting Ensemble
    print("  Training Gradient Boosting Ensemble ...")
    np.random.seed(SEED)
    gbc = GradientBoostingClassifier(
        n_estimators=70, max_depth=4, learning_rate=0.08, random_state=SEED,
    )
    gbc.fit(X_train_proc, y_train)
    p3 = gbc.predict(X_test_proc)
    prob3 = gbc.predict_proba(X_test_proc)
    results["Gradient Boosting Ensemble"] = (y_test, p3, prob3, le.classes_)

    return results


# ── Curve 1: ROC Curve (Macro-Average OVR) ───────────────────
def plot_roc_curves(all_results: dict, dataset_label: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    for spec in MODEL_SPECS:
        y_test, _, y_prob, classes = all_results[spec["name"]]
        n_classes = len(classes)
        y_bin = label_binarize(y_test, classes=np.arange(n_classes))

        # Macro-average ROC
        fpr_all, tpr_all = [], []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)

        # Aggregate: interpolate all to common FPR grid
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(
            all_fpr, mean_tpr,
            color=spec["color"], linestyle=spec["ls"], linewidth=2.2,
            label=f'{spec["name"]} (Macro AUC = {macro_auc:.4f})',
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — Macro-Average OVR\n{dataset_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    save_figure(fig, out_dir / f"{safe_filename(dataset_label)}_ROC_Curve")
    print("  [OK] ROC Curve saved")


# ── Curve 2: Precision-Recall Curve (Macro-Average) ─────────
def plot_precision_recall_curves(all_results: dict, dataset_label: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    for spec in MODEL_SPECS:
        y_test, _, y_prob, classes = all_results[spec["name"]]
        n_classes = len(classes)
        y_bin = label_binarize(y_test, classes=np.arange(n_classes))

        precision_all, recall_all = [], []
        for i in range(n_classes):
            prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            precision_all.append(prec_i)
            recall_all.append(rec_i)

        # Macro-average: interpolate to common recall grid
        all_recall = np.linspace(0, 1, 500)
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            # PR curves go from high recall to low; flip for interp
            mean_precision += np.interp(all_recall, recall_all[i][::-1], precision_all[i][::-1])
        mean_precision /= n_classes
        macro_ap = average_precision_score(y_bin, y_prob, average="macro")

        ax.plot(
            all_recall, mean_precision,
            color=spec["color"], linestyle=spec["ls"], linewidth=2.2,
            label=f'{spec["name"]} (Macro AP = {macro_ap:.4f})',
        )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — Macro-Average\n{dataset_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir / f"{safe_filename(dataset_label)}_Precision_Recall_Curve")
    print("  [OK] Precision-Recall Curve saved")


# ── Curve 3: Calibration Curve (Reliability Diagram) ────────
def plot_calibration_curves(all_results: dict, dataset_label: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    for spec in MODEL_SPECS:
        y_test, _, y_prob, classes = all_results[spec["name"]]
        # For calibration: use the max probability (confidence) vs correctness
        confidence = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_test).astype(float)

        prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=12, strategy="uniform")

        ax.plot(
            prob_pred, prob_true,
            color=spec["color"], linestyle=spec["ls"], linewidth=2.2, marker="o",
            markersize=5,
            label=f'{spec["name"]}',
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title(f"Calibration Curve — Reliability Diagram\n{dataset_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir / f"{safe_filename(dataset_label)}_Calibration_Curve")
    print("  [OK] Calibration Curve saved")


# ── Curve 4: Recall vs. Probability Threshold (Macro OVR) ───
def plot_recall_threshold_curves(all_results: dict, dataset_label: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    thresholds = np.linspace(0, 1, 101)

    for spec in MODEL_SPECS:
        y_test, _, y_prob, classes = all_results[spec["name"]]
        y_bin = label_binarize(y_test, classes=np.arange(len(classes))).astype(bool)
        positives = y_bin.sum(axis=0)
        recalls = np.zeros((len(classes), len(thresholds)))

        for class_index in range(len(classes)):
            if positives[class_index]:
                predicted_positive = y_prob[:, class_index, None] >= thresholds
                recalls[class_index] = (
                    (predicted_positive & y_bin[:, class_index, None]).sum(axis=0)
                    / positives[class_index]
                )

        macro_recall = recalls.mean(axis=0)
        ax.plot(
            thresholds, macro_recall,
            color=spec["color"], linestyle=spec["ls"], linewidth=2.2,
            label=f'{spec["name"]}',
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Probability Threshold", fontsize=12)
    ax.set_ylabel("Macro-Average Recall", fontsize=12)
    ax.set_title(f"Recall vs. Threshold — Macro One-vs-Rest\n{dataset_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir / f"{safe_filename(dataset_label)}_Recall_Threshold_Curve")
    print("  [OK] Recall-Threshold Curve saved")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("QUALITATIVE ANALYSIS - CLASSIFICATION CURVES FOR 3 METHODS")
    print("=" * 70)

    for ds in DATASET_SPECS:
        for target in ds["targets"]:
            dataset_label = f'{ds["name"]} - Target: {target}'
            tag = f'{ds["key"]}_{target}'
            sub_dir = OUTPUT_DIR / safe_filename(tag)
            sub_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 60)
            print(f"Dataset: {dataset_label}")
            print("-" * 60)

            csv_path = resolve_dataset_path(ds["file_names"])
            results = run_models_for_dataset(csv_path, target)

            # Generate four curves
            plot_roc_curves(results, dataset_label, sub_dir)
            plot_precision_recall_curves(results, dataset_label, sub_dir)
            plot_calibration_curves(results, dataset_label, sub_dir)
            plot_recall_threshold_curves(results, dataset_label, sub_dir)

    # Update availability CSV
    avail = pd.DataFrame([
        {"Requested_Graph": "One-vs-rest / macro / micro ROC curves",
         "Status": "Generated", "Reason": "Models retrained; macro-average OVR ROC plotted."},
        {"Requested_Graph": "Precision-recall / macro / micro curves",
         "Status": "Generated", "Reason": "Models retrained; macro-average PR curves plotted."},
        {"Requested_Graph": "Calibration / Reliability curves",
         "Status": "Generated", "Reason": "Models retrained; reliability diagrams plotted."},
        {"Requested_Graph": "Recall versus probability threshold curves",
         "Status": "Generated", "Reason": "Models retrained; macro-average one-vs-rest recall plotted across thresholds."},
    ])
    avail.to_csv(OUTPUT_DIR / "availability.csv", index=False)

    print("\n" + "=" * 70)
    print("ALL 4 QUALITATIVE CURVES GENERATED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
