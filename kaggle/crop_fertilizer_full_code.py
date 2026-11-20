# Full Kaggle implementation. Copy each numbered code block into a separate cell,
# or paste the entire Python file into one code cell. Attach both CSVs first.

# %% [markdown]
# # Crop and fertilizer research — full Kaggle implementation
#
# All training code is included as editable Python cells. **Attach only the two CSVs** from `ML Research/ML-Week3/` as a Kaggle Dataset: `Crop and fertilizer dataset.csv` and `Crop_recommendation.csv`. Import this notebook and run top to bottom. Do not run the two implementations out of order: their definitions share the notebook namespace.
#
# By default both suites train on all three targets. Set `QUICK_RUN=True` for a 3-epoch neural-network sanity check on all three tasks; these are not reportable full-run scores. The three paper models are the repository's implementations, not claims of exact reproduction of external papers. The Week 3 cells implement C1–C4 as in the source runner, with C2 applicable only to fertilizer.

# %% [markdown]
# ## 1. Setup and load both datasets

# %% [Code cell 01]
from pathlib import Path
from tempfile import mkdtemp
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from IPython.display import display, FileLink, Image

# Keep object dtype inference compatible with the source scripts on pandas 3.
try:
    pd.set_option("future.infer_string", False)
except pd.errors.OptionError:
    pass

QUICK_RUN = False
INPUT_ROOT = Path("/kaggle/input")  # Set to a specific attached folder if names repeat.
WORK_ROOT = Path("/kaggle/working")
RUN_ROOT = Path(mkdtemp(prefix="crop-research-", dir=WORK_ROOT))


def locate_csv(name):
    hits = sorted(INPUT_ROOT.rglob(name))
    if len(hits) != 1:
        raise ValueError(f"Expected one {name!r}, found {hits}. Attach the CSVs or narrow INPUT_ROOT.")
    return hits[0]

DATASETS = {
    "Dataset 1 (Crop & Fertilizer)": locate_csv("Crop and fertilizer dataset.csv"),
    "Dataset 2 (Crop Recommendation)": locate_csv("Crop_recommendation.csv"),
}
TASKS = [(name, target) for name, targets in zip(DATASETS, [("Crop", "Fertilizer"), ("label",)])
         for target in targets]
FRAMES = {name: pd.read_csv(path) for name, path in DATASETS.items()}
for name, frame in FRAMES.items():
    print(name, frame.shape, list(frame.columns))
    display(frame.head(3))

# %% [markdown]
# ## 2. Dataset audit and exploratory plots

# %% [Code cell 02]
import matplotlib.pyplot as plt

ANALYSIS_ROOT = RUN_ROOT / "analysis"
ANALYSIS_ROOT.mkdir(exist_ok=True)
audit, class_counts = [], []
for name, target in TASKS:
    frame = FRAMES[name]
    count = frame[target].value_counts()
    audit.append({"Dataset": name, "Target": target, "Rows": len(frame),
                  "Classes": len(count), "Min_Class_Size": count.min(),
                  "Missing_Targets": frame[target].isna().sum(),
                  "Missing_Cells": frame.isna().sum().sum(),
                  "Duplicate_Rows": frame.duplicated().sum()})
    class_counts.extend({"Dataset": name, "Target": target, "Class": cls, "Rows": n}
                        for cls, n in count.items())
    fig, ax = plt.subplots(figsize=(8, 5))
    count.sort_values().plot.barh(ax=ax, title=f"{name} — {target} class support")
    ax.set_xlabel("Rows")
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / f"class_support_{target}.png", dpi=150)
    display(fig)
    plt.close(fig)

audit = pd.DataFrame(audit)
class_counts = pd.DataFrame(class_counts)
audit.to_csv(ANALYSIS_ROOT / "dataset_audit.csv", index=False)
class_counts.to_csv(ANALYSIS_ROOT / "dataset_class_support.csv", index=False)
display(audit)

for i, (name, frame) in enumerate(FRAMES.items(), 1):
    corr = frame.select_dtypes(include="number").corr()
    corr.to_csv(ANALYSIS_ROOT / f"dataset_{i}_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title(name + " — numeric feature correlations")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / f"dataset_{i}_correlations.png", dpi=150)
    display(fig)
    plt.close(fig)

# %% [markdown]
# ## 3. Three paper implementations
# Paper 1: 1D-CNN. Paper 2: dual-branch fusion network. Paper 3: gradient boosting. The source applies Gaussian noise to numeric inputs before the stratified 70/30 split. These cells require scikit-learn, PyTorch and openpyxl; Kaggle generally preinstalls them.

# %% [markdown]
# ### Imports and baseline configuration

# %% [Code cell 03]
import os
import re
import shutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl.worksheet.table import Table, TableStyleInfo

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EPOCHS = 3 if QUICK_RUN else 150
LEARNING_RATE = 0.003
BATCH_SIZE = 32

ROOT = RUN_ROOT / "papers"

OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_DIRS = (OUTPUT_DIR, ROOT / "outputs")

for output_dir in OUTPUT_DIRS:
    output_dir.mkdir(parents=True, exist_ok=True)

REALISTIC_SENSOR_NOISE = 0.25

ALL_RESULTS = []
ALL_CONFUSION_MATRICES = []
ALL_CLASS_METRICS = []
ALL_PREDICTIONS = []
ALL_DATASET_PROFILES = []
ALL_CLASS_DISTRIBUTIONS = []
ALL_FEATURE_PROFILES = []
ALL_NUMERIC_CORRELATIONS = []
ALL_DATA_QUALITY = []
ALL_QUALITATIVE_FINDINGS = []


# ============================================================
# DATASET PATH RESOLUTION
# ============================================================

# %% [markdown]
# ### Dataset loading, profiling, sensor noise, and preprocessing

# %% [Code cell 04]
def resolve_dataset_path(possible_names):

    for name in possible_names:

        p = Path(name)

        if p.exists():
            return p

        p_root = ROOT / name

        if p_root.exists():
            return p_root

        p_data = ROOT / "data" / name

        if p_data.exists():
            return p_data

    raise FileNotFoundError(
        f"Could not find dataset file. "
        f"Looked for {possible_names} in {ROOT}"
    )


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

def find_target_column(df, target_col):

    for c in df.columns:

        if c.strip().lower() == target_col.strip().lower():
            return c

    for c in df.columns:

        if (
            target_col.strip().lower() in c.strip().lower()
            or c.strip().lower() in target_col.strip().lower()
        ):
            return c

    raise ValueError(
        f"Target column '{target_col}' not found. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_and_prep(csv_path, target_col):

    df = pd.read_csv(csv_path)

    df.columns = df.columns.astype(str).str.strip()

    actual_target = find_target_column(df, target_col)

    drop_cols = [actual_target, "Link", "link"]

    if actual_target.lower() in ["crop", "label"]:

        drop_cols += [
            c for c in df.columns
            if "fertilizer" in c.lower()
        ]

    elif "fertilizer" in actual_target.lower():

        drop_cols += [
            c for c in df.columns
            if c.lower() in ["crop", "label"]
        ]

    X = df.drop(
        columns=[
            c for c in drop_cols
            if c in df.columns
        ],
        errors="ignore"
    )

    y = df[actual_target].astype(str)

    for col in X.columns:

        if X[col].dtype == "object":

            converted = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            if converted.notna().mean() > 0.8:
                X[col] = converted

    return X, y


def profile_dataset(csv_path, dataset_name, target_name, X, y):
    source = pd.read_csv(csv_path)
    source.columns = source.columns.astype(str).str.strip()
    source_target = find_target_column(source, target_name)
    class_counts = y.value_counts(dropna=False)
    class_probabilities = class_counts / class_counts.sum()
    entropy = float(-(class_probabilities * np.log2(class_probabilities)).sum())
    class_count = len(class_counts)
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    feature_rows = []
    outlier_total = 0

    for column in X.columns:
        values = X[column]
        row = {
            "Dataset": dataset_name,
            "Target": target_name,
            "Feature": column,
            "Data_Type": str(values.dtype),
            "Unique_Values": int(values.nunique(dropna=True)),
            "Missing_Count": int(values.isna().sum()),
            "Missing_Percentage": round(float(values.isna().mean() * 100), 4),
            "Constant_Feature": bool(values.nunique(dropna=True) <= 1),
        }
        if column in numeric_columns:
            quantiles = values.quantile([0.25, 0.5, 0.75])
            iqr = quantiles.loc[0.75] - quantiles.loc[0.25]
            lower = quantiles.loc[0.25] - 1.5 * iqr
            upper = quantiles.loc[0.75] + 1.5 * iqr
            outlier_count = int(((values < lower) | (values > upper)).sum())
            outlier_total += outlier_count
            row.update({
                "Mean": values.mean(),
                "Standard_Deviation": values.std(),
                "Minimum": values.min(),
                "Q1": quantiles.loc[0.25],
                "Median": quantiles.loc[0.5],
                "Q3": quantiles.loc[0.75],
                "Maximum": values.max(),
                "Skewness": values.skew(),
                "IQR_Lower_Bound": lower,
                "IQR_Upper_Bound": upper,
                "IQR_Outlier_Count": outlier_count,
            })
        else:
            mode = values.mode(dropna=True)
            row.update({
                "Most_Frequent_Value": mode.iloc[0] if not mode.empty else "",
                "Most_Frequent_Count": int(values.value_counts(dropna=True).iloc[0])
                if values.notna().any() else 0,
            })
        feature_rows.append(row)

    conflicting_groups = 0
    if len(X.columns):
        labels_per_feature_group = pd.concat(
            [X.reset_index(drop=True), y.rename(target_name).reset_index(drop=True)],
            axis=1,
        ).groupby(
            X.columns.tolist(),
            dropna=False,
            observed=True,
        )[target_name].nunique()
        conflicting_groups = int((labels_per_feature_group > 1).sum())

    profile = {
        "Dataset": dataset_name,
        "Target": target_name,
        "Source_File": Path(csv_path).name,
        "Rows": len(source),
        "Feature_Count": len(X.columns),
        "Numeric_Feature_Count": len(numeric_columns),
        "Categorical_Feature_Count": len(categorical_columns),
        "Target_Class_Count": class_count,
        "Target_Entropy_Bits": round(entropy, 6),
        "Normalized_Target_Entropy": round(entropy / np.log2(class_count), 6)
        if class_count > 1 else 0.0,
        "Majority_Class": str(class_counts.index[0]),
        "Majority_Class_Count": int(class_counts.iloc[0]),
        "Minority_Class": str(class_counts.index[-1]),
        "Minority_Class_Count": int(class_counts.iloc[-1]),
        "Class_Imbalance_Ratio": round(
            float(class_counts.iloc[0] / class_counts.iloc[-1]), 6
        ),
    }
    data_quality = {
        "Dataset": dataset_name,
        "Target": target_name,
        "Source_Rows": len(source),
        "Source_Columns": len(source.columns),
        "Missing_Cells": int(source.isna().sum().sum()),
        "Rows_With_Missing_Values": int(source.isna().any(axis=1).sum()),
        "Duplicate_Rows": int(source.duplicated().sum()),
        "Missing_Target_Values": int(source[source_target].isna().sum()),
        "Constant_Feature_Count": sum(row["Constant_Feature"] for row in feature_rows),
        "Conflicting_Feature_Groups": conflicting_groups,
        "Numeric_IQR_Outlier_Values": outlier_total,
        "Majority_Class_Baseline_Accuracy": round(
            float(class_counts.iloc[0] / class_counts.sum()), 6
        ),
    }

    ALL_DATASET_PROFILES.append(profile)
    ALL_DATA_QUALITY.append(data_quality)
    ALL_CLASS_DISTRIBUTIONS.extend(
        {
            "Dataset": dataset_name,
            "Target": target_name,
            "Class": str(class_name),
            "Count": int(count),
            "Percentage": round(float(count / class_counts.sum() * 100), 4),
        }
        for class_name, count in class_counts.items()
    )
    ALL_FEATURE_PROFILES.extend(feature_rows)

    numeric_frame = X[numeric_columns]
    if len(numeric_columns) > 1:
        corr = numeric_frame.corr(method="pearson")
        for i, first in enumerate(numeric_columns):
            for second in numeric_columns[i + 1:]:
                ALL_NUMERIC_CORRELATIONS.append({
                    "Dataset": dataset_name,
                    "Target": target_name,
                    "Feature_1": first,
                    "Feature_2": second,
                    "Pearson_Correlation": corr.loc[first, second],
                })


# ============================================================
# REAL-WORLD SENSOR NOISE SIMULATION
# ============================================================

def apply_real_world_sensor_noise(
    X,
    noise_level=REALISTIC_SENSOR_NOISE
):

    X_noisy = X.copy()

    num_cols = X_noisy.select_dtypes(
        include=np.number
    ).columns

    for col in num_cols:

        std = X_noisy[col].std()

        if std > 0:

            noise = np.random.normal(
                0,
                noise_level * std,
                size=len(X_noisy)
            )

            X_noisy[col] = X_noisy[col] + noise

    return X_noisy


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def preprocess_features(X_train, X_test):

    num_cols = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    cat_cols = [
        c for c in X_train.columns
        if c not in num_cols
    ]

    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        )
    ]

    if cat_cols:

        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                cat_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop"
    )

    X_tr_proc = preprocessor.fit_transform(X_train)

    X_te_proc = preprocessor.transform(X_test)

    return X_tr_proc, X_te_proc


# ============================================================
# PAPER 1: 1D CNN MODEL
# ============================================================

# %% [markdown]
# ### Paper 1 and Paper 2 architectures and training

# %% [Code cell 05]
class Paper1_1DCNN(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.conv_block = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(4)
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 4, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv_block(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)


# ============================================================
# PAPER 2: DUAL BRANCH DATA FUSION NETWORK
# ============================================================

class Paper2_DualBranchFusion(nn.Module):

    def __init__(self, total_dim, num_classes):

        super().__init__()

        self.split_idx = max(1, total_dim // 2)

        self.branch1_cnn = nn.Sequential(

            nn.Conv1d(
                1,
                16,
                kernel_size=2,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(2)
        )

        self.branch2_mlp = nn.Sequential(

            nn.Linear(
                total_dim - self.split_idx,
                16
            ),

            nn.ReLU()
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                16 * 2 + 16,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(self, x):

        x1 = x[:, :self.split_idx].unsqueeze(1)

        x2 = x[:, self.split_idx:]

        out1 = self.branch1_cnn(x1).view(
            x1.size(0),
            -1
        )

        out2 = self.branch2_mlp(x2)

        fused = torch.cat(
            [out1, out2],
            dim=1
        )

        return self.fusion(fused)


# ============================================================
# PYTORCH MODEL TRAINING - 150 EPOCHS
# ============================================================

def train_pytorch_model(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    lr=LEARNING_RATE
):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    training_started = time.perf_counter()
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        for bx, by in train_loader:

            optimizer.zero_grad()

            out = model(bx)

            loss = criterion(out, by)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 25 == 0:

            avg_loss = epoch_loss / len(train_loader)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Training Loss: {avg_loss:.4f}"
            )

    training_seconds = time.perf_counter() - training_started
    model.eval()

    preds_list = []
    probabilities_list = []
    inference_started = time.perf_counter()

    with torch.no_grad():

        for bx, _ in test_loader:

            out = model(bx)

            preds_list.append(
                torch.argmax(out, dim=1).numpy()
            )
            probabilities_list.append(
                torch.softmax(out, dim=1).numpy()
            )

    inference_seconds = time.perf_counter() - inference_started
    return (
        np.concatenate(preds_list),
        np.concatenate(probabilities_list),
        training_seconds,
        inference_seconds,
    )


# ============================================================
# METRICS AND CONFUSION MATRIX
# ============================================================

# %% [markdown]
# ### Paper 3 and the common three-model training/evaluation loop

# %% [Code cell 06]
def record_metrics(
    paper_name,
    dataset_name,
    target_name,
    model_name,
    y_true,
    y_pred,
    class_names,
    probabilities,
    training_seconds,
    inference_seconds,
    sample_indices,
):
    labels = np.arange(len(class_names))
    probabilities = np.asarray(probabilities)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = matrix.sum() - true_positive - false_positive - false_negative
    specificity = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.zeros_like(true_negative, dtype=float),
        where=(true_negative + false_positive) != 0,
    )
    npv = np.divide(
        true_negative,
        true_negative + false_negative,
        out=np.zeros_like(true_negative, dtype=float),
        where=(true_negative + false_negative) != 0,
    )
    one_hot = np.eye(len(class_names), dtype=float)[np.asarray(y_true, dtype=int)]
    confidence = probabilities.max(axis=1)
    confidence_correct = (probabilities.argmax(axis=1) == np.asarray(y_true)).astype(float)
    calibration_error = 0.0
    bin_edges = np.linspace(0.0, 1.0, 16)
    for bin_index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (confidence >= lower) & (
            (confidence < upper) if bin_index < len(bin_edges) - 2 else (confidence <= upper)
        )
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(
                float(confidence_correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )

    metrics = {
        "Paper": paper_name,
        "Dataset": dataset_name,
        "Target": target_name,
        "Model": model_name,
        "Test_Samples": len(y_true),
        "Class_Count": len(class_names),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Micro_Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Micro_Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Micro_F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro_Specificity": float(np.mean(specificity)),
        "Weighted_Specificity": float(np.average(specificity, weights=supports)),
        "Macro_Negative_Predictive_Value": float(np.mean(npv)),
        "Matthews_Correlation_Coefficient": matthews_corrcoef(y_true, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
        "Hamming_Loss": hamming_loss(y_true, y_pred),
        "Macro_Jaccard": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_Jaccard": jaccard_score(y_true, y_pred, average="weighted", zero_division=0),
        "Top_2_Accuracy": float(np.mean(np.any(
            np.argsort(probabilities, axis=1)[:, -2:] == np.asarray(y_true)[:, None],
            axis=1,
        )))
        if len(class_names) > 1 else accuracy_score(y_true, y_pred),
        "Log_Loss": log_loss(y_true, probabilities, labels=labels),
        "Multiclass_Brier_Score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "Expected_Calibration_Error_15_Bins": calibration_error,
        "Macro_ROC_AUC_OVR": roc_auc_score(
            y_true, probabilities, labels=labels, multi_class="ovr", average="macro"
        ),
        "Weighted_ROC_AUC_OVR": roc_auc_score(
            y_true, probabilities, labels=labels, multi_class="ovr", average="weighted"
        ),
        "Macro_Average_Precision": average_precision_score(one_hot, probabilities, average="macro"),
        "Weighted_Average_Precision": average_precision_score(one_hot, probabilities, average="weighted"),
        "Training_Seconds": training_seconds,
        "Inference_Seconds": inference_seconds,
        "Inference_Milliseconds_Per_Sample": inference_seconds / len(y_true) * 1000,
    }
    metrics = {
        key: (round(float(value), 6) if isinstance(value, (np.number, float, int)) else value)
        for key, value in metrics.items()
    }
    ALL_RESULTS.append(metrics)
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(class_names, name="Actual"),
        columns=pd.Index(class_names, name="Predicted"),
    )
    ALL_CONFUSION_MATRICES.append({
        "Dataset": dataset_name,
        "Target": target_name,
        "Model": model_name,
        "Matrix": matrix_df,
    })

    class_frame = pd.DataFrame({
        "Paper": paper_name,
        "Dataset": dataset_name,
        "Target": target_name,
        "Model": model_name,
        "Class": [str(name) for name in class_names],
        "Support": supports,
        "Class_Percentage": supports / len(y_true) * 100,
        "Precision": precisions,
        "Recall_Sensitivity": recalls,
        "F1_Score": f1s,
        "Specificity": specificity,
        "Negative_Predictive_Value": npv,
        "True_Positive": true_positive,
        "False_Positive": false_positive,
        "False_Negative": false_negative,
        "True_Negative": true_negative,
    })
    ALL_CLASS_METRICS.extend(class_frame.to_dict(orient="records"))

    predictions = pd.DataFrame({
        "Paper": paper_name,
        "Dataset": dataset_name,
        "Target": target_name,
        "Model": model_name,
        "Source_Row_Index": np.asarray(sample_indices),
        "Actual_Label": [str(class_names[index]) for index in y_true],
        "Predicted_Label": [str(class_names[index]) for index in y_pred],
        "Correct": np.asarray(y_true) == np.asarray(y_pred),
        "Prediction_Confidence": confidence,
    })
    for class_index, class_name in enumerate(class_names):
        predictions[f"Probability_{safe_filename(class_name)}"] = probabilities[:, class_index]
    ALL_PREDICTIONS.extend(predictions.to_dict(orient="records"))

    class_frame = class_frame.sort_values(["Recall_Sensitivity", "Support"])
    weakest_class = class_frame.iloc[0]
    strongest_class = class_frame.iloc[-1]
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    common_error = None
    if off_diagonal.max() > 0:
        actual_index, predicted_index = np.unravel_index(
            int(np.argmax(off_diagonal)), off_diagonal.shape
        )
        common_error = (
            f'{class_names[actual_index]} is most often confused with '
            f'{class_names[predicted_index]} ({off_diagonal[actual_index, predicted_index]} cases)'
        )
    notes = [
        f"Best-recalled class: {strongest_class['Class']} "
        f"(recall {strongest_class['Recall_Sensitivity']:.3f}).",
        f"Weakest-recalled class: {weakest_class['Class']} "
        f"(recall {weakest_class['Recall_Sensitivity']:.3f}).",
    ]
    if common_error:
        notes.append(common_error + ".")
    notes.append(
        f"Macro-F1 is {metrics['Macro_F1']:.3f} versus weighted F1 "
        f"{metrics['Weighted_F1']:.3f}; a gap can indicate uneven class performance."
    )
    ALL_QUALITATIVE_FINDINGS.append({
        "Paper": paper_name,
        "Dataset": dataset_name,
        "Target": target_name,
        "Model": model_name,
        "Interpretation": " ".join(notes),
        "Most_Common_Confusion": common_error or "No off-diagonal errors.",
        "Weakest_Recall_Class": weakest_class["Class"],
        "Weakest_Class_Recall": weakest_class["Recall_Sensitivity"],
        "Best_Recall_Class": strongest_class["Class"],
        "Best_Class_Recall": strongest_class["Recall_Sensitivity"],
    })
    print(
        f"[{paper_name[:7]}] {model_name:30s} | "
        f"Accuracy: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro_F1']:.4f}"
    )
    return metrics


# ============================================================
# EXECUTE THREE PAPER METHODOLOGIES
# ============================================================

def execute_paper_suite(
    csv_path,
    dataset_title,
    target_col
):

    print("\n" + "=" * 85)

    print(
        f"RUNNING SUITE: {dataset_title} "
        f"| TARGET: {target_col}"
    )

    print("=" * 85)

    X, y = load_and_prep(
        csv_path,
        target_col
    )
    profile_dataset(csv_path, dataset_title, target_col, X, y)

    X_realistic = apply_real_world_sensor_noise(
        X,
        noise_level=REALISTIC_SENSOR_NOISE
    )

    le = LabelEncoder()

    y_enc = le.fit_transform(y)

    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(

        X_realistic,

        y_enc,

        test_size=0.30,

        random_state=SEED,

        stratify=y_enc
    )

    X_train_proc, X_test_proc = preprocess_features(
        X_train,
        X_test
    )

    t_X_tr = torch.tensor(
        X_train_proc,
        dtype=torch.float32
    )

    t_y_tr = torch.tensor(
        y_train,
        dtype=torch.long
    )

    t_X_te = torch.tensor(
        X_test_proc,
        dtype=torch.float32
    )

    t_y_te = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_loader = DataLoader(

        TensorDataset(
            t_X_tr,
            t_y_tr
        ),

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    test_loader = DataLoader(

        TensorDataset(
            t_X_te,
            t_y_te
        ),

        batch_size=64,

        shuffle=False
    )

    # --------------------------------------------------------
    # PAPER 1: 1D-CNN
    # --------------------------------------------------------

    print("\nTraining Paper 1: 1D-CNN")

    torch.manual_seed(SEED)
    m1 = Paper1_1DCNN(
        X_train_proc.shape[1],
        num_classes
    )

    p1, p1_probabilities, p1_train_seconds, p1_inference_seconds = train_pytorch_model(
        m1,
        train_loader,
        test_loader,
        epochs=EPOCHS
    )

    record_metrics(

        "Paper 1 - 1D-CNN Soil Classifier",

        dataset_title,

        target_col,

        "1D-CNN Model",

        y_test,

        p1,

        le.classes_,
        p1_probabilities,
        p1_train_seconds,
        p1_inference_seconds,
        X_test.index,
    )

    # --------------------------------------------------------
    # PAPER 2: DUAL BRANCH FUSION
    # --------------------------------------------------------

    print("\nTraining Paper 2: Dual-Branch Fusion Network")

    torch.manual_seed(SEED)
    m2 = Paper2_DualBranchFusion(
        X_train_proc.shape[1],
        num_classes
    )

    p2, p2_probabilities, p2_train_seconds, p2_inference_seconds = train_pytorch_model(
        m2,
        train_loader,
        test_loader,
        epochs=EPOCHS
    )

    record_metrics(

        "Paper 2 - Sentinel Data Fusion Network",

        dataset_title,

        target_col,

        "Dual-Branch Fusion Net",

        y_test,

        p2,

        le.classes_,
        p2_probabilities,
        p2_train_seconds,
        p2_inference_seconds,
        X_test.index,
    )

    # --------------------------------------------------------
    # PAPER 3: GRADIENT BOOSTING
    # --------------------------------------------------------

    print("\nTraining Paper 3: Gradient Boosting Ensemble")

    gbc = GradientBoostingClassifier(

        n_estimators=70,

        max_depth=4,

        learning_rate=0.08,

        random_state=SEED
    )

    training_started = time.perf_counter()
    gbc.fit(
        X_train_proc,
        y_train
    )
    p3_train_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    p3 = gbc.predict(
        X_test_proc
    )
    p3_probabilities = gbc.predict_proba(X_test_proc)
    p3_inference_seconds = time.perf_counter() - inference_started

    record_metrics(

        "Paper 3 - Indian Agricultural Soil Classifier",

        dataset_title,

        target_col,

        "Gradient Boosting Ensemble",

        y_test,

        p3,

        le.classes_,
        p3_probabilities,
        p3_train_seconds,
        p3_inference_seconds,
        X_test.index,
    )


# ============================================================
# SAFE FILE NAMING
# ============================================================

# %% [markdown]
# ### Baseline reports and export helpers

# %% [Code cell 07]
def safe_filename(value):

    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value)
    ).strip("_")


# ============================================================
# QUANTITATIVE SUMMARY
# ============================================================

def build_quantitative_summary(results):
    summary = []
    metric_columns = [
        column
        for column in results.select_dtypes(include=np.number).columns
        if column not in {"Test_Samples", "Class_Count"}
    ]
    for (dataset, target), group in results.groupby(
        ["Dataset", "Target"],
        sort=False
    ):
        best = group.loc[group["Macro_F1"].idxmax()]
        row = {
            "Dataset": dataset,
            "Target": target,
            "Model_Count": len(group),
            "Test_Samples": int(group["Test_Samples"].iloc[0]),
            "Class_Count": int(group["Class_Count"].iloc[0]),
            "Best_Model_by_Macro_F1": best["Model"],
            "Best_Macro_F1": best["Macro_F1"],
        }
        for metric in metric_columns:
            row[f"Mean_{metric}"] = round(float(group[metric].mean()), 6)
            row[f"Std_{metric}"] = round(float(group[metric].std(ddof=0)), 6)
        summary.append(row)
    return pd.DataFrame(summary)


# ============================================================
# EXCEL WORKBOOK GENERATION
# ============================================================

def write_results_workbook(
    results,
    path,
    sheet_name="Results",
    table_name="ThreePapersResults"
):

    with pd.ExcelWriter(
        path,
        engine="openpyxl"
    ) as writer:

        results.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        worksheet = writer.sheets[sheet_name]

        worksheet.freeze_panes = "A2"

        worksheet.auto_filter.ref = worksheet.dimensions

        table = Table(
            displayName=table_name,
            ref=worksheet.dimensions
        )

        table.tableStyleInfo = TableStyleInfo(

            name="TableStyleMedium2",

            showFirstColumn=False,

            showLastColumn=False,

            showRowStripes=True,

            showColumnStripes=False
        )

        worksheet.add_table(table)

        for column_cells in worksheet.columns:

            column_letter = column_cells[0].column_letter

            width = max(
                len(str(cell.value or ""))
                for cell in column_cells
            )

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                width + 2,
                45
            )


def write_analysis_workbook(tables, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for index, (sheet_name, dataframe) in enumerate(tables.items(), start=1):
            dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.sheets[sheet_name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            table = Table(
                displayName=f"AnalysisTable{index}",
                ref=worksheet.dimensions,
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            worksheet.add_table(table)
            for column_cells in worksheet.columns:
                letter = column_cells[0].column_letter
                width = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[letter].width = min(width + 2, 45)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

def write_confusion_matrix_png(
    matrix,
    title,
    path
):

    class_count = len(matrix.index)

    figure_size = min(
        max(7, class_count * 0.45),
        16
    )

    fig, ax = plt.subplots(
        figsize=(figure_size, figure_size)
    )

    image = ax.imshow(
        matrix.to_numpy(),
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Count"
    )

    ax.set(

        title=title,

        xlabel="Predicted label",

        ylabel="Actual label",

        xticks=np.arange(class_count),

        yticks=np.arange(class_count)
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        matrix.index
    )

    if class_count <= 15:

        threshold = matrix.to_numpy().max() / 2

        for row in range(class_count):

            for column in range(class_count):

                value = matrix.iat[
                    row,
                    column
                ]

                ax.text(

                    column,

                    row,

                    str(value),

                    ha="center",

                    va="center",

                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    )
                )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SAVE ALL ANALYSIS OUTPUTS
# ============================================================

def write_analysis_outputs(results):

    summary = build_quantitative_summary(
        results
    )

    rankings = results.copy()

    rankings["Rank_by_Macro_F1"] = (

        rankings.groupby(
            ["Dataset", "Target"]
        )["Macro_F1"]

        .rank(
            method="min",
            ascending=False
        )

        .astype(int)
    )

    rankings = rankings.sort_values(

        [
            "Dataset",
            "Target",
            "Rank_by_Macro_F1"
        ]
    )

    comparison_columns = [

        "Dataset",

        "Target",

        "Paper",

        "Model",

        "Accuracy",

        "Macro_Precision",

        "Macro_Recall",

        "Macro_F1",

        "Rank_by_Macro_F1"
    ]

    comparison = rankings[
        comparison_columns
    ]

    for output_dir in OUTPUT_DIRS:

        results.to_csv(

            output_dir / "three_papers_realistic_results.csv",

            index=False
        )

        for workbook_name, dataframe, sheet_name, table_name in (

            (

                "three_papers_realistic_results.xlsx",

                results,

                "Results",

                "ThreePapersResults"
            ),

            (

                "three_papers_comparison.xlsx",

                comparison,

                "Comparison",

                "ThreePapersComparison"
            )
        ):

            workbook_path = (
                output_dir / workbook_name
            )

            if output_dir == OUTPUT_DIR:

                write_results_workbook(

                    dataframe,

                    workbook_path,

                    sheet_name=sheet_name,

                    table_name=table_name
                )

            else:

                shutil.copyfile(

                    OUTPUT_DIR / workbook_name,

                    workbook_path
                )

        comparison.to_csv(

            output_dir / "three_papers_comparison.csv",

            index=False
        )

        results.to_csv(

            output_dir / "model_metrics.csv",

            index=False
        )

        summary.to_csv(

            output_dir / "quantitative_analysis_summary.csv",

            index=False
        )

        rankings.to_csv(

            output_dir / "model_ranking_by_macro_f1.csv",

            index=False
        )

        metrics_dir = output_dir / "metrics"

        confusion_dir = (
            output_dir / "confusion_matrices"
        )

        metrics_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        confusion_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for _, row in results.iterrows():

            filename = "__".join(

                safe_filename(row[key])

                for key in (
                    "Dataset",
                    "Target",
                    "Model"
                )
            )

            row.to_frame().T.to_csv(

                metrics_dir / f"{filename}_metrics.csv",

                index=False
            )

        for item in ALL_CONFUSION_MATRICES:

            filename = "__".join(

                safe_filename(item[key])

                for key in (
                    "Dataset",
                    "Target",
                    "Model"
                )
            )

            item["Matrix"].to_csv(

                confusion_dir /
                f"{filename}_confusion_matrix.csv"
            )

            write_confusion_matrix_png(

                item["Matrix"],

                (
                    f'{item["Dataset"]} | '
                    f'{item["Target"]} | '
                    f'{item["Model"]}'
                ),

                confusion_dir /
                f"{filename}_confusion_matrix.png"
            )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 85)

    print("THREE RESEARCH PAPER COMPARISON")

    print("TRAINING EPOCHS:", EPOCHS)

    print("LEARNING RATE:", LEARNING_RATE)

    print("BATCH SIZE:", BATCH_SIZE)

    print("SENSOR NOISE:", REALISTIC_SENSOR_NOISE)

    print("=" * 85)

    ds1_path = resolve_dataset_path([

        "Crop and fertilizer dataset_3.csv",

        "Crop and fertilizer dataset_2.csv",

        "Crop and fertilizer dataset.csv"
    ])

    ds2_path = resolve_dataset_path([

        "Crop_recommendation_3.csv",

        "Crop_recommendation_2.csv",

        "Crop_recommendation.csv"
    ])

    # DATASET 1 - FERTILIZER CLASSIFICATION

    execute_paper_suite(

        ds1_path,

        "Dataset 1 (Crop & Fertilizer)",

        "Fertilizer"
    )

    # DATASET 1 - CROP CLASSIFICATION

    execute_paper_suite(

        ds1_path,

        "Dataset 1 (Crop & Fertilizer)",

        "Crop"
    )

    # DATASET 2 - CROP RECOMMENDATION

    execute_paper_suite(

        ds2_path,

        "Dataset 2 (Crop Recommendation)",

        "label"
    )

    # FINAL RESULTS

    df_res = pd.DataFrame(
        ALL_RESULTS
    )

    write_analysis_outputs(
        df_res
    )

    print("\n" + "=" * 85)

    print("FINAL EXPERIMENTAL SUMMARY")

    print("=" * 85)

    print(
        df_res.to_string(index=False)
    )

    print("\n" + "=" * 85)

    print("ALL RESULTS SAVED SUCCESSFULLY")

    print("OUTPUT DIRECTORY:", OUTPUT_DIR)

    print("BACKUP DIRECTORY:", OUTPUT_DIRS[1])

    print("=" * 85)

# %% [Code cell 08]
# Dataset 1: crop + fertilizer; Dataset 2: crop label.
for accumulator in (ALL_RESULTS, ALL_CONFUSION_MATRICES, ALL_CLASS_METRICS,
                    ALL_PREDICTIONS, ALL_DATASET_PROFILES, ALL_CLASS_DISTRIBUTIONS,
                    ALL_FEATURE_PROFILES, ALL_NUMERIC_CORRELATIONS, ALL_DATA_QUALITY,
                    ALL_QUALITATIVE_FINDINGS):
    accumulator.clear()
for dataset_name, task_target in TASKS:
    execute_paper_suite(DATASETS[dataset_name], dataset_name, task_target)
PAPER_RESULTS = pd.DataFrame(ALL_RESULTS)
PAPER_CLASS_METRICS = pd.DataFrame(ALL_CLASS_METRICS)
PAPER_PREDICTIONS = pd.DataFrame(ALL_PREDICTIONS)
write_analysis_outputs(PAPER_RESULTS)
PAPER_PREDICTIONS.to_csv(ANALYSIS_ROOT / "paper_holdout_predictions.csv", index=False)
PAPER_CLASS_METRICS.to_csv(ANALYSIS_ROOT / "paper_per_class_metrics.csv", index=False)
assert len(PAPER_RESULTS) == 9, "Expected 3 baselines × 3 tasks."
display(PAPER_RESULTS[["Dataset", "Target", "Model", "Accuracy", "Macro_F1"]])

# %% [markdown]
# ## 4. Week 3 implementation
# C1 adds agronomic features; C2 generates crop probabilities for fertilizer prediction; C3 uses gated residual blocks with class-balanced training; C4 selects a tree/network model or blend using validation macro F1 and reports uncertainty. XGBoost is optional in the source; available tree learners affect the candidates. Week 3 does **not** apply the declared sensor-noise setting. Macro F1 rankings must be interpreted within each protocol.

# %% [markdown]
# ### Week 3 imports, configuration and dataset utilities

# %% [Code cell 09]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    hamming_loss,
    jaccard_score,
    top_k_accuracy_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 3 if QUICK_RUN else 150
LEARNING_RATE = 0.003
BATCH_SIZE = 32
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20
SENSOR_NOISE = 0.25
MC_DROPOUT_PASSES = 30

OUTPUT_DIR = RUN_ROOT / "week3"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_1 = Path("Crop and fertilizer dataset.csv")
DATASET_2 = Path("Crop_recommendation.csv")

if os.environ.get("FERTILIZER_DATASET_1"):
    DATASET_1 = Path(os.environ["FERTILIZER_DATASET_1"])

if os.environ.get("FERTILIZER_DATASET_2"):
    DATASET_2 = Path(os.environ["FERTILIZER_DATASET_2"])

DEVICE = torch.device("cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()


# ============================================================
# FOUR PROPOSED CONTRIBUTIONS
# ============================================================

CONTRIBUTIONS = {
    "C1": "Agronomic Relationship-Based Feature Engineering",
    "C2": "Crop-Conditioned Fertilizer Recommendation",
    "C3": "Gated Residual Network with Class-Balanced Learning",
    "C4": "Uncertainty-Aware Adaptive Fusion"
}


# ============================================================
# DATASET UTILITIES
# ============================================================

def find_column(df, candidates):
    lookup = {
        str(column).strip().lower().replace(" ", "").replace("_", ""): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")

        if key in lookup:
            return lookup[key]

    return None


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    for column in df.select_dtypes(include="object").columns:

        df[column] = df[column].astype(str).str.strip()

        df[column] = df[column].replace(
            ["", "nan", "None", "null"], np.nan
        )

    return df


def get_features(df, target):
    excluded = [
        target,
        find_column(df, ["Link"]),
        find_column(df, ["Fertilizer Name"])
    ]

    excluded = [c for c in excluded if c is not None]

    X = df.drop(columns=excluded, errors="ignore")

    X = X.drop(
        columns=[
            c for c in X.columns
            if str(c).lower() in [
                "id", "index", "serial_no", "unnamed: 0"
            ]
        ],
        errors="ignore"
    )

    return X.copy()


# ============================================================
# C1: AGRONOMIC FEATURE ENGINEERING
# ============================================================

# %% [markdown]
# ### C1: agronomic features and train-fitted preprocessing

# %% [Code cell 10]
def contribution_C1(X):
    X = X.copy()

    n_col = find_column(X, ["N", "Nitrogen"])
    p_col = find_column(X, ["P", "Phosphorus"])
    k_col = find_column(X, ["K", "Potassium"])
    ph_col = find_column(X, ["ph", "pH", "Soil_pH"])

    temp_col = find_column(
        X, ["temperature", "Temperature", "temp"]
    )

    humidity_col = find_column(
        X, ["humidity", "Humidity"]
    )

    rainfall_col = find_column(
        X, ["rainfall", "Rainfall"]
    )

    moisture_col = find_column(
        X, ["moisture", "Moisture"]
    )

    if n_col and p_col and k_col:

        n = pd.to_numeric(X[n_col], errors="coerce")
        p = pd.to_numeric(X[p_col], errors="coerce")
        k = pd.to_numeric(X[k_col], errors="coerce")

        X["NPK_Total"] = n + p + k
        X["NPK_Mean"] = (n + p + k) / 3

        X["N_P_Ratio"] = n / (p.abs() + 1)
        X["N_K_Ratio"] = n / (k.abs() + 1)
        X["P_K_Ratio"] = p / (k.abs() + 1)

        X["N_P_Difference"] = n - p
        X["N_K_Difference"] = n - k
        X["P_K_Difference"] = p - k

        X["N_Deficit"] = np.maximum(100 - n, 0)
        X["P_Deficit"] = np.maximum(50 - p, 0)
        X["K_Deficit"] = np.maximum(50 - k, 0)

        X["NPK_Deficit_Total"] = (
            X["N_Deficit"]
            + X["P_Deficit"]
            + X["K_Deficit"]
        )

        X["NPK_Balance"] = (
            (n / 4 - p / 2) ** 2
            + (p / 2 - k) ** 2
        ) ** 0.5

    if ph_col:

        ph = pd.to_numeric(X[ph_col], errors="coerce")

        X["pH_Deviation"] = abs(ph - 6.5)
        X["Acidity_Stress"] = np.maximum(6.5 - ph, 0)
        X["Alkalinity_Stress"] = np.maximum(ph - 7.5, 0)

    if temp_col and humidity_col:

        temp = pd.to_numeric(X[temp_col], errors="coerce")

        humidity = pd.to_numeric(
            X[humidity_col], errors="coerce"
        ).clip(0, 100)

        denominator = temp + 237.3

        denominator = denominator.where(
            denominator.abs() > 1e-8, np.nan
        )

        saturation_vp = 0.61078 * np.exp(
            (17.27 * temp) / denominator
        )

        X["VPD"] = saturation_vp * (1 - humidity / 100)

        X["Temperature_Humidity"] = temp * humidity

    if temp_col and rainfall_col:

        temp = pd.to_numeric(X[temp_col], errors="coerce")

        rainfall = pd.to_numeric(
            X[rainfall_col], errors="coerce"
        )

        X["Temperature_Rainfall"] = temp * rainfall

    if moisture_col and rainfall_col:

        moisture = pd.to_numeric(
            X[moisture_col], errors="coerce"
        )

        rainfall = pd.to_numeric(
            X[rainfall_col], errors="coerce"
        )

        X["Moisture_Rainfall"] = moisture * rainfall

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_splits(X_train, X_val, X_test):

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    categorical = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for column in categorical:

        train_values = X_train[column].fillna(
            "Missing"
        ).astype(str)

        categories = {
            value: index
            for index, value in enumerate(
                train_values.unique()
            )
        }

        for frame in [X_train, X_val, X_test]:

            frame[column] = (
                frame[column]
                .fillna("Missing")
                .astype(str)
                .map(categories)
                .fillna(-1)
            )

    for frame in [X_train, X_val, X_test]:

        for column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    train = imputer.fit_transform(X_train)
    val = imputer.transform(X_val)
    test = imputer.transform(X_test)

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    return (
        train_scaled.astype(np.float32),
        val_scaled.astype(np.float32),
        test_scaled.astype(np.float32),
        imputer,
        scaler
    )


# ============================================================
# C2: CROP-CONDITIONED FERTILIZER RECOMMENDATION
# ============================================================

# %% [markdown]
# ### C2: crop-conditioned features with out-of-fold training predictions

# %% [Code cell 11]
def contribution_C2(
    X_train,
    X_val,
    X_test,
    crop_train
):

    crop_train = np.asarray(crop_train)

    classes = np.unique(crop_train)

    counts = pd.Series(crop_train).value_counts()

    if counts.min() < 2:
        raise ValueError(
            "C2 requires at least two training samples per crop class."
        )

    n_splits = min(5, int(counts.min()))

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    categorical = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for column in categorical:

        train_values = (
            X_train[column]
            .fillna("Missing")
            .astype(str)
        )

        categories = {
            value: index
            for index, value in enumerate(
                train_values.unique()
            )
        }

        for frame in [X_train, X_val, X_test]:

            frame[column] = (
                frame[column]
                .fillna("Missing")
                .astype(str)
                .map(categories)
                .fillna(-1)
            )

    for frame in [X_train, X_val, X_test]:

        for column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce"
            )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    oof_probabilities = np.zeros(
        (len(X_train), len(classes))
    )

    for fold, (fit_idx, hold_idx) in enumerate(
        splitter.split(X_train_imp, crop_train)
    ):

        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=-1
        )

        model.fit(
            X_train_imp[fit_idx],
            crop_train[fit_idx]
        )

        probabilities = model.predict_proba(
            X_train_imp[hold_idx]
        )

        for j, cls in enumerate(model.classes_):

            position = np.where(classes == cls)[0][0]

            oof_probabilities[
                hold_idx, position
            ] = probabilities[:, j]

    final_crop_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    final_crop_model.fit(
        X_train_imp,
        crop_train
    )

    val_probabilities = final_crop_model.predict_proba(
        X_val_imp
    )

    test_probabilities = final_crop_model.predict_proba(
        X_test_imp
    )

    def append_features(X, probabilities):

        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-12),
            axis=1
        )

        return np.column_stack([
            X,
            probabilities,
            entropy
        ]).astype(np.float32)

    return (
        append_features(X_train_imp, oof_probabilities),
        append_features(X_val_imp, val_probabilities),
        append_features(X_test_imp, test_probabilities),
        final_crop_model
    )


# ============================================================
# C3: GATED RESIDUAL NETWORK
# ============================================================

# %% [markdown]
# ### C3: gated residual network, class-balanced loss and training

# %% [Code cell 12]
class GatedResidualBlock(nn.Module):

    def __init__(self, dimension, dropout=0.25):

        super().__init__()

        self.gate = nn.Linear(
            dimension,
            dimension * 2
        )

        self.transform = nn.Linear(
            dimension,
            dimension
        )

        self.norm = nn.LayerNorm(dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        gated = F.glu(
            self.gate(x),
            dim=1
        )

        gated = self.dropout(gated)

        transformed = self.transform(gated)

        return F.relu(
            self.norm(transformed + residual)
        )


class GatedResidualNetwork(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.input_layer = nn.Linear(
            input_dim,
            128
        )

        self.block1 = GatedResidualBlock(128)
        self.block2 = GatedResidualBlock(128)

        self.output_layer = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = F.relu(
            self.input_layer(x)
        )

        x = self.block1(x)
        x = self.block2(x)

        return self.output_layer(x)


# ============================================================
# CLASS-BALANCED FOCAL LOSS
# ============================================================

class ClassBalancedFocalLoss(nn.Module):

    def __init__(self, class_weights, gamma=2.0):

        super().__init__()

        self.register_buffer(
            "weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

        self.gamma = gamma

    def forward(self, logits, targets):

        log_probs = F.log_softmax(
            logits,
            dim=1
        )

        probabilities = torch.exp(log_probs)

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = probabilities.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        weights = self.weights[targets]

        focal = (
            1 - target_probs
        ) ** self.gamma

        return (
            -weights * focal * target_log_probs
        ).mean()


def get_class_weights(y, num_classes):

    counts = np.bincount(
        y,
        minlength=num_classes
    ).astype(float)

    weights = 1 / np.sqrt(
        np.maximum(counts, 1)
    )

    return weights / weights.mean()


# ============================================================
# NEURAL TRAINING
# ============================================================

def train_network(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    epochs=EPOCHS
):

    set_seed()

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    model = GatedResidualNetwork(
        X_train.shape[1],
        num_classes
    )

    weights = get_class_weights(
        y_train,
        num_classes
    )

    criterion = ClassBalancedFocalLoss(
        weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15
    )

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    n = len(X_train_tensor)

    for epoch in range(epochs):

        model.train()

        permutation = torch.randperm(n)

        total_loss = 0

        for start in range(0, n, BATCH_SIZE):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            optimizer.step()

            total_loss += loss.item() * len(indices)

        train_loss = total_loss / n

        model.eval()

        with torch.no_grad():

            val_logits = model(X_val_tensor)

            val_loss = criterion(
                val_logits,
                y_val_tensor
            ).item()

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)

        history["validation_loss"].append(
            val_loss
        )

        if (epoch + 1) % 25 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Training Loss: {train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )

    return model, history

# %% [markdown]
# ### C4 helpers: uncertainty, trees and metrics

# %% [Code cell 13]
def neural_probabilities(
    model,
    X,
    mc_dropout=False
):

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    model.eval()

    if not mc_dropout:

        with torch.no_grad():

            probabilities = F.softmax(
                model(tensor),
                dim=1
            ).numpy()

        return probabilities, np.zeros(len(X))

    all_probabilities = []

    for module in model.modules():

        if isinstance(module, nn.Dropout):
            module.train()

    with torch.no_grad():

        for _ in range(MC_DROPOUT_PASSES):

            probabilities = F.softmax(
                model(tensor),
                dim=1
            ).numpy()

            all_probabilities.append(
                probabilities
            )

    all_probabilities = np.asarray(
        all_probabilities
    )

    mean_probabilities = all_probabilities.mean(
        axis=0
    )

    uncertainty = all_probabilities.var(
        axis=0
    ).mean(axis=1)

    model.eval()

    return mean_probabilities, uncertainty


# ============================================================
# BASELINE TREE MODELS FOR C4 FUSION
# ============================================================

def train_tree_models(
    X_train,
    y_train,
    X_val,
    X_test,
    num_classes
):

    models = {}

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1
        )

    validation_probabilities = {}
    test_probabilities = {}

    for name, model in models.items():

        print(
            f"Training C4 component: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        validation_probabilities[name] = (
            model.predict_proba(X_val)
        )

        test_probabilities[name] = (
            model.predict_proba(X_test)
        )

    return (
        validation_probabilities,
        test_probabilities
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    num_classes = probabilities.shape[1]

    metrics = {

        "Test_Samples": len(y_true),

        "Class_Count": num_classes,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),

        "Macro_Precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted_Precision": precision_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Micro_Precision": precision_score(
            y_true,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Macro_Recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted_Recall": recall_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Micro_Recall": recall_score(
            y_true,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Macro_F1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted_F1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Micro_F1": f1_score(
            y_true,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Matthews_Correlation_Coefficient": matthews_corrcoef(
            y_true,
            predictions
        ),

        "Cohen_Kappa": cohen_kappa_score(
            y_true,
            predictions
        ),

        "Hamming_Loss": hamming_loss(
            y_true,
            predictions
        ),

        "Macro_Jaccard": jaccard_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted_Jaccard": jaccard_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Top_2_Accuracy": top_k_accuracy_score(
            y_true,
            probabilities,
            k=min(2, num_classes),
            labels=np.arange(num_classes)
        ),

        "Log_Loss": log_loss(
            y_true,
            probabilities,
            labels=np.arange(num_classes)
        ),

        "Multiclass_Brier_Score": np.mean(
            np.sum(
                (
                    probabilities
                    - np.eye(num_classes)[y_true]
                ) ** 2,
                axis=1
            )
        )
    }

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(num_classes)
    )

    specificity = []
    npv = []

    for i in range(num_classes):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - tp - fn - fp

        specificity.append(
            tn / (tn + fp) if tn + fp else 0
        )

        npv.append(
            tn / (tn + fn) if tn + fn else 0
        )

    metrics["Macro_Specificity"] = np.mean(
        specificity
    )

    metrics["Weighted_Specificity"] = np.average(
        specificity,
        weights=np.bincount(
            y_true,
            minlength=num_classes
        )
    )

    metrics["Macro_Negative_Predictive_Value"] = np.mean(
        npv
    )

    confidence = probabilities.max(axis=1)

    correct = (
        predictions == y_true
    ).astype(float)

    ece = 0

    bins = np.linspace(0, 1, 16)

    for lower, upper in zip(
        bins[:-1],
        bins[1:]
    ):

        mask = (
            (confidence > lower)
            & (confidence <= upper)
        )

        if np.any(mask):

            ece += (
                np.mean(mask)
                * abs(
                    np.mean(correct[mask])
                    - np.mean(confidence[mask])
                )
            )

    metrics[
        "Expected_Calibration_Error_15_Bins"
    ] = ece

    try:

        binary = label_binarize(
            y_true,
            classes=np.arange(num_classes)
        )

        metrics["Macro_ROC_AUC_OVR"] = roc_auc_score(
            binary,
            probabilities,
            average="macro"
        )

        metrics["Weighted_ROC_AUC_OVR"] = roc_auc_score(
            binary,
            probabilities,
            average="weighted"
        )

        metrics["Macro_Average_Precision"] = average_precision_score(
            binary,
            probabilities,
            average="macro"
        )

        metrics["Weighted_Average_Precision"] = average_precision_score(
            binary,
            probabilities,
            average="weighted"
        )

    except Exception:

        metrics["Macro_ROC_AUC_OVR"] = np.nan
        metrics["Weighted_ROC_AUC_OVR"] = np.nan
        metrics["Macro_Average_Precision"] = np.nan
        metrics["Weighted_Average_Precision"] = np.nan

    return metrics, predictions


# ============================================================
# VISUALIZATION
# ============================================================

# %% [markdown]
# ### Plotting and classification report helpers

# %% [Code cell 14]
def save_confusion_matrix(
    y_true,
    probabilities,
    class_names,
    path,
    title
):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(len(class_names))
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(title)

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_xticks(
        np.arange(len(class_names))
    )

    ax.set_yticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=7
    )

    ax.set_yticklabels(
        class_names,
        fontsize=7
    )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=250
    )

    plt.close(fig)


def save_architecture_diagrams(output):

    diagrams = {

        "C1_Agronomic_Feature_Engineering": [
            "Original Soil Features",
            "N-P-K Relationships",
            "pH Stress Features",
            "Climate Interactions",
            "Engineered Feature Matrix"
        ],

        "C2_Crop_Conditioned_Recommendation": [
            "Soil and Climate Features",
            "Out-of-Fold Crop Prediction",
            "Crop Probability Features",
            "Fertilizer Feature Fusion",
            "Fertilizer Recommendation"
        ],

        "C3_Gated_Residual_Network": [
            "Input Feature Layer",
            "Dense Layer (128)",
            "Gated Residual Block 1",
            "Gated Residual Block 2",
            "Class-Balanced Focal Loss",
            "Output Classification"
        ],

        "C4_Uncertainty_Aware_Fusion": [
            "Gated Residual Network",
            "ExtraTrees",
            "Random Forest / XGBoost",
            "Uncertainty Estimation",
            "Validation-Based Fusion",
            "Final Prediction"
        ]
    }

    for name, stages in diagrams.items():

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        ax.set_xlim(0, 10)

        ax.set_ylim(0, len(stages) + 1)

        ax.axis("off")

        for i, stage in enumerate(stages):

            y = len(stages) - i

            ax.text(
                5,
                y,
                stage,
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    edgecolor="black",
                    facecolor="white"
                )
            )

            if i < len(stages) - 1:

                ax.annotate(
                    "",
                    xy=(5, y - 0.65),
                    xytext=(5, y - 0.25),
                    arrowprops=dict(
                        arrowstyle="->"
                    )
                )

        ax.set_title(
            name.replace("_", " "),
            fontsize=13
        )

        fig.tight_layout()

        fig.savefig(
            output / f"{name}.png",
            dpi=250
        )

        plt.close(fig)


def save_loss_curve(
    history,
    output,
    name
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        history["train_loss"],
        label="Training Loss"
    )

    ax.plot(
        history["validation_loss"],
        label="Validation Loss"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.set_title(name)

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        output / f"{name}_loss_curve.png",
        dpi=250
    )

    plt.close(fig)


def save_combined_classification_curves(
    y_true,
    model_probabilities,
    class_names,
    output,
    name,
    title
):

    num_classes = len(class_names)
    y_true = np.asarray(y_true)
    binary = (
        y_true[:, None] == np.arange(num_classes)[None, :]
    ).astype(int)
    output = output / "Combined_Contribution_Curves"
    output.mkdir(parents=True, exist_ok=True)

    mean_fpr = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(9, 7))
    for model_name, probabilities in model_probabilities.items():
        class_tprs = []
        for i in range(num_classes):
            if len(np.unique(binary[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary[:, i], probabilities[:, i])
            class_tprs.append(np.interp(mean_fpr, fpr, tpr))
        if class_tprs:
            ax.plot(
                mean_fpr,
                np.mean(class_tprs, axis=0),
                linewidth=2,
                label=model_name,
            )

    if model_probabilities:
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Chance",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} - ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"{name}_ROC.png", dpi=250)
    plt.close(fig)

    mean_recall = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(9, 7))
    for model_name, probabilities in model_probabilities.items():
        class_precisions = []
        for i in range(num_classes):
            if len(np.unique(binary[:, i])) < 2:
                continue
            precision, recall, _ = precision_recall_curve(
                binary[:, i],
                probabilities[:, i],
            )
            class_precisions.append(
                np.interp(mean_recall, recall[::-1], precision[::-1])
            )
        if class_precisions:
            ax.plot(
                mean_recall,
                np.mean(class_precisions, axis=0),
                linewidth=2,
                label=model_name,
            )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} - Precision-Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"{name}_precision_recall.png", dpi=250)
    plt.close(fig)

    thresholds = np.linspace(0, 1, 201)
    actual = binary[:, :, None].astype(bool)
    threshold_metrics = {
        "precision": ("Precision", {}),
        "recall": ("Recall", {}),
        "f1": ("F1 score", {}),
    }
    for model_name, probabilities in model_probabilities.items():
        predicted = probabilities[:, :, None] >= thresholds[None, None, :]
        true_positive = np.sum(predicted & actual, axis=0)
        false_positive = np.sum(predicted & ~actual, axis=0)
        false_negative = np.sum(~predicted & actual, axis=0)
        class_precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros_like(true_positive, dtype=float),
            where=(true_positive + false_positive) != 0,
        )
        class_recall = np.divide(
            true_positive,
            true_positive + false_negative,
            out=np.zeros_like(true_positive, dtype=float),
            where=(true_positive + false_negative) != 0,
        )
        class_f1 = np.divide(
            2 * class_precision * class_recall,
            class_precision + class_recall,
            out=np.zeros_like(class_precision),
            where=(class_precision + class_recall) != 0,
        )
        for metric_name, values in (
            ("precision", class_precision),
            ("recall", class_recall),
            ("f1", class_f1),
        ):
            threshold_metrics[metric_name][1][model_name] = np.mean(
                values,
                axis=0,
            )

    for metric_name, (ylabel, model_values) in threshold_metrics.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        for model_name, values in model_values.items():
            ax.plot(thresholds, values, linewidth=2, label=model_name)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {ylabel} vs Threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / f"{name}_{metric_name}_vs_threshold.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name, probabilities in model_probabilities.items():
        confidence = np.max(probabilities, axis=1)
        predicted_class = np.argmax(probabilities, axis=1)
        observed_accuracy = predicted_class == y_true
        fraction_positive, mean_predicted_value = calibration_curve(
            observed_accuracy.astype(int),
            confidence,
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(
            mean_predicted_value,
            fraction_positive,
            marker="o",
            label=model_name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{title} - Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"{name}_calibration.png", dpi=250)
    plt.close(fig)


# ============================================================
# CONTRIBUTION RESULT HANDLING
# ============================================================

def evaluate_and_record(
    name,
    contribution,
    y_test,
    probabilities,
    class_names,
    output,
    training_time=0,
    inference_time=0
):

    metrics, predictions = calculate_metrics(
        y_test,
        probabilities
    )

    metrics["Training_Seconds"] = training_time

    metrics["Inference_Seconds"] = inference_time

    metrics["Inference_Milliseconds_Per_Sample"] = (
        inference_time / max(1, len(y_test)) * 1000
    )

    print(
        f"\n[{contribution}] {name:45s} | "
        f"Accuracy: {metrics['Accuracy']:.4f} | "
        f"Macro-F1: {metrics['Macro_F1']:.4f}"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=[str(c) for c in class_names],
        zero_division=0
    )

    with open(
        output / f"{name}_classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    save_confusion_matrix(
        y_test,
        probabilities,
        class_names,
        output / f"{name}_confusion_matrix.png",
        f"{name} Confusion Matrix"
    )

    result = {

        "Contribution": contribution,

        "Contribution_Name": CONTRIBUTIONS.get(
            contribution,
            contribution
        ),

        "Model": name,

        **metrics
    }

    return result


# ============================================================
# ONE COMPLETE DATASET EXPERIMENT
# ============================================================

# %% [markdown]
# ### Full Week 3 workflow: training, validation-based fusion and test evaluation

# %% [Code cell 15]
def run_experiment(
    dataset_path,
    dataset_name,
    target_name
):

    print("\n" + "=" * 85)

    print(
        f"RUNNING SUITE: {dataset_name} | TARGET: {target_name}"
    )

    print("=" * 85)

    df = load_dataset(dataset_path)

    if target_name.lower() == "fertilizer":

        target = find_column(
            df,
            [
                "Fertilizer",
                "Fertilizer Name",
                "fertilizer_name"
            ]
        )

    else:

        target = find_column(
            df,
            [
                target_name,
                "Crop",
                "label"
            ]
        )

    if target is None:

        raise ValueError(
            f"Target {target_name} not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[target]
    ).reset_index(drop=True)

    y_encoder = LabelEncoder()

    y = y_encoder.fit_transform(
        df[target].astype(str)
    )

    class_names = y_encoder.classes_

    num_classes = len(class_names)

    X_original = get_features(
        df,
        target
    )

    indices = np.arange(len(y))

    (
        X_trainval,
        X_test,
        y_trainval,
        y_test,
        idx_trainval,
        idx_test
    ) = train_test_split(
        X_original,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        idx_train,
        idx_val
    ) = train_test_split(
        X_trainval,
        y_trainval,
        idx_trainval,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y_trainval
    )

    output = (
        OUTPUT_DIR
        / dataset_name.replace(" ", "_")
        / target_name
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_val)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Number of classes: {num_classes}"
    )

    results = []
    curve_probabilities = {}

    # --------------------------------------------------------
    # C1: AGRONOMIC FEATURE ENGINEERING
    # --------------------------------------------------------

    print("\n" + "-" * 85)

    print(
        "CONTRIBUTION C1: AGRONOMIC FEATURE ENGINEERING"
    )

    print("-" * 85)

    X_train_c1 = contribution_C1(X_train)

    X_val_c1 = contribution_C1(X_val)

    X_test_c1 = contribution_C1(X_test)

    (
        train_c1,
        val_c1,
        test_c1,
        imputer_c1,
        scaler_c1
    ) = preprocess_splits(
        X_train_c1,
        X_val_c1,
        X_test_c1
    )

    start = time.time()

    model_c1, history_c1 = train_network(
        train_c1,
        y_train,
        val_c1,
        y_val,
        num_classes
    )

    training_time = time.time() - start

    start = time.time()

    c1_test_probs, _ = neural_probabilities(
        model_c1,
        test_c1
    )

    inference_time = time.time() - start

    curve_probabilities["C1"] = c1_test_probs

    results.append(
        evaluate_and_record(
            "C1_Agronomic_Feature_Engineering",
            "C1",
            y_test,
            c1_test_probs,
            class_names,
            output,
            training_time,
            inference_time
        )
    )

    save_loss_curve(
        history_c1,
        output,
        "C1"
    )

    # --------------------------------------------------------
    # C2: CROP-CONDITIONED FERTILIZER RECOMMENDATION
    # --------------------------------------------------------

    is_fertilizer = (
        "fertilizer" in target_name.lower()
    )

    if is_fertilizer:

        print("\n" + "-" * 85)

        print(
            "CONTRIBUTION C2: CROP-CONDITIONED FERTILIZER RECOMMENDATION"
        )

        print("-" * 85)

        crop_col = find_column(
            df,
            ["Crop", "crop"]
        )

        if crop_col is None:

            raise ValueError(
                "C2 requires a Crop column in the fertilizer dataset."
            )

        crop_encoder = LabelEncoder()

        crop_all = crop_encoder.fit_transform(
            df[crop_col].astype(str)
        )

        crop_train = crop_all[idx_train]

        (
            train_c2_raw,
            val_c2_raw,
            test_c2_raw,
            crop_model
        ) = contribution_C2(
            X_train_c1,
            X_val_c1,
            X_test_c1,
            crop_train
        )

        c2_scaler = StandardScaler()

        train_c2 = c2_scaler.fit_transform(
            train_c2_raw
        ).astype(np.float32)

        val_c2 = c2_scaler.transform(
            val_c2_raw
        ).astype(np.float32)

        test_c2 = c2_scaler.transform(
            test_c2_raw
        ).astype(np.float32)

        start = time.time()

        model_c2, history_c2 = train_network(
            train_c2,
            y_train,
            val_c2,
            y_val,
            num_classes
        )

        training_time = time.time() - start

        start = time.time()

        c2_test_probs, _ = neural_probabilities(
            model_c2,
            test_c2
        )

        inference_time = time.time() - start

        curve_probabilities["C2"] = c2_test_probs

        results.append(
            evaluate_and_record(
                "C2_Crop_Conditioned_Fertilizer",
                "C2",
                y_test,
                c2_test_probs,
                class_names,
                output,
                training_time,
                inference_time
            )
        )

        save_loss_curve(
            history_c2,
            output,
            "C2"
        )

        train_final = train_c2

        val_final = val_c2

        test_final = test_c2

        c2_probabilities_val = c2_test_probs

    else:

        print(
            "\n[C2] NOT APPLICABLE: "
            "Crop-conditioned fertilizer features are used only "
            "for fertilizer prediction."
        )

        train_final = train_c1

        val_final = val_c1

        test_final = test_c1

        model_c2 = model_c1

        history_c2 = history_c1

    # --------------------------------------------------------
    # C3: GATED RESIDUAL NETWORK
    # --------------------------------------------------------

    print("\n" + "-" * 85)

    print(
        "CONTRIBUTION C3: GATED RESIDUAL NETWORK"
    )

    print("-" * 85)

    start = time.time()

    model_c3, history_c3 = train_network(
        train_final,
        y_train,
        val_final,
        y_val,
        num_classes
    )

    training_time = time.time() - start

    start = time.time()

    c3_test_probs, c3_uncertainty = neural_probabilities(
        model_c3,
        test_final,
        mc_dropout=True
    )

    inference_time = time.time() - start

    curve_probabilities["C3"] = c3_test_probs

    results.append(
        evaluate_and_record(
            "C3_Gated_Residual_Class_Balanced_Network",
            "C3",
            y_test,
            c3_test_probs,
            class_names,
            output,
            training_time,
            inference_time
        )
    )

    save_loss_curve(
        history_c3,
        output,
        "C3"
    )

    # --------------------------------------------------------
    # C4: UNCERTAINTY-AWARE ADAPTIVE FUSION
    # --------------------------------------------------------

    print("\n" + "-" * 85)

    print(
        "CONTRIBUTION C4: UNCERTAINTY-AWARE ADAPTIVE FUSION"
    )

    print("-" * 85)

    start = time.time()

    tree_val_probs, tree_test_probs = train_tree_models(
        train_final,
        y_train,
        val_final,
        test_final,
        num_classes
    )

    training_time = time.time() - start

    c3_val_probs, c3_val_uncertainty = neural_probabilities(
        model_c3,
        val_final,
        mc_dropout=True
    )

    candidate_names = (
        ["C3_Neural"]
        + list(tree_val_probs.keys())
    )

    validation_candidates = {
        "C3_Neural": c3_val_probs,
        **tree_val_probs
    }

    test_candidates = {
        "C3_Neural": c3_test_probs,
        **tree_test_probs
    }

    validation_scores = {}

    for name, probabilities in validation_candidates.items():

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        validation_scores[name] = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

    best_single = max(
        validation_scores,
        key=validation_scores.get
    )

    best_score = validation_scores[best_single]

    best_weights = None

    if len(candidate_names) >= 2:

        for i in range(len(candidate_names)):

            for j in range(i + 1, len(candidate_names)):

                name_a = candidate_names[i]

                name_b = candidate_names[j]

                for weight_a in np.arange(0.1, 1.0, 0.1):

                    weight_b = 1.0 - weight_a

                    combined = (
                        weight_a * validation_candidates[name_a]
                        + weight_b * validation_candidates[name_b]
                    )

                    predictions = np.argmax(
                        combined,
                        axis=1
                    )

                    score = f1_score(
                        y_val,
                        predictions,
                        average="macro",
                        zero_division=0
                    )

                    if score > best_score:

                        best_score = score

                        best_single = None

                        best_weights = (
                            name_a,
                            name_b,
                            weight_a,
                            weight_b
                        )

    if best_weights is None:

        c4_test_probs = test_candidates[best_single]

        selected_description = best_single

    else:

        name_a, name_b, weight_a, weight_b = best_weights

        c4_test_probs = (
            weight_a * test_candidates[name_a]
            + weight_b * test_candidates[name_b]
        )

        selected_description = (
            f"{name_a} ({weight_a:.2f}) + "
            f"{name_b} ({weight_b:.2f})"
        )

    uncertainty_threshold = np.percentile(
        c3_val_uncertainty,
        75
    )

    high_uncertainty = (
        c3_uncertainty > uncertainty_threshold
    )

    uncertainty_df = pd.DataFrame({

        "Uncertainty": c3_uncertainty,

        "High_Uncertainty": high_uncertainty

    })

    uncertainty_df.to_csv(
        output / "C4_uncertainty_analysis.csv",
        index=False
    )

    start = time.time()

    c4_test_probs = np.clip(
        c4_test_probs,
        1e-12,
        1
    )

    c4_test_probs /= c4_test_probs.sum(
        axis=1,
        keepdims=True
    )

    inference_time = time.time() - start

    curve_probabilities["C4"] = c4_test_probs

    results.append(
        evaluate_and_record(
            "C4_Uncertainty_Aware_Adaptive_Fusion",
            "C4",
            y_test,
            c4_test_probs,
            class_names,
            output,
            training_time,
            inference_time
        )
    )

    save_combined_classification_curves(
        y_test,
        curve_probabilities,
        class_names,
        output,
        (
            "combined_all_4_contributions"
            if is_fertilizer
            else "combined_applicable_contributions"
        ),
        (
            "Combined Contributions (C1-C4)"
            if is_fertilizer
            else "Combined Contributions (C1, C3, C4; C2 not applicable)"
        ),
    )

    print(
        f"\nC4 selected model/fusion: {selected_description}"
    )

    print(
        f"C4 validation Macro-F1 selection score: {best_score:.4f}"
    )

    print(
        f"C4 uncertainty threshold: {uncertainty_threshold:.8f}"
    )

    print(
        f"High uncertainty samples: "
        f"{high_uncertainty.sum()}/{len(high_uncertainty)}"
    )

    # --------------------------------------------------------
    # SAVE CONTRIBUTION RESULTS
    # --------------------------------------------------------

    contribution_df = pd.DataFrame(results)

    contribution_df.to_csv(
        output / "contribution_wise_results.csv",
        index=False
    )

    contribution_df.to_excel(
        output / "contribution_wise_results.xlsx",
        index=False
    )

    return results


# ============================================================
# MAIN EXECUTION
# ============================================================

# %% [Code cell 16]
# Capture fresh Week 3 predictions without changing training or model selection.
from functools import wraps


def capture_predictions(evaluator, dataset, target, sink):
    @wraps(evaluator)
    def wrapped(name, contribution, y_test, probabilities, class_names, *args, **kwargs):
        names = np.asarray(class_names)
        actual = names[np.asarray(y_test, dtype=int)]
        predicted = names[np.asarray(probabilities).argmax(axis=1)]
        frame = pd.DataFrame({"Dataset": dataset, "Target": target, "Model": name,
                              "Actual_Label": actual, "Predicted_Label": predicted,
                              "Correct": actual == predicted,
                              "Prediction_Confidence": np.asarray(probabilities).max(axis=1)})
        for i, label in enumerate(names):
            frame[f"Probability_{label}"] = np.asarray(probabilities)[:, i]
        sink.append(frame)
        return evaluator(name, contribution, y_test, probabilities, class_names, *args, **kwargs)
    return wrapped


week3_rows, week3_predictions = [], []
original_evaluator = evaluate_and_record
set_seed(SEED)
save_architecture_diagrams(OUTPUT_DIR)
for dataset_name, task_target in TASKS:
    evaluate_and_record = capture_predictions(original_evaluator, dataset_name, task_target, week3_predictions)
    try:
        rows = run_experiment(DATASETS[dataset_name], dataset_name, task_target)
    finally:
        evaluate_and_record = original_evaluator
    week3_rows.extend({"Dataset": dataset_name, "Target": task_target, **row} for row in rows)
WEEK3_RESULTS = pd.DataFrame(week3_rows)
WEEK3_PREDICTIONS = pd.concat(week3_predictions, ignore_index=True)
WEEK3_PREDICTIONS.to_csv(ANALYSIS_ROOT / "week3_holdout_predictions.csv", index=False)
assert len(WEEK3_RESULTS) == 10, "Expected C1/C3/C4 on crop tasks and C1–C4 on fertilizer."
display(WEEK3_RESULTS[["Dataset", "Target", "Contribution", "Accuracy", "Macro_F1"]])

# %% [markdown]
# ## 5. Quantitative analysis: held-out metrics and per-class errors
# Rank models **within** each dataset, target, and protocol. The paper baseline and Week 3 use different inputs and splitting/noise protocols, so these tables are not a controlled comparison. Read small-class support alongside recall and F1.

# %% [Code cell 17]
from sklearn.metrics import classification_report as per_class_report


def with_protocol(paper, week3):
    return pd.concat([paper.assign(Protocol="Three papers"), week3.assign(Protocol="Week 3")],
                     ignore_index=True, sort=False)

metrics = with_protocol(PAPER_RESULTS, WEEK3_RESULTS)
predictions = with_protocol(PAPER_PREDICTIONS, WEEK3_PREDICTIONS)
keys = ["Protocol", "Dataset", "Target"]
metrics["Rank_Macro_F1"] = metrics.groupby(keys)["Macro_F1"].rank(method="min", ascending=False)
metrics = metrics.sort_values(keys + ["Rank_Macro_F1"])
metric_columns = keys + ["Model", "Rank_Macro_F1", "Accuracy", "Balanced_Accuracy",
                         "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1", "Log_Loss"]
metric_columns = [name for name in metric_columns if name in metrics]
metrics.to_csv(ANALYSIS_ROOT / "all_heldout_metrics.csv", index=False)
display(metrics[metric_columns])

per_class = []
for group, frame in predictions.groupby(keys + ["Model"], sort=False):
    labels = sorted(set(frame.Actual_Label) | set(frame.Predicted_Label))
    report = per_class_report(frame.Actual_Label, frame.Predicted_Label, labels=labels,
                              output_dict=True, zero_division=0)
    for label in labels:
        per_class.append({**dict(zip(keys + ["Model"], group)), "Class": label, **report[label]})
per_class = pd.DataFrame(per_class)
per_class.to_csv(ANALYSIS_ROOT / "all_per_class_metrics.csv", index=False)
display(per_class.sort_values(["recall", "support"]).head(20))

# %% [markdown]
# ## 6. Qualitative analysis: model scores, confusion patterns and confidence
# The score figures include every model. The confusion and confidence figures show the highest **observed test** macro-F1 model per task and protocol for descriptive error inspection only, not for selecting a deployment model. Confidence reliability here means top-label confidence versus observed accuracy, not classwise calibration. The Week 3 implementation also saves its native training loss and classification curves.

# %% [Code cell 18]
from sklearn.metrics import confusion_matrix as confusion_table

errors = []
for i, (group, scores) in enumerate(metrics.groupby(keys, sort=False), 1):
    protocol, dataset, target = group
    title = f"{protocol} / {target} — {dataset}"
    winner = scores.iloc[0].Model
    heldout = predictions.loc[(predictions.Protocol == protocol) & (predictions.Dataset == dataset)
                              & (predictions.Target == target) & (predictions.Model == winner)]
    labels = sorted(set(heldout.Actual_Label) | set(heldout.Predicted_Label))
    matrix = confusion_table(heldout.Actual_Label, heldout.Predicted_Label, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    for (a, b), n in np.ndenumerate(matrix):
        if a != b and n:
            errors.append({**dict(zip(keys, group)), "Model": winner, "Actual": labels[a],
                           "Predicted": labels[b], "Count": n, "Share_of_Actual_Class": normalized[a, b]})

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), layout="constrained")
    scores.set_index("Model")[["Accuracy", "Macro_F1"]].plot.barh(ax=axes[0], xlim=(0, 1))
    axes[0].set_title("All models: held-out scores")
    axes[0].invert_yaxis()
    im = axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[1].set(xlabel="Predicted", ylabel="Actual", title="Row-normalized confusion")
    fig.colorbar(im, ax=axes[1], shrink=0.7)
    bins = pd.cut(heldout.Prediction_Confidence, np.linspace(0, 1, 11), include_lowest=True)
    calibration = heldout.groupby(bins, observed=True).agg(
        Mean_Confidence=("Prediction_Confidence", "mean"), Observed_Accuracy=("Correct", "mean"),
        Support=("Correct", "size"))
    calibration.to_csv(ANALYSIS_ROOT / f"confidence_bins_{i}.csv")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].scatter(calibration.Mean_Confidence, calibration.Observed_Accuracy,
                    s=20 + 150 * calibration.Support / len(heldout))
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy",
                title="Top-label reliability (size = bin support)")
    fig.suptitle(title + "\nError plots: " + winner)
    fig.savefig(ANALYSIS_ROOT / f"qualitative_comparison_{i}.png", dpi=150)
    display(fig)
    plt.close(fig)

error_table = pd.DataFrame(errors, columns=keys + ["Model", "Actual", "Predicted", "Count", "Share_of_Actual_Class"])
error_table = error_table.sort_values("Count", ascending=False)
error_table.to_csv(ANALYSIS_ROOT / "common_confusions.csv", index=False)
display(error_table.head(20))

# %% [markdown]
# ## 7. Export and interpretation
# - In Week 3, the original feature selector retains `Fertilizer` when predicting `Crop`, and `Crop` when predicting `Fertilizer`. That is valid only when the counterpart is known at inference. Do not describe these as sensor-only results. In particular, the C2 crop classifier receives the original C1 features, which can already include the known crop field in this reproduction.
# - A random row split does not measure transfer to unseen districts or farms. Small classes may have only a few test rows; treat differences cautiously.
# - The notebook runs the repository's available implementations. It does not recreate archived ablations, paired significance tests, or noise sweeps. Differences between C1–C4 rows are descriptive; they are not isolated component effects.

# %% [Code cell 19]
import shutil

archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT)
print("Results archive:", archive)
display(FileLink(os.path.relpath(archive)))
