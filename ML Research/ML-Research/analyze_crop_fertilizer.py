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

EPOCHS = 150
LEARNING_RATE = 0.003
BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parent

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

    print(\
        df_res.to_string(index=False)
    )

    print("\n" + "=" * 85)

    print("ALL RESULTS SAVED SUCCESSFULLY")

    print("OUTPUT DIRECTORY:", OUTPUT_DIR)

    print("BACKUP DIRECTORY:", OUTPUT_DIRS[1])

    print("=" * 85)


if __name__ == "__main__":

    main()