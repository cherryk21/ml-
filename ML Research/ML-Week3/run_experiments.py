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
EPOCHS = 150
LEARNING_RATE = 0.003
BATCH_SIZE = 32
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20
SENSOR_NOISE = 0.25
MC_DROPOUT_PASSES = 30

OUTPUT_DIR = Path("analysis_outputs")
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

if __name__ == "__main__":

    print("=" * 85)

    print(
        "FOUR-CONTRIBUTION FERTILIZER RECOMMENDATION EXPERIMENT"
    )

    print(
        f"TRAINING EPOCHS: {EPOCHS}"
    )

    print(
        f"LEARNING RATE: {LEARNING_RATE}"
    )

    print(
        f"BATCH SIZE: {BATCH_SIZE}"
    )

    print(
        f"SENSOR NOISE CONFIGURATION: {SENSOR_NOISE}"
    )

    print("=" * 85)

    save_architecture_diagrams(
        OUTPUT_DIR
    )

    all_results = []

    experiments = [

        (
            DATASET_1,
            "Dataset 1 (Crop & Fertilizer)",
            "Fertilizer"
        ),

        (
            DATASET_1,
            "Dataset 1 (Crop & Fertilizer)",
            "Crop"
        ),

        (
            DATASET_2,
            "Dataset 2 (Crop Recommendation)",
            "label"
        )

    ]

    for dataset_path, dataset_name, target in experiments:

        try:

            results = run_experiment(
                dataset_path,
                dataset_name,
                target
            )

            # ADD DATASET AND TARGET METADATA
            for result in results:

                result["Dataset"] = dataset_name

                result["Target"] = target

            all_results.extend(results)

        except Exception as error:

            print("\nEXPERIMENT ERROR")

            print(
                f"Dataset: {dataset_name}"
            )

            print(
                f"Target: {target}"
            )

            print(
                f"Reason: {error}"
            )

            import traceback

            traceback.print_exc()

    # --------------------------------------------------------
    # FINAL EXPERIMENTAL SUMMARY
    # --------------------------------------------------------

    if all_results:

        final_df = pd.DataFrame(
            all_results
        )

        # ADD DESCRIPTIVE PAPER COLUMN
        final_df.insert(
            0,
            "Paper",
            final_df["Contribution"] + " - " +
            final_df["Contribution_Name"]
        )

        # COMPLETE METRIC COLUMN ORDER
        preferred_columns = [

            "Paper",
            "Dataset",
            "Target",
            "Model",

            "Test_Samples",
            "Class_Count",

            "Accuracy",
            "Balanced_Accuracy",

            "Macro_Precision",
            "Weighted_Precision",
            "Micro_Precision",

            "Macro_Recall",
            "Weighted_Recall",
            "Micro_Recall",

            "Macro_F1",
            "Weighted_F1",
            "Micro_F1",

            "Macro_Specificity",
            "Weighted_Specificity",

            "Macro_Negative_Predictive_Value",

            "Matthews_Correlation_Coefficient",
            "Cohen_Kappa",

            "Hamming_Loss",

            "Macro_Jaccard",
            "Weighted_Jaccard",

            "Top_2_Accuracy",

            "Log_Loss",
            "Multiclass_Brier_Score",

            "Expected_Calibration_Error_15_Bins",

            "Macro_ROC_AUC_OVR",
            "Weighted_ROC_AUC_OVR",

            "Macro_Average_Precision",
            "Weighted_Average_Precision",

            "Training_Seconds",
            "Inference_Seconds",
            "Inference_Milliseconds_Per_Sample"
        ]

        # RETAIN ALL AVAILABLE METRICS
        available_columns = [
            column
            for column in preferred_columns
            if column in final_df.columns
        ]

        # RETAIN ANY ADDITIONAL CALCULATED METRICS
        additional_columns = [
            column
            for column in final_df.columns
            if column not in available_columns
            and column not in [
                "Contribution",
                "Contribution_Name"
            ]
        ]

        final_df = final_df[
            available_columns + additional_columns
        ]

        # SAVE COMPLETE CSV
        final_df.to_csv(
            OUTPUT_DIR / "FINAL_EXPERIMENTAL_SUMMARY.csv",
            index=False
        )

        # SAVE COMPLETE EXCEL
        final_df.to_excel(
            OUTPUT_DIR / "FINAL_EXPERIMENTAL_SUMMARY.xlsx",
            index=False
        )

        # DISPLAY ALL COLUMNS IN TERMINAL
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        print("\n" + "=" * 85)

        print(
            "FINAL EXPERIMENTAL SUMMARY"
        )

        print("=" * 85)

        print(
            final_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        print("\n" + "=" * 85)

        print(
            "ALL RESULTS SAVED SUCCESSFULLY"
        )

        print(
            f"OUTPUT DIRECTORY: {OUTPUT_DIR.resolve()}"
        )

        print("=" * 85)

    else:

        print(
            "No experiments completed. Check your dataset paths."
        )