# ml_pipeline_top48.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
GLOBAL_DIR = Path("Global_Feature_Selection")
GLOBAL_MERGED_FILE = GLOBAL_DIR / "Global_Feature_Matrix.csv"
TOP48_FILE = GLOBAL_DIR / "Top_48_Global_Features.csv"

OUT_DIR = Path("ML-Result-V2")
KFOLD_DIR = OUT_DIR / "KFold"
LOSO_DIR = OUT_DIR / "LeaveOne"
KFOLD_DIR.mkdir(parents=True, exist_ok=True)
LOSO_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2           # subject-level holdout proportion
N_SPLITS = 5             # K for KFold (on subjects) during CV
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                             use_label_encoder=False, eval_metric="mlogloss"),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
}

# -------------------------
# UTILITIES
# -------------------------
def plot_and_save_confusion(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return cm

def save_classification_report(y_true, y_pred, labels, out_csv):
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    rep_df = pd.DataFrame(rep).T
    rep_df.to_csv(out_csv)
    return rep_df

def subject_mode_label(df, subj_col="subj_id", label_col="label"):
    """Return mapping subj -> most common label for that subject"""
    g = df[[subj_col, label_col]].drop_duplicates()
    # If windows have multiple labels (unlikely), we take mode per subject from all rows
    modes = df.groupby(subj_col)[label_col].agg(lambda s: Counter(s).most_common(1)[0][0])
    return modes.to_dict()

def stratified_subject_split(subjects, subj_labels, test_size=0.2, random_state=42):
    """Split subject ids into train/test preserving label distribution using stratify on subj_labels"""
    subj_df = pd.DataFrame({"subj_id": subjects})
    subj_df["label"] = subj_df["subj_id"].map(subj_labels)
    train_subj, test_subj = train_test_split(
        subj_df["subj_id"],
        test_size=test_size,
        stratify=subj_df["label"],
        random_state=random_state
    )
    return list(train_subj), list(test_subj)

def subject_stratified_kfold(subjects, subj_labels, n_splits=5, random_state=42):
    """Yield folds of subject ids using StratifiedKFold on subjects (stratify by subject-level label)"""
    subj_df = pd.DataFrame({"subj_id": subjects})
    subj_df["label"] = subj_df["subj_id"].map(subj_labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(subj_df["subj_id"], subj_df["label"]):
        train_subj = subj_df.iloc[train_idx]["subj_id"].tolist()
        val_subj = subj_df.iloc[val_idx]["subj_id"].tolist()
        yield train_subj, val_subj

# -------------------------
# LOAD DATA & TOP FEATURES
# -------------------------
print("Loading global matrix and top48 list...")
if not GLOBAL_MERGED_FILE.exists():
    raise FileNotFoundError(f"{GLOBAL_MERGED_FILE} not found.")
if not TOP48_FILE.exists():
    raise FileNotFoundError(f"{TOP48_FILE} not found.")

global_df = pd.read_csv(GLOBAL_MERGED_FILE)
top48_df = pd.read_csv(TOP48_FILE)
top48 = top48_df["feature"].tolist()
print(f"Global matrix shape: {global_df.shape}; Top features: {len(top48)}")

# keep only needed columns (plus subj/window/label)
keep_cols = ["subj_id", "window_idx", "label"] + top48
global_df = global_df[keep_cols].copy()

# drop rows with NaNs in selected features (or fill? we choose drop)
global_df = global_df.dropna(subset=top48)
print(f"After dropping NaNs shape: {global_df.shape}")

# encode label
le = LabelEncoder()
global_df["y"] = le.fit_transform(global_df["label"].astype(str))
class_names = list(le.classes_)

# subject-level label map
subjects = sorted(global_df["subj_id"].unique().tolist())
subj_labels = subject_mode_label(global_df, subj_col="subj_id", label_col="label")
print(f"Found {len(subjects)} subjects; label distribution (subjects): {Counter(list(subj_labels.values()))}")

# -------------------------
# Subject-level train/test split (outer holdout)
# -------------------------
train_subj, test_subj = stratified_subject_split(subjects, subj_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_df = global_df[global_df["subj_id"].isin(train_subj)].reset_index(drop=True)
test_df = global_df[global_df["subj_id"].isin(test_subj)].reset_index(drop=True)
print(f"Train subjects: {len(train_subj)}, Test subjects: {len(test_subj)}")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# prepare X/y for final test evaluation later
X_test_final = test_df[top48].values
y_test_final = test_df["y"].values

# Standard scaler will be fit inside evaluation per model

# -------------------------
# FUNCTION: run CV (KFold or LOSO) on training subjects
# -------------------------
def run_kfold_cv_and_test(train_df, train_subjects, subj_labels, top_features, n_splits, models, out_base_dir, le):
    """Perform Stratified K-Fold on subjects (subject-level folds), aggregate CV predictions, then fit on full train and evaluate on held-out test (provided separately outside)."""
    out_base_dir.mkdir(parents=True, exist_ok=True)
    # Subject-level stratified KFold generator
    folds = list(subject_stratified_kfold(train_subjects, subj_labels, n_splits=n_splits, random_state=RANDOM_STATE))

    # For each model run CV
    summary_rows = []
    for model_name, model in models.items():
        model_dir = out_base_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        y_true_cv_all = []
        y_pred_cv_all = []

        # iterate folds
        for fold_idx, (subj_train, subj_val) in enumerate(folds, start=1):
            tr_df = train_df[train_df["subj_id"].isin(subj_train)]
            val_df = train_df[train_df["subj_id"].isin(subj_val)]
            X_tr = tr_df[top_features].values
            y_tr = tr_df["y"].values
            X_val = val_df[top_features].values
            y_val = val_df["y"].values

            # scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            # clone model: simple re-init using same params
            mdl = clone_model_by_name(model_name, models)

            mdl.fit(X_tr_s, y_tr)
            y_val_pred = mdl.predict(X_val_s)

            y_true_cv_all.extend(y_val.tolist())
            y_pred_cv_all.extend(y_val_pred.tolist())

            # optionally save per-fold confusion? (skip to reduce files)

        # aggregated CV metrics
        acc_cv = accuracy_score(y_true_cv_all, y_pred_cv_all)
        f1_macro_cv = f1_score(y_true_cv_all, y_pred_cv_all, average="macro")
        f1_weighted_cv = f1_score(y_true_cv_all, y_pred_cv_all, average="weighted")

        # save aggregated CV classification report and confusion matrix
        rep_cv_df = save_classification_and_cm(y_true_cv_all, y_pred_cv_all, le, model_dir, f"CV_KFold_{n_splits}_{model_name}")

        # Train final on full train_df and evaluate on held-out final test (outside)
        scaler_full = StandardScaler()
        X_train_full = train_df[top_features].values
        y_train_full = train_df["y"].values
        X_train_full_s = scaler_full.fit_transform(X_train_full)
        X_test_final_s = scaler_full.transform(X_test_final)

        mdl_final = clone_model_by_name(model_name, models)
        mdl_final.fit(X_train_full_s, y_train_full)
        y_test_pred = mdl_final.predict(X_test_final_s)

        # save test results
        rep_test_df = save_classification_and_cm(y_test_final, y_test_pred, le, model_dir, f"Test_{model_name}")

        summary_rows.append({
            "model": model_name,
            "cv_accuracy": acc_cv,
            "cv_f1_macro": f1_macro_cv,
            "cv_f1_weighted": f1_weighted_cv,
            "test_accuracy": accuracy_score(y_test_final, y_test_pred),
            "test_f1_macro": f1_score(y_test_final, y_test_pred, average="macro"),
            "test_f1_weighted": f1_score(y_test_final, y_test_pred, average="weighted"),
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subj)
        })

    # save summary
    pd.DataFrame(summary_rows).to_csv(out_base_dir / "summary.csv", index=False)
    return

def run_loso_cv_and_test(train_df, train_subjects, top_features, models, out_base_dir, le):
    """Leave-one-subject-out CV across training subjects, then evaluate final model on held-out test set."""
    out_base_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for model_name, model in models.items():
        model_dir = out_base_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        y_true_cv_all = []
        y_pred_cv_all = []

        # iterate over each subject in training subjects as validation once
        for val_subj in train_subjects:
            subj_train = [s for s in train_subjects if s != val_subj]
            tr_df = train_df[train_df["subj_id"].isin(subj_train)]
            val_df = train_df[train_df["subj_id"] == val_subj]

            X_tr = tr_df[top_features].values
            y_tr = tr_df["y"].values
            X_val = val_df[top_features].values
            y_val = val_df["y"].values

            if len(y_val) == 0:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            mdl = clone_model_by_name(model_name, models)
            mdl.fit(X_tr_s, y_tr)
            y_val_pred = mdl.predict(X_val_s)

            y_true_cv_all.extend(y_val.tolist())
            y_pred_cv_all.extend(y_val_pred.tolist())

        # aggregated CV metrics
        acc_cv = accuracy_score(y_true_cv_all, y_pred_cv_all)
        f1_macro_cv = f1_score(y_true_cv_all, y_pred_cv_all, average="macro")
        f1_weighted_cv = f1_score(y_true_cv_all, y_pred_cv_all, average="weighted")

        rep_cv_df = save_classification_and_cm(y_true_cv_all, y_pred_cv_all, le, model_dir, f"CV_LOSO_{model_name}")

        # final train on full train_df and test on final holdout
        scaler_full = StandardScaler()
        X_train_full = train_df[top_features].values
        y_train_full = train_df["y"].values
        X_train_full_s = scaler_full.fit_transform(X_train_full)
        X_test_final_s = scaler_full.transform(X_test_final)

        mdl_final = clone_model_by_name(model_name, models)
        mdl_final.fit(X_train_full_s, y_train_full)
        y_test_pred = mdl_final.predict(X_test_final_s)

        rep_test_df = save_classification_and_cm(y_test_final, y_test_pred, le, model_dir, f"Test_{model_name}")

        summary_rows.append({
            "model": model_name,
            "cv_accuracy": acc_cv,
            "cv_f1_macro": f1_macro_cv,
            "cv_f1_weighted": f1_weighted_cv,
            "test_accuracy": accuracy_score(y_test_final, y_test_pred),
            "test_f1_macro": f1_score(y_test_final, y_test_pred, average="macro"),
            "test_f1_weighted": f1_score(y_test_final, y_test_pred, average="weighted"),
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subj)
        })

    pd.DataFrame(summary_rows).to_csv(out_base_dir / "summary.csv", index=False)
    return

# -------------------------
# small helpers used above
# -------------------------
def clone_model_by_name(name, models_dict):
    """Return a fresh instance of model by name (same hyperparams)."""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                             use_label_encoder=False, eval_metric="mlogloss")
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")

def save_classification_and_cm(y_true, y_pred, label_encoder, base_dir, prefix):
    """Save classification report and confusion matrix to base_dir with given prefix."""
    base_dir.mkdir(parents=True, exist_ok=True)
    # classification report
    labels = list(label_encoder.classes_)
    rep_df = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    rep_df = pd.DataFrame(rep_df).T
    rep_path = base_dir / f"{prefix}_classification_report.csv"
    rep_df.to_csv(rep_path)

    # confusion
    cm_path = base_dir / f"{prefix}_confusion.png"
    plot_and_save_confusion(y_true, y_pred, labels, cm_path, prefix)
    return rep_df

# -------------------------
# RUN K-FOLD CV (on train subjects) and final test eval
# -------------------------
print("\n=== Running Subject-level Stratified K-Fold CV then final test ===")
run_kfold_cv_and_test(train_df, train_subj, subj_labels, top48, N_SPLITS, MODELS, KFOLD_DIR, le)

# -------------------------
# RUN LOSO CV (on train subjects) and final test eval
# -------------------------
print("\n=== Running Subject-level Leave-One-Subject-Out CV then final test ===")
run_loso_cv_and_test(train_df, train_subj, top48, MODELS, LOSO_DIR, le)

print("\nAll done. Results saved in:", OUT_DIR)
