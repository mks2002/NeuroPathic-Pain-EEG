import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_MERGED_DIR = Path("Merged_Features_AllMethods")
BASE_RANK_DIR = Path("Merged_Features_AllMethods")
OUT_MAIN_DIR = Path("ML_Model_Results")
OUT_MAIN_DIR.mkdir(exist_ok=True, parents=True)
FEATURE_COUNTS = [2, 3, 4, 5]

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def subject_stratified_split(df, subject_col="subj_id", label_col="label", test_size=0.2, random_state=42):
    subjects = df[[subject_col, label_col]].drop_duplicates()
    train_subj, test_subj = train_test_split(
        subjects,
        test_size=test_size,
        stratify=subjects[label_col],
        random_state=random_state
    )
    train_df = df[df[subject_col].isin(train_subj[subject_col])]
    test_df = df[df[subject_col].isin(test_subj[subject_col])]
    return train_df, test_df


def plot_confusion_matrix(y_true, y_pred, labels, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder, model_name, out_dir):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    labels = list(label_encoder.classes_)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Save classification report
    report_text = classification_report(y_test, y_pred, target_names=labels)
    report_path = out_dir / f"classification_report_{model_name}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Plot confusion matrix
    cm_path = out_dir / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(y_test, y_pred, labels, cm_path, f"Confusion Matrix - {model_name}")

    print(f"\n✅ {model_name} → Accuracy: {acc:.3f}, Macro-F1: {f1_macro:.3f}, Weighted-F1: {f1_weighted:.3f}")
    print(report_text)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path)
    }


# ==========================================================
# MAIN PIPELINE
# ==========================================================
for n_feats in FEATURE_COUNTS:
    print(f"\n🚀 Processing {n_feats} best features per channel...")
    out_dir = OUT_MAIN_DIR / f"{n_feats}best_features"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Merge top N features from each channel
    channels = sorted([f.stem.replace("_merged", "") for f in BASE_MERGED_DIR.glob("*_merged.csv")])
    dfs = []

    for ch in channels:
        merged_file = BASE_MERGED_DIR / f"{ch}_merged.csv"
        rank_file = BASE_RANK_DIR / f"{ch}_feature_ranking.csv"
        if not merged_file.exists() or not rank_file.exists():
            continue

        df = pd.read_csv(merged_file)
        rank_df = pd.read_csv(rank_file)
        top_feats = rank_df.head(n_feats)["feature"].tolist()

        keep_cols = ["subj_id", "window_idx", "label"] + top_feats
        dfs.append(df[keep_cols])

    # Merge all channels horizontally by subj_id, window_idx, label
    merged_global = dfs[0]
    for df in dfs[1:]:
        merged_global = pd.merge(merged_global, df, on=["subj_id", "window_idx", "label"], how="inner")

    print(f"✅ Combined global matrix shape: {merged_global.shape}")

    # Encode labels
    le = LabelEncoder()
    merged_global["y"] = le.fit_transform(merged_global["label"].astype(str))

    # Subject-level stratified split
    train_df, test_df = subject_stratified_split(merged_global)
    X_train = train_df.drop(columns=["subj_id", "window_idx", "label", "y"], errors="ignore")
    X_test = test_df.drop(columns=["subj_id", "window_idx", "label", "y"], errors="ignore")
    y_train, y_test = train_df["y"], test_df["y"]

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss"
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    results = []
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, le, model_name, out_dir)
        results.append(metrics)

    # Save summary
    pd.DataFrame(results).to_csv(out_dir / "summary_results.csv", index=False)
    print(f"\n📁 Results saved for top {n_feats} features → {out_dir}")




# this is for top features amonng each channels .....