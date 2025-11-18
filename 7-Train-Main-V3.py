"""
Per-channel ML pipeline for EEG merged features
"""

import os
import sys
import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except:
    HAS_LGBM = False


RANDOM_STATE = 42
MERGED_DIR = "Merged_Features_AllMethods"
OUTPUT_ROOT = "ML-MODEL-RESULTS-PerChannel_V2"
TEST_SIZE = 0.2
LABEL_COL = 'label'
META_COLS = ['subj_id', 'window_idx', 'label', 'pain_score']


models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
    "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE),
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False
    )

if HAS_LGBM:
    models["LGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)

models.update({
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "KNearestNeighbors": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "MLPClassifier": MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
})

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =====================================================================================
# Helper Functions
# =====================================================================================

def find_merged_files(merged_dir):
    return sorted(glob.glob(os.path.join(merged_dir, "*_merged.csv")))


def infer_channel_name(filename):
    base = os.path.basename(filename)
    return base.replace("_merged.csv", "")


def check_required_columns(df):
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' missing in CSV")


def map_labels(series):
    mapping = {"low": 0, "mid": 1, "high": 2}
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            key = v.strip().lower()
            if key in mapping:
                return mapping[key]
            if key.isdigit():
                return int(key)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise ValueError(f"Unknown label value: {v}")
    return series.map(conv)


def impute_subject_median(df, feature_cols):
    df2 = df.copy()
    for subj, idxs in df.groupby("subj_id").groups.items():
        med = df.loc[idxs, feature_cols].median()
        df2.loc[idxs, feature_cols] = df.loc[idxs, feature_cols].fillna(med)
    return df2


def ensure_numeric_features(df):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df, feature_cols


def save_channel_log(channel_outdir, df, train_subjs, test_subjs):
    lines = []
    lines.append(f"Channel log for folder: {channel_outdir}\n")

    lines.append("Train subjects by label:")
    train_map = defaultdict(list)
    for s in train_subjs:
        mode = map_labels(df[df["subj_id"] == s][LABEL_COL]).mode().iloc[0]
        train_map[int(mode)].append(int(s))
    for k in sorted(train_map.keys()):
        lines.append(f"  label {k}: {train_map[k]}")

    lines.append("\nTest subjects by label:")
    test_map = defaultdict(list)
    for s in test_subjs:
        mode = map_labels(df[df["subj_id"] == s][LABEL_COL]).mode().iloc[0]
        test_map[int(mode)].append(int(s))
    for k in sorted(test_map.keys()):
        lines.append(f"  label {k}: {test_map[k]}")

    lines.append("\nDetailed per-subject row ranges:")
    for s in sorted(df["subj_id"].unique(), key=int):
        idxs = df.index[df["subj_id"] == s]
        start0, end0 = int(idxs[0]), int(idxs[-1])
        lines.append(f"  subj {s}: rows [{start0},{end0}]  total={len(idxs)}")

    with open(os.path.join(channel_outdir, "split_log.txt"), "w") as f:
        f.write("\n".join(lines))


# =====================================================================================
# MAIN TRAINING FUNCTION (now includes accuracy export)
# =====================================================================================

def run_models_for_channel(csv_path, global_summary):
    channel = infer_channel_name(csv_path)
    print("\n" + "="*80)
    print("Processing", channel)

    # ------------------------------
    # CHANNEL FOLDER + SUBFOLDERS
    # ------------------------------
    outdir = os.path.join(OUTPUT_ROOT, channel)
    classrep_dir = os.path.join(outdir, "classification_reports")
    confmat_dir = os.path.join(outdir, "confusion_matrices")
    model_dir = os.path.join(outdir, "models")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(classrep_dir, exist_ok=True)
    os.makedirs(confmat_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)

    check_required_columns(df)
    df[LABEL_COL] = map_labels(df[LABEL_COL])
    df, feature_cols = ensure_numeric_features(df)
    df = impute_subject_median(df, feature_cols)

    subjs = df["subj_id"].unique()
    subj_labels = [map_labels(df[df["subj_id"] == s][LABEL_COL]).mode().iloc[0] for s in subjs]

    train_subs, test_subs = train_test_split(
        subjs, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=subj_labels
    )

    save_channel_log(outdir, df, train_subs, test_subs)

    # ------------------------------
    # BUILD TRAIN/TEST
    # ------------------------------
    train_df = df[df["subj_id"].isin(train_subs)].reset_index(drop=True)
    test_df = df[df["subj_id"].isin(test_subs)].reset_index(drop=True)

    # save train/test CSV inside channel folder
    train_df.to_csv(os.path.join(outdir, "train_df.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test_df.csv"), index=False)

    train_meds = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(train_meds)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_meds)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df[LABEL_COL].astype(int).values
    y_test = test_df[LABEL_COL].astype(int).values

    # store accuracies for this channel
    channel_accuracies = []

    # =====================================================================================
    # TRAIN ALL MODELS
    # =====================================================================================
    for model_name, model in models.items():
        print(" Training:", model_name)

        try:
            clf = model
            clf.fit(X_train, y_train)
        except Exception as e:
            print("[ERROR] Train fail:", model_name, e)
            continue

        try:
            y_pred = clf.predict(X_test)
        except Exception as e:
            print("[ERROR] Predict fail:", model_name, e)
            continue

        # compute accuracy
        acc = accuracy_score(y_test, y_pred)
        channel_accuracies.append((model_name, acc))

        # classification report
        report = classification_report(y_test, y_pred)
        with open(os.path.join(classrep_dir, f"{model_name}_classification_report.txt"), "w") as f:
            f.write(report)

        # confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
            fig, ax = plt.subplots(figsize=(6,6))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            plt.title(f"{channel} - {model_name}")
            fig.savefig(os.path.join(confmat_dir, f"{model_name}_confusion_matrix.png"))
            plt.close(fig)
        except:
            pass

        # save model
        with open(os.path.join(model_dir, f"{model_name}_model.pkl"), "wb") as f:
            pickle.dump(clf, f)

    # sort this channel's accuracies descending
    channel_accuracies.sort(key=lambda x: x[1], reverse=True)

    # add to global summary
    global_summary[channel] = channel_accuracies

    print("Finished", channel)


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    files = find_merged_files(MERGED_DIR)
    if not files:
        print("No merged files found!")
        return

    print("Found", len(files), "files.")

    global_summary = {}

    for f in files:
        run_models_for_channel(f, global_summary)

    # =========================================================================
    # WRITE GLOBAL SUMMARY FILE
    # =========================================================================
    summary_path = os.path.join(OUTPUT_ROOT, "all-channel-model-summary.txt")
    with open(summary_path, "w") as f:
        for channel, results in global_summary.items():
            f.write(f"\nChannel: {channel}\n")
            for model_name, acc in results:
                f.write(f"  {model_name}: {acc:.4f}\n")
            f.write("-"*40 + "\n")

    print("\nGlobal summary saved to:", summary_path)
    print("All done.")


if __name__ == "__main__":
    main()
