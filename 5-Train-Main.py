"""
Per-channel ML pipeline for EEG merged features

Creates folder ML-MODEL-RESULTS-PerChannel and runs a set of classifiers on each
channel CSV that ends with "_merged.csv" inside a folder named "Merged_features".

Key behaviour (matches user's requirements):
- Only files ending with _merged.csv are processed
- For each channel a subfolder is created under ML-MODEL-RESULTS-PerChannel/<channelName>
- Subject-level train/test split: each subject's windows go entirely into train or test.
  Stratification is done at subject-level by using the subject's most frequent label.
- No rows are dropped for NaNs; numeric NaNs are imputed using per-subject median and
  if still NaN, overall-feature median from training data is used.
- Label mapping: 'low'->0, 'mid'->1, 'high'->2 (case-insensitive). If labels are already
  numeric strings of those names, they will be mapped. The pipeline will fail loudly if
  unknown labels are present.
- The exact models requested are trained. If XGBoost or LightGBM are missing the script
  will skip those models but continue.
- For each model the classification report is saved as text and confusion matrix saved as PNG.
- A channel-level log file is created listing train/test subject IDs, which labels they
  belong to, and for each subject the start and end row indices in the CSV (0-based and 1-based).


"""

import os
import sys
import glob
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


RANDOM_STATE = 42
MERGED_DIR = "Merged_Features_AllMethods"  # change if your folder is elsewhere
OUTPUT_ROOT = "ML-MODEL-RESULTS-PerChannel_V2"
TEST_SIZE = 0.2
LABEL_COL = 'label'  # expected label column name in CSV
META_COLS = ['subj_id', 'window_idx', 'label', 'pain_score']

models = {
    # Linear Models
    "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
    "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=RANDOM_STATE),

    # Tree-Based Models
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),

    # Boosting Models
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE),
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE,
        eval_metric="logloss", use_label_encoder=False
    )
else:
    print("[WARN] xgboost not available; skipping XGBoost model.")

if HAS_LGBM:
    models["LGBM"] = LGBMClassifier(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
else:
    print("[WARN] lightgbm not available; skipping LGBM model.")

# Other Models (added after to ensure optional libs are detected first)
models.update({
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "KNearestNeighbors": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "MLPClassifier": MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
})


os.makedirs(OUTPUT_ROOT, exist_ok=True)


def find_merged_files(merged_dir):
    pattern = os.path.join(merged_dir, "*_merged.csv")
    files = sorted(glob.glob(pattern))
    return files


def infer_channel_name(filename):
    # filename like /path/to/AFz_merged.csv -> AFz
    base = os.path.basename(filename)
    if base.endswith('_merged.csv'):
        return base.replace('_merged.csv', '')
    return os.path.splitext(base)[0]


def check_required_columns(df):
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(
                f"Required metadata column '{c}' not found in CSV")


def map_labels(series):
    # map 'low','mid','high' (case-insensitive) -> 0,1,2
    mapping = {'low': 0, 'mid': 1, 'high': 2}

    def _map_val(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            key = v.strip().lower()
            if key in mapping:
                return mapping[key]
            # if it's numeric string like '0' '1' then try int
            if key.isdigit():
                return int(key)
            raise ValueError(f"Unknown label string: {v}")
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise ValueError(f"Unsupported label value type: {v} (type={type(v)})")

    return series.map(_map_val)


def impute_subject_median(df, feature_cols):
    # For each subject, fill NaNs in feature cols with subject median; if still NaN, leave for later
    df_imputed = df.copy()
    for subj, subdf_idx in df.groupby('subj_id').groups.items():
        idx = subdf_idx
        med = df.loc[idx, feature_cols].median(axis=0, skipna=True)
        # fillna per subject
        df_imputed.loc[idx, feature_cols] = df.loc[idx,
                                                   feature_cols].fillna(med)
    return df_imputed


def safe_train_test_subject_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    # Assign subjects to train/test while stratifying by subject-level majority label
    subjects = df['subj_id'].unique()
    subj_labels = []
    for s in subjects:
        labs = df.loc[df['subj_id'] == s, LABEL_COL]
        # map to int if needed
        labs_mapped = map_labels(labs)
        if labs_mapped.isna().any():
            raise ValueError(f"NaN label found for subject {s}")
        # use most common label for this subject
        mode_lab = int(labs_mapped.mode().iloc[0])
        subj_labels.append(mode_lab)
    # Now stratify split subjects
    train_subjs, test_subjs = train_test_split(subjects, test_size=test_size, random_state=random_state,
                                               stratify=subj_labels)
    return list(train_subjs), list(test_subjs)


def ensure_numeric_features(df, exclude_cols=META_COLS):
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    # coerce to numeric where possible
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    return df, feature_cols


def save_channel_log(channel_outdir, df, train_subjs, test_subjs):
    log_lines = []
    log_lines.append(f"Channel log for folder: {channel_outdir}")
    log_lines.append("")
    log_lines.append("Train subjects by label:")
    train_by_label = defaultdict(list)
    for s in train_subjs:
        labs = df.loc[df['subj_id'] == s, LABEL_COL]
        lab_mode = map_labels(labs).mode().iloc[0]
        train_by_label[int(lab_mode)].append(int(s))
    for lab in sorted(train_by_label.keys()):
        log_lines.append(f"  label {lab}: {train_by_label[lab]}")

    log_lines.append("")
    log_lines.append("Test subjects by label:")
    test_by_label = defaultdict(list)
    for s in test_subjs:
        labs = df.loc[df['subj_id'] == s, LABEL_COL]
        lab_mode = map_labels(labs).mode().iloc[0]
        test_by_label[int(lab_mode)].append(int(s))
    for lab in sorted(test_by_label.keys()):
        log_lines.append(f"  label {lab}: {test_by_label[lab]}")

    log_lines.append("")
    log_lines.append(
        "Detailed per-subject row ranges (0-based index and 1-based):")
    for s in sorted(df['subj_id'].unique(), key=int):
        idxs = df.index[df['subj_id'] == s]
        if len(idxs) == 0:
            continue
        start0 = int(idxs[0])
        end0 = int(idxs[-1])
        log_lines.append(
            f"  subj {s}: rows 0-based [{start0}, {end0}] 1-based [{start0+1}, {end0+1}]  total_rows={len(idxs)}")

    # leakage check
    intersect = set(train_subjs).intersection(set(test_subjs))
    log_lines.append("")
    if intersect:
        log_lines.append(
            f"[ERROR] Subjects in both train and test: {sorted(list(intersect))}")
    else:
        log_lines.append(
            "No subject-level leakage detected (train and test subject sets disjoint).")

    # write log to file
    log_path = os.path.join(channel_outdir, 'split_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Wrote channel log to {log_path}")


def run_models_for_channel(csv_path):
    channel_name = infer_channel_name(csv_path)
    print('\n' + '='*80)
    print(f"Processing channel: {channel_name}")

    channel_outdir = os.path.join(OUTPUT_ROOT, channel_name)
    os.makedirs(channel_outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Read {csv_path} with shape {df.shape}")

    # basic checks
    check_required_columns(df)

    # ensure label mapping
    df[LABEL_COL] = map_labels(df[LABEL_COL])

    # ensure numeric features and get feature column list
    df, feature_cols = ensure_numeric_features(df, exclude_cols=META_COLS)
    print(f"Detected {len(feature_cols)} feature columns")

    # impute per-subject median (so we don't drop rows)
    df = impute_subject_median(df, feature_cols)

    # if any remaining NaNs, they'll be filled using global median of training set later

    # train/test split by subjects with stratification by majority subject label
    train_subjs, test_subjs = safe_train_test_subject_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # create channel-level split log (includes row ranges)
    save_channel_log(channel_outdir, df, train_subjs, test_subjs)

    # create train/test dataframes
    train_df = df[df['subj_id'].isin(train_subjs)].reset_index(drop=True)
    test_df = df[df['subj_id'].isin(test_subjs)].reset_index(drop=True)
    print(
        f"Train rows: {len(train_df)}, Test rows: {len(test_df)}; subjects train={len(train_subjs)}, test={len(test_subjs)}")

    # final NaN handling: use feature-wise median from train
    train_feature_medians = train_df[feature_cols].median(axis=0, skipna=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(
        train_feature_medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_feature_medians)

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df[LABEL_COL].astype(int).values
    y_test = test_df[LABEL_COL].astype(int).values

    # save scaler
    with open(os.path.join(channel_outdir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # for reproducibility save train/test subject lists
    with open(os.path.join(channel_outdir, 'train_subjects.txt'), 'w') as f:
        f.write('\n'.join(map(str, sorted(train_subjs))))
    with open(os.path.join(channel_outdir, 'test_subjects.txt'), 'w') as f:
        f.write('\n'.join(map(str, sorted(test_subjs))))

    for model_name, model in models.items():
        print(f" Training model: {model_name}")
        try:
            clf = model
            clf.fit(X_train, y_train)
        except Exception as e:
            print(
                f"[ERROR] Failed to train model {model_name} on channel {channel_name}: {e}")
            continue

        # predictions
        try:
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"[ERROR] Prediction failed for model {model_name}: {e}")
            continue

        # classification report
        creport = classification_report(y_test, y_pred, digits=4)
        report_path = os.path.join(
            channel_outdir, f"{model_name}_classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(creport)
        print(f" Wrote classification report to {report_path}")

        # confusion matrix plot
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                          'low(0)', 'mid(1)', 'high(2)'])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax)
            plt.title(f"{channel_name} - {model_name}")
            fig_path = os.path.join(
                channel_outdir, f"{model_name}_confusion_matrix.png")
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            print(f" Wrote confusion matrix image to {fig_path}")
        except Exception as e:
            print(
                f"[WARN] Could not create confusion matrix for {model_name}: {e}")

        # optionally save the trained model
        try:
            with open(os.path.join(channel_outdir, f"{model_name}_model.pkl"), 'wb') as f:
                pickle.dump(clf, f)
        except Exception as e:
            print(f"[WARN] Could not pickle model {model_name}: {e}")

    print(f"Finished channel: {channel_name}")


def main():
    files = find_merged_files(MERGED_DIR)
    if not files:
        print(f"No '_merged.csv' files found in directory: {MERGED_DIR}")
        sys.exit(1)

    print(
        f"Found {len(files)} merged channel files. Processing each file individually...")

    for fpath in files:
        try:
            run_models_for_channel(fpath)
        except Exception as e:
            print(f"[ERROR] Failed processing file {fpath}: {e}")

    print('\nAll done. Results are under: ', OUTPUT_ROOT)


if __name__ == '__main__':
    main()
