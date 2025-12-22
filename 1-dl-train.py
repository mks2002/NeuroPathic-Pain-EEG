import os
import re
import numpy as np
import mne
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_ROOT = Path("Segments")
RESULT_ROOT = Path("DL-Result-V1")

FS = 250
WINDOW_SEC = 5
WINDOW_SAMPLES = WINDOW_SEC * FS
STEP_SAMPLES = WINDOW_SAMPLES // 2  # 50% overlap

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'M1', 'M2', 'AFz', 'CPz', 'POz'
]

# ==========================================================
# PAIN SCORE & LABEL MAPPING
# ==========================================================
PAIN_SCORE = {
    0: 7, 1: 4, 2: 3, 3: 8, 4: 5, 5: 2, 6: 7, 7: 3, 8: 4, 9: 9,
    10: 3, 11: 6, 13: 3, 14: 3, 15: 8, 16: 5, 18: 5, 19: 8, 20: 7,
    21: 6, 22: 7, 23: 6, 24: 9, 25: 8, 26: 0, 27: 1, 30: 3, 31: 9,
    33: 6, 35: 1, 37: 0, 38: 7, 39: 8, 40: 4, 41: 7, 43: 6
}


def pain_label_v1(score):
    if score in (1, 2, 3, 4):
        return "low"
    if score in (5, 6):
        return "mid"
    if score in (7, 8, 9):
        return "high"
    return None


LABEL_MAP = {"low": 0, "mid": 1, "high": 2}

# ==========================================================
# MODEL DEFINITIONS
# ==========================================================


def build_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 7, activation='relu',
                               input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 7, activation='relu',
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================


def extract_subject_id(fname):
    return int(re.findall(r'\d+', fname)[0])


def sliding_windows(signal):
    windows = []
    for start in range(0, len(signal) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        windows.append(signal[start:start+WINDOW_SAMPLES])
    return np.array(windows)


def plot_and_save_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================================
# MAIN PIPELINE
# ==========================================================
for condition in ["EO", "EC"]:
    print(f"\n=== PROCESSING {condition} ===")
    cond_dir = DATA_ROOT / condition
    out_cond_dir = RESULT_ROOT / condition
    out_cond_dir.mkdir(parents=True, exist_ok=True)

    subjects = []
    labels = []

    for f in cond_dir.glob("*.fif"):
        sid = extract_subject_id(f.name)
        if sid not in PAIN_SCORE:
            continue
        label = pain_label_v1(PAIN_SCORE[sid])
        if label is None:
            continue
        subjects.append(sid)
        labels.append(label)

    subjects = np.array(subjects)
    labels = np.array(labels)

    # -------- SUBJECT-LEVEL STRATIFIED SPLIT --------
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(sss1.split(subjects, labels))

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),
        random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(sss2.split(
        subjects[train_val_idx], labels[train_val_idx]))

    train_subjects = subjects[train_val_idx][train_idx]
    val_subjects = subjects[train_val_idx][val_idx]
    test_subjects = subjects[test_idx]

    # -------- LOOP OVER CHANNELS --------
    for ch in CHANNELS:
        print(f"\n--- Channel: {ch} ---")
        ch_dir = out_cond_dir / ch
        ch_dir.mkdir(exist_ok=True)

        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []

        for f in cond_dir.glob("*.fif"):
            sid = extract_subject_id(f.name)
            if sid not in PAIN_SCORE:
                continue
            label_name = pain_label_v1(PAIN_SCORE[sid])
            if label_name is None:
                continue
            label = LABEL_MAP[label_name]

            raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
            raw.pick_channels([ch])
            data = raw.get_data()[0]

            windows = sliding_windows(data)

            if sid in train_subjects:
                X_train.append(windows)
                y_train.extend([label]*len(windows))
            elif sid in val_subjects:
                X_val.append(windows)
                y_val.extend([label]*len(windows))
            elif sid in test_subjects:
                X_test.append(windows)
                y_test.extend([label]*len(windows))

        X_train = np.vstack(X_train)
        X_val = np.vstack(X_val)
        X_test = np.vstack(X_test)

        # -------- NORMALIZATION (TRAIN ONLY) --------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # -------- CLASS WEIGHTS --------
        cw = Counter(y_train)
        class_weights = {k: max(cw.values())/v for k, v in cw.items()}

        # ==================================================
        # MODEL 1: CNN
        # ==================================================
        cnn = build_cnn(X_train.shape[1:])
        cnn.fit(
            X_train, np.array(y_train),
            validation_data=(X_val, np.array(y_val)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=0
        )

        y_pred = np.argmax(cnn.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, digits=4)

        with open(ch_dir / "cnn_report.txt", "w") as f:
            f.write(report)

        plot_and_save_confusion(
            y_test, y_pred, ch_dir / "cnn_confusion.png"
        )

        # ==================================================
        # MODEL 2: CNN + LSTM
        # ==================================================
        cnn_lstm = build_cnn_lstm(X_train.shape[1:])
        cnn_lstm.fit(
            X_train, np.array(y_train),
            validation_data=(X_val, np.array(y_val)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=0
        )

        y_pred = np.argmax(cnn_lstm.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, digits=4)

        with open(ch_dir / "cnn_lstm_report.txt", "w") as f:
            f.write(report)

        plot_and_save_confusion(
            y_test, y_pred, ch_dir / "cnn_lstm_confusion.png"
        )

print("\n✅ ALL EXPERIMENTS COMPLETED")
