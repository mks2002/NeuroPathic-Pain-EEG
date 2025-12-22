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
RESULT_ROOT = Path("DL-Result-V2")

FS = 250
WINDOW_SEC = 2
WINDOW_SAMPLES = WINDOW_SEC * FS
STEP_SAMPLES = WINDOW_SAMPLES // 2  # 50% overlap

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EPOCHS = 40
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
# TRANSFORMER MODEL
# ==========================================================


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(inputs.shape[-1])(ff)
    x = tf.keras.layers.Add()([x, ff])
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)


def build_transformer(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Dense(64)(inputs)   # embedding
    for _ in range(2):
        x = transformer_encoder(
            x,
            head_size=32,
            num_heads=4,
            ff_dim=128
        )

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================


def extract_subject_id(fname):
    return int(re.findall(r'\d+', fname)[0])


def sliding_windows(signal):
    return np.array([
        signal[i:i+WINDOW_SAMPLES]
        for i in range(0, len(signal)-WINDOW_SAMPLES+1, STEP_SAMPLES)
    ])


def save_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ==========================================================
# MAIN PIPELINE
# ==========================================================
for condition in ["EO", "EC"]:
    print(f"\n=== TRANSFORMER | {condition} ===")

    cond_dir = DATA_ROOT / condition
    out_cond_dir = RESULT_ROOT / condition
    out_cond_dir.mkdir(parents=True, exist_ok=True)

    subjects, labels = [], []

    for f in cond_dir.glob("*.fif"):
        sid = extract_subject_id(f.name)
        if sid not in PAIN_SCORE:
            continue
        lbl = pain_label_v1(PAIN_SCORE[sid])
        if lbl is None:
            continue
        subjects.append(sid)
        labels.append(lbl)

    subjects = np.array(subjects)
    labels = np.array(labels)

    # -------- SUBJECT-LEVEL STRATIFIED SPLIT --------
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(sss1.split(subjects, labels))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),
        random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        sss2.split(subjects[train_val_idx], labels[train_val_idx])
    )

    train_sub = subjects[train_val_idx][train_idx]
    val_sub = subjects[train_val_idx][val_idx]
    test_sub = subjects[test_idx]

    # -------- LOOP OVER CHANNELS --------
    for ch in CHANNELS:
        print(f"Channel: {ch}")
        ch_dir = out_cond_dir / ch
        ch_dir.mkdir(exist_ok=True)

        Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []

        for f in cond_dir.glob("*.fif"):
            sid = extract_subject_id(f.name)
            if sid not in PAIN_SCORE:
                continue
            lbl_name = pain_label_v1(PAIN_SCORE[sid])
            if lbl_name is None:
                continue
            lbl = LABEL_MAP[lbl_name]

            raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
            raw.pick_channels([ch])
            sig = raw.get_data()[0]

            wins = sliding_windows(sig)

            if sid in train_sub:
                Xtr.append(wins)
                ytr += [lbl]*len(wins)
            elif sid in val_sub:
                Xva.append(wins)
                yva += [lbl]*len(wins)
            elif sid in test_sub:
                Xte.append(wins)
                yte += [lbl]*len(wins)

        Xtr = np.vstack(Xtr)
        Xva = np.vstack(Xva)
        Xte = np.vstack(Xte)

        # -------- NORMALIZATION --------
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)

        Xtr = Xtr[..., np.newaxis]
        Xva = Xva[..., np.newaxis]
        Xte = Xte[..., np.newaxis]

        # -------- CLASS WEIGHTS --------
        cnt = Counter(ytr)
        class_weights = {k: max(cnt.values())/v for k, v in cnt.items()}

        # -------- TRAIN TRANSFORMER --------
        model = build_transformer(Xtr.shape[1:])
        model.fit(
            Xtr, np.array(ytr),
            validation_data=(Xva, np.array(yva)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=0
        )

        # -------- EVALUATION --------
        y_pred = np.argmax(model.predict(Xte), axis=1)
        report = classification_report(yte, y_pred, digits=4)

        with open(ch_dir / "transformer_report.txt", "w") as f:
            f.write(report)

        save_confusion(yte, y_pred, ch_dir / "transformer_confusion.png")

print("\n✅ TEMPORAL TRANSFORMER PIPELINE COMPLETED")
