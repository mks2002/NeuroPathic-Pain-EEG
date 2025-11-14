# ==========================================================
# deep_autoencoder_feature_extraction.py
# ==========================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import mne

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = Path("Segment-Joined")
OUT_DIR = Path("DL-Features_V1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'M1', 'M2', 'AFz', 'CPz', 'POz'
]

SFREQ = 250
WIN_SECS = 5
WIN_SAMPLES = WIN_SECS * SFREQ
STEP = WIN_SAMPLES // 2
LATENT_DIM = 20    # features per channel
EPOCHS = 30
BATCH_SIZE = 64

# ------------------------------------------------------
# Pain score mapping
# ------------------------------------------------------
PAIN_SCORE = {
    0: 7, 1: 4, 2: 3, 3: 8, 4: 5, 5: 2, 6: 7, 7: 3, 8: 4, 9: 9,
    10: 3, 11: 6, 13: 3, 14: 3, 15: 8, 16: 5, 18: 5, 19: 8, 20: 7,
    21: 6, 22: 7, 23: 6, 24: 9, 25: 8, 26: 0, 27: 1, 30: 3, 31: 9,
    33: 6, 35: 1, 37: 0, 38: 7, 39: 8, 40: 4, 41: 7, 43: 6
}

def pain_label(score):
    """Convert numeric pain score into categorical label."""
    if score in (1, 2, 3, 4):
        return "low"
    elif score in (5, 6):
        return "mid"
    elif score in (7, 8, 9):
        return "high"
    else:
        return None  # exclude score=0



# ======================================================
#  CBAM BLOCK (Keras 3 / TF 2.16 Compatible)
# ======================================================
def cbam_block(inputs, ratio=8):
    channel = int(inputs.shape[-1])

    # --- Channel Attention ---
    shared_dense_one = layers.Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_two = layers.Dense(channel, activation='sigmoid', use_bias=False)

    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    max_pool = layers.GlobalMaxPooling1D()(inputs)

    avg_dense = shared_dense_two(shared_dense_one(avg_pool))
    max_dense = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = layers.Add()([avg_dense, max_dense])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_refined = layers.Multiply()([inputs, layers.Reshape((1, channel))(channel_attention)])

    # --- Spatial Attention (using Lambda layers) ---
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = layers.Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined = layers.Multiply()([channel_refined, spatial_attention])
    return refined




# ======================================================
#  AUTOENCODER (robust, TF 2.15+ / Keras 3 safe)
# ======================================================
def build_autoencoder(input_shape=(1250, 1), latent_dim=20):
    inp = layers.Input(shape=input_shape)

    # ----- ENCODER -----
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same', strides=2)(x)
    x = cbam_block(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    latent = layers.Dense(latent_dim, activation=None, name='latent')(x)

    # ----- DECODER -----
    x = layers.Dense((input_shape[0] // 4) * 64, activation='relu')(latent)
    x = layers.Reshape((input_shape[0] // 4, 64))(x)
    x = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(1, 3, padding='same', activation='linear')(x)

    # --- Ensure exact output length (1250) ---
    out_len = x.shape[1]
    target_len = input_shape[0]
    if out_len is not None and out_len > target_len:
        crop = int(out_len - target_len)
        x = layers.Cropping1D((0, crop))(x)
    elif out_len is not None and out_len < target_len:
        pad = int(target_len - out_len)
        x = layers.ZeroPadding1D((0, pad))(x)

    model = models.Model(inp, x, name='conv_cbam_autoencoder')
    encoder = models.Model(inp, latent, name='encoder')
    model.compile(optimizer='adam', loss='mse')
    return model, encoder



# ======================================================
# LOAD EEG FILES AND WINDOWING
# ======================================================
def load_channel_data(file_path, channel_name):
    if file_path.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
        chs = [c.upper() for c in raw.ch_names]
        data = raw.get_data(picks=[chs.index(channel_name.upper())]).squeeze()
    else:
        arr = np.load(file_path)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim == 2:
            ch_idx = CHANNELS.index(channel_name)
            data = arr[ch_idx, :]
        else:
            data = arr
    return np.asarray(data, dtype=np.float32)

def create_windows(data, win_size, step):
    n = len(data)
    windows = []
    for start in range(0, n - win_size + 1, step):
        end = start + win_size
        windows.append(data[start:end])
    return np.stack(windows, axis=0)

# ======================================================
# MAIN LOOP
# ======================================================
files = sorted(DATA_DIR.glob("ID*_combine.*"))
if not files:
    raise SystemExit("❌ No combined files found in Segment-Joined folder.")

print(f"Found {len(files)} combined subject files.")

for ch in CHANNELS:
    print(f"\n========== Processing Channel {ch} ==========")
    ch_out = OUT_DIR / ch
    ch_out.mkdir(parents=True, exist_ok=True)

    all_windows = []
    subj_ids = []
    subj_window_counts = {}

    # --- gather windows from all subjects to train AE ---
    for p in files:
        subj_id = int(''.join([c for c in p.stem if c.isdigit()]))
        data = load_channel_data(p, ch)
        if len(data) < WIN_SAMPLES:
            continue
        windows = create_windows(data, WIN_SAMPLES, STEP)
        all_windows.append(windows)
        subj_ids.append(subj_id)
        subj_window_counts[subj_id] = len(windows)

    all_windows = np.concatenate(all_windows, axis=0)
    all_windows = np.expand_dims(all_windows, -1)
    print(f"{ch}: total windows={len(all_windows)}, shape={all_windows.shape}")

    # --- build & train autoencoder ---
    ae, encoder = build_autoencoder((WIN_SAMPLES,1), LATENT_DIM)
    early = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ae.fit(
        all_windows, all_windows,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=1, shuffle=True, callbacks=[early]
    )

    # --- extract latent features for each subject ---
    start_idx = 0
    for subj_id in subj_ids:
        n_win = subj_window_counts[subj_id]
        subj_data = all_windows[start_idx:start_idx+n_win]
        start_idx += n_win

        latent = encoder.predict(subj_data, batch_size=128, verbose=0)

        # Get pain score and label
        pain_score = PAIN_SCORE.get(subj_id, None)
        label = pain_label(pain_score)

        if label is None:
            print(f"⚠️ Skipping subject {subj_id} (pain_score={pain_score})")
            continue

        cols = [f"feat{i+1}_{ch}" for i in range(LATENT_DIM)]
        df = pd.DataFrame(latent, columns=cols)
        df.insert(0, "window_idx", np.arange(len(df)))
        df.insert(0, "subj_id", subj_id)
        df["pain_score"] = pain_score
        df["label"] = label

        out_file = ch_out / f"ID{subj_id}_DLfeatures.csv"
        df.to_csv(out_file, index=False)
        print(f"✅ Saved {out_file} ({df.shape})")

print("\n🎯 Deep-learned features (20-dim/channel) + pain labels saved in DL-Features_V1/")





