# ==========================================================
# deep_autoencoder_feature_extraction_with_validation.py
# ==========================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import mne

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = Path("Segment-Joined")
OUT_DIR = Path("DL-Features_V1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz',
    'M1','M2','AFz','CPz','POz'
]

SFREQ = 250
WIN_SECS = 5
WIN_SAMPLES = WIN_SECS * SFREQ
STEP = WIN_SAMPLES // 2
LATENT_DIM = 20
EPOCHS = 30
BATCH_SIZE = 64

# ======================================================
# PAIN SCORE
# ======================================================
PAIN_SCORE = {
    0:7,1:4,2:3,3:8,4:5,5:2,6:7,7:3,8:4,9:9,
    10:3,11:6,13:3,14:3,15:8,16:5,18:5,19:8,20:7,
    21:6,22:7,23:6,24:9,25:8,26:0,27:1,30:3,31:9,
    33:6,35:1,37:0,38:7,39:8,40:4,41:7,43:6
}

def pain_label(score):
    if score in (1,2,3,4): return "low"
    if score in (5,6): return "mid"
    if score in (7,8,9): return "high"
    return None

# ======================================================
# CBAM BLOCK
# ======================================================
def cbam_block(inputs, ratio=8):
    ch = int(inputs.shape[-1])
    dense1 = layers.Dense(ch//ratio, activation='relu', use_bias=False)
    dense2 = layers.Dense(ch, activation='sigmoid', use_bias=False)

    avg = layers.GlobalAveragePooling1D()(inputs)
    mx = layers.GlobalMaxPooling1D()(inputs)

    ca = layers.Add()([dense2(dense1(avg)), dense2(dense1(mx))])
    ca = layers.Reshape((1,ch))(ca)
    x = layers.Multiply()([inputs, ca])

    sa_avg = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    sa_max = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    sa = layers.Conv1D(1, 7, padding='same', activation='sigmoid')(
        layers.Concatenate()([sa_avg, sa_max])
    )
    return layers.Multiply()([x, sa])

# ======================================================
# AUTOENCODER
# ======================================================
def build_autoencoder():
    inp = layers.Input(shape=(WIN_SAMPLES,1))
    x = layers.Conv1D(32,7,activation='relu',padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64,5,strides=2,activation='relu',padding='same')(x)
    x = cbam_block(x)
    x = layers.Conv1D(128,3,strides=2,activation='relu',padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    latent = layers.Dense(LATENT_DIM,name="latent")(x)

    x = layers.Dense((WIN_SAMPLES//4)*64,activation='relu')(latent)
    x = layers.Reshape((WIN_SAMPLES//4,64))(x)
    x = layers.Conv1DTranspose(64,3,strides=2,padding='same',activation='relu')(x)
    x = layers.Conv1DTranspose(32,3,strides=2,padding='same',activation='relu')(x)
    out = layers.Conv1D(1,3,padding='same')(x)

    ae = models.Model(inp,out)
    encoder = models.Model(inp,latent)
    ae.compile(optimizer='adam', loss='mse')
    return ae, encoder

# ======================================================
# LOAD + WINDOW
# ======================================================
def load_channel_data(fp, ch):
    raw = mne.io.read_raw_fif(str(fp), preload=True, verbose=False)
    idx = [c.upper() for c in raw.ch_names].index(ch)
    return raw.get_data(picks=[idx]).squeeze()

def create_windows(x):
    return np.stack([x[i:i+WIN_SAMPLES]
                     for i in range(0,len(x)-WIN_SAMPLES+1,STEP)])

# ======================================================
# MAIN
# ======================================================
files = sorted(DATA_DIR.glob("ID*_combine.fif"))

for ch in CHANNELS:
    print(f"\n===== Channel {ch} =====")
    ch_out = OUT_DIR / ch
    ch_out.mkdir(exist_ok=True)

    windows, subj_map = [], []

    for f in files:
        sid = int(''.join(filter(str.isdigit, f.stem)))
        data = load_channel_data(f, ch)
        if len(data) < WIN_SAMPLES: continue
        w = create_windows(data)
        windows.append(w)
        subj_map += [sid]*len(w)

    windows = np.expand_dims(np.concatenate(windows), -1)

    ae, encoder = build_autoencoder()
    history = ae.fit(
        windows, windows,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
        verbose=1
    )

    # ------------------ LOSS PLOT ------------------
    plt.plot(history.history['loss'])
    plt.title(f"Reconstruction Loss - {ch}")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.savefig(ch_out/"reconstruction_loss.png")
    plt.close()

    # ------------------ RECONSTRUCTION CHECK ------------------
    idx = np.random.randint(len(windows))
    orig = windows[idx:idx+1]
    recon = ae.predict(orig)

    plt.plot(orig[0].squeeze(), label="Original")
    plt.plot(recon[0].squeeze(), label="Reconstructed")
    plt.legend()
    plt.savefig(ch_out/"original_vs_reconstructed.png")
    plt.close()

    # ------------------ METRICS ------------------
    mse = np.mean((orig - recon)**2)
    corr = pearsonr(orig.flatten(), recon.flatten())[0]
    snr = 10*np.log10(np.mean(orig**2)/np.mean((orig-recon)**2))

    pd.DataFrame([{
        "MSE": mse,
        "Correlation": corr,
        "SNR_dB": snr
    }]).to_csv(ch_out/"reconstruction_metrics.csv", index=False)

    # ------------------ LATENT SPACE ------------------
    latent = encoder.predict(windows, batch_size=256)
    pca = PCA(2).fit_transform(latent)

    plt.scatter(pca[:,0], pca[:,1], s=5)
    plt.title(f"Latent Space PCA - {ch}")
    plt.savefig(ch_out/"latent_pca.png")
    plt.close()

print("\n✅ Feature extraction + validation complete.")
