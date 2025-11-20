# this is the main file in which all channels are processed and features are extracted it is 3 class classification based ..


import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import stats, signal
import pywt

# ======================================================
#           BASIC CONFIGURATION
# ======================================================

SFREQ = 250
WIN_SECS = 5
WIN_SAMPLES = WIN_SECS * SFREQ
STEP = WIN_SAMPLES // 2  # 50% overlap

# Data dir stays constant
DATA_DIR = Path("Segment-Joined")


# ======================================================
#           USER SELECTION + CONFIGURATION
# ======================================================


# version 1
def pain_label_v1(score):
    """Convert numeric pain score into categorical label."""
    if score in (1, 2, 3, 4):
        return "low"
    elif score in (5, 6):
        return "mid"
    elif score in (7, 8, 9):
        return "high"
    else:
        return None  # exclude score=0

# version 2


def pain_label_v2(score):
    """Convert numeric pain score into categorical label."""
    if score in (0, 1, 2, 3):
        return "low"
    elif score in (4, 5, 6):
        return "mid"
    elif score in (7, 8, 9):
        return "high"
    else:
        return None  # this case never occurs

# For Binary classification V1


def pain_label_binary_v1(score):
    """Convert numeric pain score into categorical label."""
    if score in (1, 2, 3, 4, 5):
        return "low"
    elif score in (6, 7, 8, 9):
        return "high"
    else:
        return None  # exclude score=0

# For Binary classification V2


def pain_label_binary_v2(score):
    """Convert numeric pain score into categorical label."""
    if score in (0, 1, 2, 3, 4):
        return "low"
    elif score in (5, 6, 7, 8, 9):
        return "high"
    else:
        return None  # this case never occurs


print("\nSelect Mode:")
print("1 → 3-class WITHOUT 0  (V1)")
print("2 → 3-class WITH 0     (V2)")
print("3 → Binary WITHOUT 0   (Binary V1)")
print("4 → Binary WITH 0      (Binary V2)")

choice = input("\nEnter choice (1/2/3/4): ").strip()

if choice == "1":
    pain_label = pain_label_v1
    MAIN_OUT_DIR = Path("ML-Features-V1")
elif choice == "2":
    pain_label = pain_label_v2
    MAIN_OUT_DIR = Path("ML-Features-V2")
elif choice == "3":
    pain_label = pain_label_binary_v1
    MAIN_OUT_DIR = Path("ML-Features-Binary-V1")
elif choice == "4":
    pain_label = pain_label_binary_v2
    MAIN_OUT_DIR = Path("ML-Features-Binary-V2")
else:
    raise ValueError("Invalid choice — must be 1, 2, 3, or 4.")

# Create output directory
MAIN_OUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"\nSelected labeling function: {pain_label.__name__}")
print(f"Output directory set to: {MAIN_OUT_DIR}\n")


# ======================================================
# Channel information
# ======================================================
CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'M1',
            'M2', 'AFz', 'CPz', 'POz']
CHANNEL_INDEX_MAP = {ch.upper(): i for i, ch in enumerate(CHANNELS)}


# ======================================================
# Pain score mapping
# ======================================================
PAIN_SCORE = {
    0: 7, 1: 4, 2: 3, 3: 8, 4: 5, 5: 2, 6: 7, 7: 3, 8: 4, 9: 9,
    10: 3, 11: 6, 13: 3, 14: 3, 15: 8, 16: 5, 18: 5, 19: 8, 20: 7,
    21: 6, 22: 7, 23: 6, 24: 9, 25: 8, 26: 0, 27: 1, 30: 3, 31: 9,
    33: 6, 35: 1, 37: 0, 38: 7, 39: 8, 40: 4, 41: 7, 43: 6
}


# ======================================================
#               HELPER FUNCTIONS
# ======================================================


try:
    from scipy.stats import median_abs_deviation as mad_func
except Exception:
    def mad_func(x):
        med = np.median(x)
        return np.median(np.abs(x - med))


def safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def bandpower_welch(x, sf, band, nperseg=None):
    nperseg = nperseg or min(sf * 2, len(x))
    f, Pxx = signal.welch(x, sf, nperseg=nperseg)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return float(np.trapezoid(Pxx[idx], f[idx])) if np.any(idx) else 0.0


def hjorth_params(x):
    dx = np.diff(x)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(np.diff(dx)) if len(dx) > 1 else 0.0
    activity = var0
    mobility = np.sqrt(safe_div(var1, var0))
    complexity = np.sqrt(safe_div(var2, var1)) / (mobility + 1e-12)
    return activity, mobility, complexity


def teager_kaiser(x):
    if len(x) < 3:
        return 0.0
    return float(np.mean(x[1:-1]**2 - x[:-2]*x[2:]))


# ======================================================
#            FEATURE EXTRACTION CORE
# ======================================================
def compute_features_window(x, sf, channel):
    """Compute all features for one time window for one channel."""
    x = np.asarray(x, dtype=float)
    feats = {}

    # ---------- Statistical ----------
    mean = np.mean(x)
    std = np.std(x)
    feats.update({
        f"mean_{channel}": mean,
        f"std_{channel}": std,
        f"var_{channel}": np.var(x),
        f"min_{channel}": np.min(x),
        f"max_{channel}": np.max(x),
        f"median_{channel}": np.median(x),
        f"skew_{channel}": float(stats.skew(x)),
        f"kurtosis_{channel}": float(stats.kurtosis(x)),
        f"energy_{channel}": float(np.sum(x**2)),
        f"rms_{channel}": float(np.sqrt(np.mean(x**2))),
        f"trim_mean_10_{channel}": float(stats.trim_mean(x, 0.1)),
        f"trim_mean_15_{channel}": float(stats.trim_mean(x, 0.15)),
        f"mad_{channel}": float(mad_func(x)),
        f"p10_{channel}": float(np.percentile(x, 10)),
        f"p25_{channel}": float(np.percentile(x, 25)),
        f"p75_{channel}": float(np.percentile(x, 75)),
        f"iqr_{channel}": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"cv_{channel}": float(safe_div(std, mean))
    })

    # ---------- Time domain ----------
    diff = np.diff(x)
    feats.update({
        f"mean_diff_{channel}": float(np.mean(diff)),
        f"std_diff_{channel}": float(np.std(diff)),
        f"mean_abs_diff_{channel}": float(np.mean(np.abs(diff))),
        f"max_abs_diff_{channel}": float(np.max(np.abs(diff))),
        f"zero_cross_{channel}": int(np.sum(np.diff(np.sign(x)) != 0)),
        f"signal_energy_{channel}": float(np.sum(x**2)),
        f"sign_change_rate_{channel}": float(np.mean(np.diff(np.sign(diff)) != 0)) if len(diff) > 1 else 0.0,
        f"autocorr_lag1_{channel}": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0,
        f"tk_energy_{channel}": float(teager_kaiser(x))
    })

    # ---------- Frequency domain ----------
    delta = bandpower_welch(x, sf, (0.5, 4))
    theta = bandpower_welch(x, sf, (4, 8))
    alpha = bandpower_welch(x, sf, (8, 13))
    beta = bandpower_welch(x, sf, (13, 30))
    gamma = bandpower_welch(x, sf, (30, 45))
    total = delta + theta + alpha + beta + gamma + 1e-12
    feats.update({
        f"delta_{channel}": delta,
        f"theta_{channel}": theta,
        f"alpha_{channel}": alpha,
        f"beta_{channel}": beta,
        f"gamma_{channel}": gamma,
        f"rel_delta_{channel}": delta / total,
        f"rel_theta_{channel}": theta / total,
        f"rel_alpha_{channel}": alpha / total,
        f"rel_beta_{channel}": beta / total,
        f"rel_gamma_{channel}": gamma / total,
        f"theta_alpha_ratio_{channel}": safe_div(theta, alpha),
        f"theta_beta_ratio_{channel}": safe_div(theta, beta),
        f"alpha_beta_ratio_{channel}": safe_div(alpha, beta)
    })

    # ---------- Hjorth ----------
    act, mob, comp = hjorth_params(x)
    feats.update({
        f"hjorth_activity_{channel}": act,
        f"hjorth_mobility_{channel}": mob,
        f"hjorth_complexity_{channel}": comp
    })

    # ---------- Spectral ----------
    f, Pxx = signal.welch(x, sf, nperseg=min(sf * 2, len(x)))
    Psum = np.sum(Pxx) + 1e-12
    feats[f"spectral_centroid_{channel}"] = float(np.sum(f * Pxx) / Psum)
    pnorm = Pxx / Psum
    feats[f"spectral_entropy_{channel}"] = float(
        -np.sum(pnorm * np.log(pnorm + 1e-12)))
    cumsum = np.cumsum(Pxx)
    idx90 = np.where(cumsum >= 0.9 * np.sum(Pxx))[0]
    feats[f"spectral_edge_90_{channel}"] = float(
        f[idx90[0]]) if len(idx90) else float(f[-1])

    # ---------- Wavelet ----------
    try:
        coeffs = pywt.wavedec(x, 'db4', level=4)
        for i, c in enumerate(coeffs):
            feats[f"wavelet_energy_L{i}_{channel}"] = float(
                np.sum(np.square(c)))
    except Exception:
        for i in range(5):
            feats[f"wavelet_energy_L{i}_{channel}"] = 0.0

    # ---------- Permutation entropy ----------
    def permutation_entropy(x, order=3, delay=1):
        n = len(x)
        if n < order:
            return 0.0
        perms = {}
        for i in range(n - delay * (order - 1)):
            idx = tuple(np.argsort(x[i:i + delay * order:delay]))
            perms[idx] = perms.get(idx, 0) + 1
        p = np.array(list(perms.values()), dtype=float)
        p /= (np.sum(p) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    feats[f"permutation_entropy_{channel}"] = permutation_entropy(x)

    # ---------- Differential entropy ----------
    varx = np.var(x)
    feats[f"differential_entropy_{channel}"] = float(
        0.5 * np.log(2 * np.pi * np.e * varx + 1e-12)) if varx > 0 else 0.0

    # ---------- Higuchi fractal dimension ----------
    def higuchi_fd(x, kmax=10):
        x = np.asarray(x)
        n = len(x)
        if n < 10:
            return 0.0
        L = []
        x = x - np.mean(x)
        for k in range(1, min(kmax, n // 2) + 1):
            Lk = []
            for m in range(k):
                idx = np.arange(m, n, k)
                xm = x[idx]
                if len(xm) < 2:
                    continue
                diff = np.abs(np.diff(xm))
                Lm = (np.sum(diff) * (n - 1) / (len(xm) * k)) / k
                Lk.append(Lm)
            if len(Lk):
                L.append(np.mean(Lk))
        if len(L) > 1:
            kvals = np.arange(1, len(L) + 1)
            p = np.polyfit(np.log(kvals), np.log(L), 1)
            return float(p[0])
        return 0.0

    feats[f"higuchi_fd_{channel}"] = higuchi_fd(x)
    feats = {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
             for k, v in feats.items()}
    return feats


# ======================================================
#                  MAIN LOOP
# ======================================================
files = sorted(DATA_DIR.glob("ID*_combine.fif"))
if not files:
    raise SystemExit("❌ No combined files found in Segment-Joined folder.")

print(f"✅ Found {len(files)} combined files in {DATA_DIR}")

for channel in CHANNELS:
    print(
        f"\n================= Processing channel {channel} =================")
    channel_dir = MAIN_OUT_DIR / channel
    channel_dir.mkdir(exist_ok=True, parents=True)

    for p in files:
        try:
            # Get subject ID
            digits = ''.join([c for c in p.stem if c.isdigit()])
            if not digits:
                continue
            subj_id = int(digits)

            pain_score = PAIN_SCORE.get(subj_id)
            if pain_score is None:
                continue

            label = pain_label(pain_score)
            if label is None:
                print(f"⚠️ ID{subj_id}: Neutral pain (score=0) skipped")
                continue

            # Load data
            if p.suffix.lower() == ".fif":
                raw = mne.io.read_raw_fif(
                    str(p), preload=True, verbose="ERROR")
                chs = [c.upper() for c in raw.ch_names]
                if channel.upper() not in chs:
                    raise ValueError(
                        f"Channel {channel} not found in {p.name}")
                idx = chs.index(channel.upper())
                data = raw.get_data(picks=[idx]).squeeze()
            else:
                data = np.load(p)
                if data.ndim == 3:
                    data = data.squeeze()
                if data.ndim == 2 and data.shape[0] == len(CHANNELS):
                    ch_idx = CHANNEL_INDEX_MAP[channel.upper()]
                    data = data[ch_idx, :]
                elif data.ndim > 1:
                    data = data.reshape(data.shape[0], -1)[0, :]

            n_samples = len(data)
            if n_samples < WIN_SAMPLES:
                continue

            n_windows = ((n_samples - WIN_SAMPLES) // STEP) + 1
            rows = []

            for w in range(n_windows):
                start = w * STEP
                stop = start + WIN_SAMPLES
                win = data[start:stop]
                feats = compute_features_window(win, sf=SFREQ, channel=channel)
                feats.update({
                    "subj_id": subj_id,
                    "window_idx": w,
                    "label": label,
                    "pain_score": pain_score
                })
                rows.append(feats)

            df = pd.DataFrame(rows)
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cols = ["subj_id", "window_idx", "label", "pain_score"] + \
                   [c for c in df.columns if c not in (
                       "subj_id", "window_idx", "label", "pain_score")]
            df = df[cols]

            out_file = channel_dir / f"ID{subj_id}_feature.csv"
            df.to_csv(out_file, index=False)
            print(f"✅ {channel}: Saved → {out_file} ({len(df)} rows)")

        except Exception as e:
            print(f"❌ Error ({channel}, {p.name}): {e}")

print("\n🎯 All channels processed successfully!")


# files = sorted(DATA_DIR.glob("ID*_combine.*"))
# This code is processed both .fif as well as the .npy files because both are in the Segment-Joined folder and we use .* pattern to capture all available file types .
