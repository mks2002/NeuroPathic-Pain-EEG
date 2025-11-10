import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import stats, signal
import pywt

# ======================================================
#                CONFIGURATION
# ======================================================
CHANNEL = "FP1"
DATA_DIR = Path("Segment-Joined")
OUT_DIR = Path(f"{CHANNEL}_FeatureData_Simple_Overlapping_v2")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SFREQ = 250
WIN_SECS = 5
WIN_SAMPLES = WIN_SECS * SFREQ          # 1250
STEP = WIN_SAMPLES // 2                 # 50% overlap → stride = 625

# ------------------------------------------------------
# Channel order (only needed if loading from .npy)
# ------------------------------------------------------
CHANNEL_ORDER = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'M1',
                 'M2', 'AFz', 'CPz', 'POz']
CHANNEL_INDEX_MAP = {ch.upper(): i for i, ch in enumerate(CHANNEL_ORDER)}

# ------------------------------------------------------
# Pain score mapping (from dataset)
# ------------------------------------------------------
PAIN_SCORE = {
    0: 7, 1: 4, 2: 3, 3: 8, 4: 5, 5: 2, 6: 7, 7: 3, 8: 4, 9: 9,
    10: 3, 11: 6, 13: 3, 14: 3, 15: 8, 16: 5, 18: 5, 19: 8, 20: 7,
    21: 6, 22: 7, 23: 6, 24: 9, 25: 8, 26: 0, 27: 1, 30: 3, 31: 9,
    33: 6, 35: 1, 37: 0, 38: 7, 39: 8, 40: 4, 41: 7, 43: 6
}

# ======================================================
#                 LABEL FUNCTION
# ======================================================
def pain_label(score):
    """
    Map numeric pain scores to categorical labels.
    Score = 0 → neutral/no pain → excluded.
    """
    if score in (1, 2, 3, 4):
        return "low"
    elif score in (5, 6):
        return "mid"
    elif score in (7, 8, 9):
        return "high"
    else:
        return None   # exclude score=0 or any invalid


# ======================================================
#              FEATURE HELPER FUNCTIONS
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
#               FEATURE EXTRACTION CORE
# ======================================================
def compute_features_window(x, sf=250):
    x = np.asarray(x, dtype=float)
    feats = {}

    # ----- Statistical -----
    mean = np.mean(x)
    std = np.std(x)
    feats.update({
        "mean": mean, "std": std, "var": np.var(x),
        "min": np.min(x), "max": np.max(x),
        "median": np.median(x),
        "skew": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x)),
        "energy": float(np.sum(x**2)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "trim_mean_10": float(stats.trim_mean(x, 0.1)),
        "trim_mean_15": float(stats.trim_mean(x, 0.15)),
        "mad": float(mad_func(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "cv": float(safe_div(std, mean))
    })

    # ----- Time-domain -----
    diff = np.diff(x)
    feats.update({
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "zero_cross": int(np.sum(np.diff(np.sign(x)) != 0)),
        "signal_energy": float(np.sum(x**2)),
        "sign_change_rate": float(np.mean(np.diff(np.sign(diff)) != 0)) if len(diff) > 1 else 0.0,
        "autocorr_lag1": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0,
        "tk_energy": float(teager_kaiser(x))
    })

    # ----- Frequency-domain -----
    delta = bandpower_welch(x, sf, (0.5, 4))
    theta = bandpower_welch(x, sf, (4, 8))
    alpha = bandpower_welch(x, sf, (8, 13))
    beta = bandpower_welch(x, sf, (13, 30))
    gamma = bandpower_welch(x, sf, (30, 45))
    total = delta + theta + alpha + beta + gamma + 1e-12
    feats.update({
        "delta": delta, "theta": theta, "alpha": alpha,
        "beta": beta, "gamma": gamma,
        "rel_delta": delta / total, "rel_theta": theta / total,
        "rel_alpha": alpha / total, "rel_beta": beta / total, "rel_gamma": gamma / total,
        "theta_alpha_ratio": safe_div(theta, alpha),
        "theta_beta_ratio": safe_div(theta, beta),
        "alpha_beta_ratio": safe_div(alpha, beta),
    })

    # ----- Hjorth -----
    act, mob, comp = hjorth_params(x)
    feats.update({
        "hjorth_activity": act,
        "hjorth_mobility": mob,
        "hjorth_complexity": comp
    })

    # ----- Spectral -----
    f, Pxx = signal.welch(x, sf, nperseg=min(sf * 2, len(x)))
    Psum = np.sum(Pxx) + 1e-12
    feats["spectral_centroid"] = float(np.sum(f * Pxx) / Psum)
    pnorm = Pxx / Psum
    feats["spectral_entropy"] = float(-np.sum(pnorm * np.log(pnorm + 1e-12)))
    cumsum = np.cumsum(Pxx)
    idx90 = np.where(cumsum >= 0.9 * np.sum(Pxx))[0]
    feats["spectral_edge_90"] = float(f[idx90[0]]) if len(idx90) else float(f[-1])

    # ----- Wavelet -----
    try:
        coeffs = pywt.wavedec(x, 'db4', level=4)
        for i, c in enumerate(coeffs):
            feats[f"wavelet_energy_L{i}"] = float(np.sum(np.square(c)))
    except Exception:
        for i in range(5):
            feats[f"wavelet_energy_L{i}"] = 0.0

    # ----- Permutation entropy -----
    def permutation_entropy(x, order=3, delay=1):
        n = len(x)
        if n < order:
            return 0.0
        perms = {}
        for i in range(n - delay*(order-1)):
            idx = tuple(np.argsort(x[i:i+delay*order:delay]))
            perms[idx] = perms.get(idx, 0) + 1
        p = np.array(list(perms.values()), dtype=float)
        p /= (np.sum(p) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    feats["permutation_entropy"] = permutation_entropy(x)

    # ----- Differential entropy -----
    varx = np.var(x)
    feats["differential_entropy"] = float(0.5 * np.log(2 * np.pi * np.e * varx + 1e-12)) if varx > 0 else 0.0

    # ----- Higuchi fractal dimension -----
    def higuchi_fd(x, kmax=10):
        x = np.asarray(x)
        n = len(x)
        if n < 10:
            return 0.0
        L = []
        x = x - np.mean(x)
        for k in range(1, min(kmax, n//2) + 1):
            Lk = []
            for m in range(k):
                idx = np.arange(m, n, k)
                xm = x[idx]
                if len(xm) < 2: continue
                diff = np.abs(np.diff(xm))
                Lm = (np.sum(diff) * (n - 1) / (len(xm) * k)) / k
                Lk.append(Lm)
            if len(Lk): L.append(np.mean(Lk))
        if len(L) > 1:
            kvals = np.arange(1, len(L) + 1)
            p = np.polyfit(np.log(kvals), np.log(L), 1)
            return float(p[0])
        return 0.0

    feats["higuchi_fd"] = higuchi_fd(x)

    # Clean NaNs/infs
    for k, v in feats.items():
        feats[k] = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    return feats


# ======================================================
#                  MAIN LOOP
# ======================================================
files = sorted(DATA_DIR.glob("ID*_combine.*"))
if not files:
    raise SystemExit("❌ No combined files found in Segment-Joined folder.")

print(f"✅ Found {len(files)} combined files in {DATA_DIR}")

for p in files:
    try:
        # extract numeric ID
        digits = ''.join([c for c in p.stem if c.isdigit()])
        if not digits:
            print(f"⚠️ Skipping {p.name}: No ID found")
            continue
        subj_id = int(digits)

        # pain info
        pain_score = PAIN_SCORE.get(subj_id)
        if pain_score is None:
            print(f"⚠️ Skipping {p.name}: Pain score missing.")
            continue

        label = pain_label(pain_score)
        if label is None:
            print(f"⚠️ Skipping ID{subj_id}: Neutral (score={pain_score})")
            continue

        # ------------------------------
        # LOAD SIGNAL
        # ------------------------------
        if p.suffix.lower() == ".fif":
            raw = mne.io.read_raw_fif(str(p), preload=True, verbose="ERROR")
            chs = [c.upper() for c in raw.ch_names]
            if CHANNEL.upper() not in chs:
                raise ValueError(f"Channel {CHANNEL} not found in {p.name}")
            idx = chs.index(CHANNEL.upper())
            data = raw.get_data(picks=[idx]).squeeze()
        else:
            data = np.load(p)
            if data.ndim == 3:
                data = data.squeeze()
            if data.ndim == 2 and data.shape[0] == len(CHANNEL_ORDER):
                ch_idx = CHANNEL_INDEX_MAP.get(CHANNEL.upper(), 0)
                data = data[ch_idx, :]
            elif data.ndim > 1:
                data = data.reshape(data.shape[0], -1)[0, :]

        n_samples = len(data)
        if n_samples < WIN_SAMPLES:
            print(f"⚠️ ID{subj_id} too short ({n_samples}) → skipping")
            continue

        n_windows = ((n_samples - WIN_SAMPLES) // STEP) + 1
        print(f"▶️ ID{subj_id} | {n_windows} overlapping windows | {n_samples} samples")

        # ------------------------------
        # FEATURE EXTRACTION
        # ------------------------------
        rows = []
        for w in range(n_windows):
            start = w * STEP
            stop = start + WIN_SAMPLES
            win = data[start:stop]
            feats = compute_features_window(win, sf=SFREQ)
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
            [c for c in df.columns if c not in ("subj_id", "window_idx", "label", "pain_score")]
        df = df[cols]

        out_file = OUT_DIR / f"ID{subj_id}_feature.csv"
        df.to_csv(out_file, index=False)
        print(f"✅ Saved → {out_file} ({len(df)} rows)\n")

    except Exception as e:
        print(f"❌ Error processing {p.name}: {e}")

print("🎯 All subjects processed successfully!")
