import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import stats, signal
from fooof import FOOOF
import pywt

# -------------------------------
# Configurations
# -------------------------------
CHANNEL = "FP1"
DATA_DIR = Path("Segment-Joined")
OUT_DIR = Path(f"{CHANNEL}_FeatureData_Simple_Overlapping")
OUT_DIR.mkdir(exist_ok=True)
SFREQ = 250
WIN_SECS = 5
WIN_SAMPLES = WIN_SECS * SFREQ
STEP = WIN_SAMPLES // 2   # 50% overlap → stride = 625

# Pain scores mapping
PAIN_SCORE = {
    0: 7, 1: 4, 2: 3, 3: 8, 4: 5, 5: 2, 6: 7, 7: 3, 8: 4, 9: 9,
    10: 3, 11: 6, 13: 3, 14: 3, 15: 8, 16: 5, 18: 5, 19: 8, 20: 7,
    21: 6, 22: 7, 23: 6, 24: 9, 25: 8, 26: 0, 27: 1, 30: 3, 31: 9,
    33: 6, 35: 1, 37: 0, 38: 7, 39: 8, 40: 4, 41: 7, 43: 6
}


def pain_label(score):
    if score in (1, 2, 3, 4):
        return "low"
    elif score in ( 5, 6):
        return "mid"
    elif score in (7, 8, 9):
        return "high"


# -------------------------------
# Helper Functions
# -------------------------------
try:
    from scipy.stats import median_abs_deviation as mad_func
except ImportError:
    def mad_func(x):
        med = np.median(x)
        return np.median(np.abs(x - med))


def bandpower_welch(x, sf, band):
    f, Pxx = signal.welch(x, sf, nperseg=sf*2)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapezoid(Pxx[idx], f[idx])


def hjorth_params(x):
    dx = np.diff(x)
    var0 = np.var(x)
    var1 = np.var(dx)
    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(np.var(np.diff(dx)) / var1) / \
        mobility if var1 > 0 else 0
    return activity, mobility, complexity


def teager_kaiser(x):
    return np.mean(x[1:-1]**2 - x[:-2]*x[2:])


def compute_features_window(x, sf=250):
    feats = {}
    # --- Statistical ---
    feats.update({
        "mean": np.mean(x),
        "std": np.std(x),
        "var": np.var(x),
        "min": np.min(x),
        "max": np.max(x),
        "median": np.median(x),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "energy": np.sum(x**2),
        "rms": np.sqrt(np.mean(x**2)),
        "trim_mean_10": stats.trim_mean(x, 0.1),
        "trim_mean_15": stats.trim_mean(x, 0.15),
        "mad": mad_func(x),
        "p10": np.percentile(x, 10),
        "p25": np.percentile(x, 25),
        "p75": np.percentile(x, 75),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "cv": np.std(x)/(np.mean(x)+1e-12)
    })

    # --- Time domain ---
    diff = np.diff(x)
    feats.update({
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
        "mean_abs_diff": np.mean(np.abs(diff)),
        "max_abs_diff": np.max(np.abs(diff)),
        "zero_cross": np.sum(np.diff(np.sign(x)) != 0),
        "signal_energy": np.sum(x**2),
        "sign_change_rate": np.mean(np.diff(np.sign(diff)) != 0),
        "autocorr_lag1": np.corrcoef(x[:-1], x[1:])[0, 1],
        "tk_energy": teager_kaiser(x)
    })

    # --- Frequency domain ---
    delta = bandpower_welch(x, sf, (0.5, 4))
    theta = bandpower_welch(x, sf, (4, 8))
    alpha = bandpower_welch(x, sf, (8, 13))
    beta = bandpower_welch(x, sf, (13, 30))
    gamma = bandpower_welch(x, sf, (30, 45))
    total_power = delta + theta + alpha + beta + gamma + 1e-12
    feats.update({
        "delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma,
        "rel_delta": delta/total_power, "rel_theta": theta/total_power,
        "rel_alpha": alpha/total_power, "rel_beta": beta/total_power, "rel_gamma": gamma/total_power,
        "theta_alpha_ratio": theta/(alpha+1e-12),
        "theta_beta_ratio": theta/(beta+1e-12),
        "alpha_beta_ratio": alpha/(beta+1e-12),
    })

    activity, mobility, complexity = hjorth_params(x)
    feats.update({
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity
    })

    # --- Spectral Features ---
    f, Pxx = signal.welch(x, sf, nperseg=sf*2)
    feats["spectral_centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
    feats["spectral_entropy"] = - \
        np.sum((Pxx / np.sum(Pxx)) * np.log(Pxx / np.sum(Pxx) + 1e-12))
    feats["spectral_edge_90"] = f[np.where(
        np.cumsum(Pxx) >= 0.9 * np.sum(Pxx))[0][0]]

    # --- Wavelet-based energy ---
    coeffs = pywt.wavedec(x, 'db4', level=4)
    for i, c in enumerate(coeffs):
        feats[f"wavelet_energy_L{i}"] = np.sum(np.square(c))

    # --- Nonlinear ---
    def sampen(x, m=2, r=0.2):
        x = np.array(x)
        n = len(x)
        if r is None:
            r = 0.2 * np.std(x) if np.std(x) > 0 else 0.2

        def _phi(m):
            count = 0
            for i in range(n - m):
                xi = x[i:i + m]
                for j in range(i + 1, n - m + 1):
                    xj = x[j:j + m]
                    if np.max(np.abs(xi - xj)) <= r:
                        count += 1
            return count

        try:
            A = _phi(m + 1)
            B = _phi(m)
            return -np.log(A / B) if (A > 0 and B > 0) else 0.0
        except Exception:
            return 0.0

    feats["sample_entropy"] = sampen(x)

    def apen(x, m=2, r=None):  # approximate entropy
        # crude ApEn (slower)
        x = np.array(x)
        n = len(x)
        if r is None:
            r = 0.2 * np.std(x) if np.std(x) > 0 else 0.2

        def _phi(m):
            C = []
            for i in range(n - m + 1):
                xi = x[i:i+m]
                cnt = 0
                for j in range(n - m + 1):
                    xj = x[j:j+m]
                    if np.max(np.abs(xi-xj)) <= r:
                        cnt += 1
                C.append(cnt/(n - m + 1))
            return np.sum(np.log(C)) / (n - m + 1)
        try:
            return float(_phi(m) - _phi(m+1))
        except Exception:
            return 0.0

    feats["approximate_entropy"] = apen(x)

    def permutation_entropy(x, order=3, delay=1):
        # simple permutation entropy
        x = np.array(x)
        n = len(x)
        permutations = {}
        for i in range(n - delay*(order-1)):
            sort_idx = tuple(np.argsort(x[i:i+delay*order:delay]))
            permutations[sort_idx] = permutations.get(sort_idx, 0) + 1
        p = np.array(list(permutations.values()), dtype=float)
        p = p / np.sum(p)
        return float(-np.sum(p * np.log(p + 1e-20)))

    feats["permutation_entropy"] = permutation_entropy(x)

    def differential_entropy(x):
        # Gaussian approx: DE = 0.5 * ln(2*pi*e*var)
        var = np.var(x)
        if var <= 0:
            return 0.0
        return 0.5 * np.log(2 * np.pi * np.e * var)

    feats["differential_entropy"] = differential_entropy(x)

    # Higuchi Fractal Dimension (HFD)
    def higuchi_fd(x, kmax=10):
        # simple implementation
        x = np.asarray(x)
        n = len(x)
        L = []
        x = x - np.mean(x)
        for k in range(1, kmax+1):
            Lk = []
            for m in range(k):
                idx = np.arange(m, n, k)
                xm = x[idx]
                if len(xm) < 2:
                    continue
                diffs = np.abs(np.diff(xm))
                Lm = (np.sum(diffs) * (n - 1) / (len(xm)*k)) / k
                Lk.append(Lm)
            if len(Lk) > 0:
                L.append(np.mean(Lk))
        L = np.array(L)
        kvals = np.arange(1, len(L)+1)
        if len(L) > 1:
            p = np.polyfit(np.log(kvals), np.log(L), 1)
            return float(p[0])
        else:
            return 0.0

    feats["higuchi_fd"] = higuchi_fd(x)

    return feats

# So Now after overlapping the number of rows in csvs is 239 (for all) and 237(for ID0 and ID26)..


# -------------------------------
# MAIN LOOP
# -------------------------------
files = sorted(DATA_DIR.glob("ID*_combine.*"))
if not files:
    print("❌ No combined files found in Segment-Joined folder.")
else:
    print(f"✅ Found {len(files)} combined files in {DATA_DIR}")

for p in files:
    try:
        subj_id = int(''.join([c for c in p.stem if c.isdigit()]))
        pain_score = PAIN_SCORE.get(subj_id)
        if pain_score is None:
            print(f"⚠️ Skipping {p.name}: No pain score.")
            continue
        label = pain_label(pain_score)

        # --- Load ---
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
            if data.shape[0] == 24:
                data = data[0, :]

        # --- Trim lengths ---
        target_len = 148750 if subj_id in (0, 26) else 150000
        data = data[:target_len]
        n_windows = ((target_len - WIN_SAMPLES) // STEP) + 1

        print(
            f"▶️ ID{subj_id} | {n_windows} overlapping windows | {target_len} samples")

        # --- Extract features ---
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

        # --- Save ---
        df = pd.DataFrame(rows)
        cols = ["subj_id", "window_idx", "label", "pain_score"] + \
            [c for c in df.columns if c not in (
                "subj_id", "window_idx", "label", "pain_score")]
        df = df[cols]
        out_file = OUT_DIR / f"ID{subj_id}_feature.csv"
        df.to_csv(out_file, index=False)
        print(f"✅ Saved → {out_file} ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Error processing {p.name}: {e}")

print("\n🎯 All subjects processed successfully!")
