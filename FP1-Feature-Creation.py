"""
Full pipeline to extract ~80 features for a single channel (FP1) from your Segments folder.
Produces per-subject CSVs in folder: FP1_featureData/
"""

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
from scipy.signal import find_peaks, hilbert, welch, stft
import mne
import math
import warnings
warnings.filterwarnings("ignore")

# Optional libs (try import; set flags)
have_pywt = False
have_nolds = False
have_antropy = False
have_statsmodels = False
have_fooof = False
try:
    import pywt
    have_pywt = True
except Exception:
    pass

try:
    import nolds
    have_nolds = True
except Exception:
    pass

try:
    import antropy as ant
    have_antropy = True
except Exception:
    pass

try:
    import statsmodels.api as sm
    have_statsmodels = True
except Exception:
    pass

try:
    from fooof import FOOOF
    have_fooof = True
except Exception:
    pass

# --------------------------
# USER CONFIG
# --------------------------
DATA_DIR = Path("Segments")
OUT_DIR = Path("FP1_featureData")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_FORMAT = "fif"         # 'fif' preferred, fallback to 'npy'
CHANNEL = "FP1"                  # <-- the channel you told me
SFREQ = 250                      # Hz
WIN_SEC = 5                      # seconds
WIN_SAMPLES = WIN_SEC * SFREQ    # 1250
TRIM_EXTRA = 1                   # remove extra 1 sample at end if present

# Pain score mapping (from your message)
PAIN_SCORE = {
    0: 7.0, 1:4.0, 2:3.0, 3:8.0, 4:5.0, 5:2.0, 6:7.0, 7:3.0, 8:4.0, 9:9.0,
    10:3.0,11:6.0,13:3.0,14:3.0,15:8.0,16:5.0,18:5.0,19:8.0,20:7.0,21:6.0,
    22:7.0,23:6.0,24:9.0,25:8.0,26:0.0,27:1.0,30:3.0,31:9.0,33:6.0,35:1.0,
    37:0.0,38:7.0,39:8.0,40:4.0,41:7.0,43:6.0
}
# Label mapping
def pain_label(score):
    if score in (0,1,2):
        return "low"
    if score in (3,4,5,6):
        return "mid"
    return "high"

# helper to extract ID integer from filename 'ID12_EO_raw.fif' etc
def id_from_name(stem):
    s = ''.join(ch for ch in stem if (ch.isdigit() or ch=='_'))
    # find leading number sequence after 'ID'
    import re
    m = re.search(r'ID(\d+)', stem)
    if m:
        return int(m.group(1))
    return None

# ---------------------------------------
# Feature helper functions (many)
# ---------------------------------------

def safe_div(a,b):
    return a/b if (b!=0 and not np.isnan(b)) else 0.0

# Basic statistics
def feat_basic_stats(x):
    x = np.asarray(x)
    res = {}
    res["mean"] = np.mean(x)
    res["std"] = np.std(x, ddof=0)
    res["var"] = np.var(x, ddof=0)
    res["min"] = np.min(x)
    res["max"] = np.max(x)
    res["median"] = np.median(x)
    res["skew"] = float(stats.skew(x))
    res["kurtosis"] = float(stats.kurtosis(x))
    res["energy"] = float(np.sum(x**2))
    res["rms"] = float(np.sqrt(np.mean(x**2)))
    # trimmed means
    res["trimmed_mean_10"] = float(stats.trim_mean(x, 0.1))
    res["trimmed_mean_15"] = float(stats.trim_mean(x, 0.15))
    # MAD
    res["mad"] = float(stats.median_absolute_deviation(x))
    # percentiles
    p10 = np.percentile(x, 10)
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    res["p10"], res["p25"], res["p75"] = float(p10), float(p25), float(p75)
    res["iqr"] = float(p75 - p25)
    res["coeff_var"] = safe_div(res["std"], res["mean"])
    return res

# Time-domain diffs and slopes
def feat_time_domain(x):
    x = np.asarray(x)
    dx = np.diff(x)
    res = {}
    res["mean_diff"] = float(np.mean(dx))
    res["std_diff"] = float(np.std(dx))
    res["mean_abs_diff"] = float(np.mean(np.abs(dx)))
    res["max_abs_diff"] = float(np.max(np.abs(dx)))
    # zero crossing count
    zc = np.sum(np.abs(np.diff(np.sign(x)))>0)
    res["zero_crossings"] = int(zc)
    # slopes (local)
    slopes = dx  # simple difference
    res["slope_mean"] = float(np.mean(slopes))
    res["slope_std"] = float(np.std(slopes))
    res["slope_skew"] = float(stats.skew(slopes))
    res["slope_kurtosis"] = float(stats.kurtosis(slopes))
    # sign change rate
    res["sign_change_rate"] = float(np.mean(np.abs(np.diff((x>0).astype(int)))))
    # autocorrelation lag1
    if len(x) > 1:
        res["autocorr_lag1"] = float(np.corrcoef(x[:-1], x[1:])[0,1])
    else:
        res["autocorr_lag1"] = 0.0
    # partial autocorrelation at lag 1 (approx)
    if have_statsmodels:
        try:
            pacf_vals = sm.tsa.stattools.pacf(x, nlags=1, method='ywunbiased')
            res["pacf_lag1"] = float(pacf_vals[1])
        except Exception:
            res["pacf_lag1"] = 0.0
    else:
        res["pacf_lag1"] = 0.0
    # Teager-Kaiser Energy Operator
    tkeo = np.zeros_like(x)
    if len(x) >= 3:
        tkeo[1:-1] = x[1:-1]**2 - x[0:-2]*x[2:]
        res["tkeo_mean"] = float(np.mean(tkeo[1:-1]))
        res["tkeo_var"] = float(np.var(tkeo[1:-1]))
    else:
        res["tkeo_mean"], res["tkeo_var"] = 0.0, 0.0
    return res

# Amplitude domain features (peaks etc)
def feat_amplitude(x, sf=250):
    x = np.asarray(x)
    res = {}
    res["mean_amplitude"] = float(np.mean(np.abs(x)))
    res["peak_to_peak"] = float(np.ptp(x))
    res["positive_ratio"] = float(np.mean(x>0))
    res["negative_ratio"] = float(np.mean(x<0))
    # amplitude entropy: Shannon entropy over histogram
    p, _ = np.histogram(x, bins=50, density=True)
    p = p + 1e-12
    ent = -np.sum(p * np.log(p))
    res["amplitude_entropy"] = float(ent)
    # peaks
    # default min distance = 0.2s -> 50 samples; changeable
    peaks, props = find_peaks(x, distance=int(0.2*sf), prominence=None)
    res["peak_count"] = int(len(peaks))
    if len(peaks) > 0:
        widths = signal.peak_widths(x, peaks, rel_height=0.5)[0]
        res["peak_width_mean"] = float(np.mean(widths))
        res["peak_width_std"] = float(np.std(widths))
        # rise/fall times: approximate using neighborhoods
        # rise time: distance from left base to peak (samples)
        res["rise_time_mean"] = float(np.mean(widths)/2.0)  # rough proxy
        res["fall_time_mean"] = float(np.mean(widths)/2.0)
    else:
        res["peak_width_mean"] = 0.0
        res["peak_width_std"] = 0.0
        res["rise_time_mean"] = 0.0
        res["fall_time_mean"] = 0.0
    # asymmetry index and amplitude skew
    res["asymmetry_index"] = float((np.mean(x[x>0]) if np.any(x>0) else 0.0) - (np.mean(np.abs(x[x<0])) if np.any(x<0) else 0.0))
    res["amplitude_skew"] = float(stats.skew(x))
    return res

# Frequency / EEG specific features
BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta": (13,30),
    "gamma": (30,45)
}

def bandpower_welch(x, sf=250, band=(1,4), nperseg=1024):
    f, Pxx = welch(x, fs=sf, nperseg=min(nperseg, len(x)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    bp = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
    return bp, f, Pxx

def feat_frequency(x, sf=250):
    x = np.asarray(x)
    res = {}
    # compute PSD with decent resolution
    nperseg = 1024
    f, Pxx = welch(x, fs=sf, nperseg=min(nperseg, len(x)))
    total_power = np.trapz(Pxx, f)
    # band powers
    band_powers = {}
    for name, rng in BANDS.items():
        bp, _, _ = bandpower_welch(x, sf=sf, band=rng, nperseg=nperseg)
        band_powers[name] = bp
        res[f"{name}_power"] = float(bp)
    res["total_power"] = float(total_power)
    # relative powers
    for name in BANDS.keys():
        res[f"{name}_relpower"] = float(safe_div(band_powers[name], total_power))
    # ratios
    res["theta_alpha_ratio"] = float(safe_div(band_powers["theta"], band_powers["alpha"]))
    res["theta_beta_ratio"] = float(safe_div(band_powers["theta"], band_powers["beta"]))
    res["alpha_beta_ratio"] = float(safe_div(band_powers["alpha"], band_powers["beta"]))
    # Hjorth params
    # activity = variance
    res["hjorth_activity"] = float(np.var(x))
    # mobility = sqrt(var(diff(x)) / var(x))
    dx = np.diff(x)
    res["hjorth_mobility"] = float(math.sqrt(safe_div(np.var(dx), np.var(x))))
    # complexity = mobility(diff(x)) / mobility(x)
    ddx = np.diff(dx)
    mobility_dx = math.sqrt(safe_div(np.var(ddx), np.var(dx))) if np.var(dx) > 0 else 0.0
    res["hjorth_complexity"] = float(safe_div(mobility_dx, res["hjorth_mobility"])) if res["hjorth_mobility"]!=0 else 0.0
    # spectral centroid
    if total_power > 0:
        res["spectral_centroid"] = float(np.sum(f * Pxx) / np.sum(Pxx))
    else:
        res["spectral_centroid"] = 0.0
    # spectral entropy
    Pn = Pxx / (np.sum(Pxx)+1e-20)
    res["spectral_entropy"] = float(-np.sum(Pn * np.log(Pn + 1e-20)))
    # spectral edge frequency (95% energy)
    cdf = np.cumsum(Pxx)
    if cdf[-1] > 0:
        sef95_idx = np.searchsorted(cdf, 0.95 * cdf[-1])
        res["SEF_95"] = float(f[sef95_idx]) if sef95_idx < len(f) else float(f[-1])
    else:
        res["SEF_95"] = 0.0
    # 1/f aperiodic exponent (simple robust linear fit on log-log in 2-40 Hz)
    try:
        idx_fit = np.where((f>=2) & (f<=40))[0]
        if len(idx_fit) > 3:
            xf = np.log10(f[idx_fit])
            yf = np.log10(Pxx[idx_fit] + 1e-20)
            # robust: use np.polyfit
            p = np.polyfit(xf, yf, 1)
            res["aperiodic_exponent"] = float(-p[0])  # slope negative -> exponent positive
        else:
            res["aperiodic_exponent"] = 0.0
    except Exception:
        res["aperiodic_exponent"] = 0.0
    return res

# Time-frequency features: wavelet band energy (CWT) or DWT (pywt)
def feat_timefreq(x, sf=250):
    res = {}
    x = np.asarray(x)
    # Wavelet energy by band (if pywt present we can compute DWT energy in bands; else fallback to STFT integrate)
    if have_pywt:
        try:
            # Using discrete wavelet decomposition (db4) levels approximate bands; this is heuristic
            coeffs = pywt.wavedec(x, 'db4', level=5)
            energies = [np.sum(c**2) for c in coeffs]
            # store energies
            for i, e in enumerate(energies):
                res[f"wavelet_energy_L{i}"] = float(e)
        except Exception:
            pass
    # STFT-based energy in bands
    f, t, Z = stft(x, fs=sf, nperseg=256)
    P = np.abs(Z)**2
    # energy in each EEG band via STFT integration (sum over freq bins)
    for name, rng in BANDS.items():
        idx = np.logical_and(f >= rng[0], f <= rng[1])
        if np.any(idx):
            band_energy = np.sum(P[idx,:])
        else:
            band_energy = 0.0
        res[f"stft_{name}_energy"] = float(band_energy)
    # instantaneous frequency via hilbert on bandpassed alpha as example (compute mean)
    try:
        # bandpass 8-13 for inst freq
        b, a = signal.butter(4, [8/(sf/2),13/(sf/2)], btype='band')
        xf = signal.filtfilt(b,a,x)
        ph = np.angle(hilbert(xf))
        inst_freq = np.diff(np.unwrap(ph)) * sf / (2.0*np.pi)
        res["instfreq_alpha_mean"] = float(np.mean(inst_freq)) if inst_freq.size>0 else 0.0
        res["instfreq_alpha_std"] = float(np.std(inst_freq)) if inst_freq.size>0 else 0.0
    except Exception:
        res["instfreq_alpha_mean"] = 0.0
        res["instfreq_alpha_std"] = 0.0
    return res

# Nonlinear complexity features
def samp_entropy(x, m=2, r=None):
    # Basic SampEn implementation
    x = np.array(x)
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x) if np.std(x)>0 else 0.2
    def _phi(m):
        count = 0
        for i in range(n - m):
            xi = x[i:i+m]
            for j in range(i+1, n - m + 1):
                xj = x[j:j+m]
                if np.max(np.abs(xi-xj)) <= r:
                    count += 1
        return count
    try:
        A = _phi(m+1)
        B = _phi(m)
        return -np.log(A / B) if (A>0 and B>0) else 0.0
    except Exception:
        return 0.0

def approx_entropy(x, m=2, r=None):
    # crude ApEn (slower)
    x = np.array(x)
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x) if np.std(x)>0 else 0.2
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

def permutation_entropy(x, order=3, delay=1):
    # simple permutation entropy
    x = np.array(x)
    n = len(x)
    permutations = {}
    for i in range(n - delay*(order-1)):
        sort_idx = tuple(np.argsort(x[i:i+delay*order:delay]))
        permutations[sort_idx] = permutations.get(sort_idx,0) + 1
    p = np.array(list(permutations.values()), dtype=float)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p + 1e-20)))

def lempel_ziv_complexity(x):
    # binary sequence by median
    s = (x > np.median(x)).astype(int)
    # implement simple LZ complexity
    seq = ''.join(map(str, s.tolist()))
    i, k, l = 0,1,1
    complexity = 1
    n = len(seq)
    while True:
        if seq[i+k-1:i+k-1+l] in seq[0:i+k-1]:
            l += 1
            if i + k - 1 + l > n:
                complexity += 1
                break
        else:
            complexity += 1
            i = i + 1
            if i == k:
                k += 1
                i = 0
            l = 1
        if k > n:
            break
    return float(complexity)

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
            if len(xm) < 2: continue
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

def dfa_alpha(x):
    # simple DFA using nolds if present else fallback quick method
    if have_nolds:
        try:
            return float(nolds.dfa(x))
        except Exception:
            pass
    # fallback: return 0
    return 0.0

# Extra novel features (DE, burst detection, relative changes)
def differential_entropy(x):
    # Gaussian approx: DE = 0.5 * ln(2*pi*e*var)
    var = np.var(x)
    if var <= 0:
        return 0.0
    return 0.5 * np.log(2 * np.pi * np.e * var)

def burst_features(x, sf=250, band=(8,13), env_thresh=2.0):
    # detect amplitude bursts in a band (e.g., alpha)
    res = {}
    try:
        b, a = signal.butter(3, [band[0]/(sf/2), band[1]/(sf/2)], btype='band')
        xf = signal.filtfilt(b,a,x)
        env = np.abs(hilbert(xf))
        # threshold = mean + env_thresh*std
        thr = np.mean(env) + env_thresh * np.std(env)
        bursts = env > thr
        # compute burst count and mean duration
        # find rising edges
        idx = np.where(np.diff(bursts.astype(int))==1)[0]
        idx_end = np.where(np.diff(bursts.astype(int))==-1)[0]
        # pair them
        if len(idx_end) < len(idx):
            idx_end = np.append(idx_end, len(env)-1)
        durations = []
        for s,e in zip(idx, idx_end):
            durations.append((e - s)/sf)
        res["burst_count"] = int(len(durations))
        res["burst_dur_mean"] = float(np.mean(durations)) if durations else 0.0
        res["burst_dur_std"] = float(np.std(durations)) if durations else 0.0
    except Exception:
        res["burst_count"] = 0
        res["burst_dur_mean"] = 0.0
        res["burst_dur_std"] = 0.0
    return res

# wrapper to compute all features for a window
def compute_features_window(x, sf=250):
    feats = {}
    feats.update(feat_basic_stats(x))
    feats.update(feat_time_domain(x))
    feats.update(feat_amplitude(x, sf=sf))
    feats.update(feat_frequency(x, sf=sf))
    feats.update(feat_timefreq(x, sf=sf))
    # nonlinear
    feats["sampen_m2_r02"] = float(samp_entropy(x, m=2, r=0.2*np.std(x) if np.std(x)>0 else 0.2))
    feats["ap_en_m2_r02"] = float(approx_entropy(x, m=2, r=0.2*np.std(x) if np.std(x)>0 else 0.2))
    feats["perm_entropy_o3"] = float(permutation_entropy(x, order=3))
    feats["lempel_ziv"] = float(lempel_ziv_complexity(x))
    feats["higuchi_fd"] = float(higuchi_fd(x))
    feats["dfa_alpha"] = float(dfa_alpha(x))
    # extra novel
    feats["diff_entropy"] = float(differential_entropy(x))
    feats.update(burst_features(x, sf=sf, band=(8,13)))
    # relative change features left to session-level calculation (handled outside if needed)
    return feats

# ----------------------------------------------
# Main processing loop - per ID create CSV
# ----------------------------------------------
def find_subject_files(data_dir):
    # returns dict id -> {'EO': path, 'EC': path}
    all_files = list(Path(data_dir).glob("*"))
    mapping = {}
    for p in all_files:
        name = p.name
        # normalize name stamps
        stem = p.stem.upper()
        if "_EO" in stem or "_E0" in stem or "_E O" in stem: # allow slight name variations
            key = stem.split('_')[0]  # ID0 etc
            mapping.setdefault(key, {})['EO'] = p
        elif "_EC" in stem:
            key = stem.split('_')[0]
            mapping.setdefault(key, {})['EC'] = p
    return mapping

def load_channel_from_file(p, channel, prefer='fif'):
    # returns data array shape (n_channels, n_samples) and channel index
    p = Path(p)
    if p.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(str(p), preload=True, verbose='ERROR')
        data = raw.get_data()
        ch_names = [c.upper() for c in raw.ch_names]
        # find requested channel (case-insensitive)
        if channel.upper() in ch_names:
            idx = ch_names.index(channel.upper())
            arr = data[idx, :]
            return arr, idx, raw
        else:
            raise ValueError(f"Channel {channel} not found in {p.name}. Found: {raw.ch_names}")
    elif p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if arr.ndim == 3:
            arr = arr.squeeze()
        # we don't have channel names; assume same channel order as .fif
        # fallback: user must ensure channel order consistent. We'll try to read a corresponding .fif to get order.
        return arr, None, None
    else:
        raise ValueError("Unsupported format")

# find all subject EO/EC files
mapping = find_subject_files(DATA_DIR)
if not mapping:
    print("No EO/EC files found in Segments/. Please check folder and naming (EO/EC).")
else:
    print(f"Found {len(mapping)} subject entries (by filename stem) in {DATA_DIR}")

# process each subject key (like 'ID0', 'ID1', ...)
for subj_key, files_dict in sorted(mapping.items(), key=lambda x: x[0]):
    try:
        subject_id = id_from_name(subj_key)
        if subject_id is None:
            print(f"Skipping {subj_key} (cannot parse ID).")
            continue
        pain_score = PAIN_SCORE.get(subject_id, None)
        if pain_score is None:
            print(f"Warning: Pain score not found for ID {subject_id}. Skipping.")
            continue
        label = pain_label(pain_score)
        # determine EO and EC file paths; prefer .fif if available
        eo_path = files_dict.get('EO', None)
        ec_path = files_dict.get('EC', None)
        if eo_path is None or ec_path is None:
            print(f"Skipping ID{subject_id}: EO or EC missing (EO={eo_path}, EC={ec_path})")
            continue

        # load channel values
        # prefer fif; if npy then use it but must ensure channel ordering is correct
        def load_arr(p):
            p = Path(p)
            if p.suffix.lower() == ".fif":
                raw = mne.io.read_raw_fif(str(p), preload=True, verbose='ERROR')
                chs = [c.upper() for c in raw.ch_names]
                if CHANNEL.upper() not in chs:
                    raise ValueError(f"{CHANNEL} not in file {p.name} channels {raw.ch_names}")
                idx = chs.index(CHANNEL.upper())
                return raw.get_data(picks=[idx]).squeeze()
            elif p.suffix.lower() == ".npy":
                arr = np.load(str(p))
                if arr.ndim == 3:
                    arr = arr.squeeze()
                # attempt channel by name: we can't; assume arr rows correspond to same channel ordering as your fif files
                # We'll try to find a matching .fif with same stem to get index
                # fallback: use first row if can't find mapping
                # Try to guess by reading any .fif counterpart in the same folder with same ID
                candidate_fif = list(DATA_DIR.glob(f"{subj_key}*.fif"))
                if candidate_fif:
                    raw = mne.io.read_raw_fif(str(candidate_fif[0]), preload=False, verbose='ERROR')
                    chs = [c.upper() for c in raw.ch_names]
                    if CHANNEL.upper() in chs:
                        idx = chs.index(CHANNEL.upper())
                        return arr[idx,:]
                # otherwise return first row
                return arr[0,:]
            else:
                raise ValueError("Unsupported format")
        eo_arr = load_arr(eo_path)
        ec_arr = load_arr(ec_path)

        # Trim extra samples at end (remove 1 extra sample)
        def trim_samples(arr):
            n = arr.shape[0]
            # If arr length is 75001 -> trim to 75000
            if n >= 75001:
                return arr[:75000]
            # If arr length is 73751 -> trim to 73750
            if n >= 73751 and n < 75001:
                return arr[:73750]
            # If exact 75000 or 73750, keep as is
            if n in (75000, 73750):
                return arr
            # If slightly longer, floor to nearest multiple of 5 seconds? But spec says remove 1 extra sample
            # We'll trim to floor(n / 5) * 5 to be safe, but prefer exact
            # fallback: if length > WIN_SAMPLES, trim to closest expected length (75000) if n>74000
            if n > 74000:
                return arr[:75000]
            if n > 73000:
                return arr[:73750]
            # else return as-is
            return arr
        eo_trim = trim_samples(eo_arr)
        ec_trim = trim_samples(ec_arr)

        # Make sure both trimmed lengths are identical target per-subject expectation
        # For normal subjects expected len=75000, for ID0/26 expected len=73750
        # Determine expected length dynamically: use min(eo_trim, ec_trim) and enforce equality by cropping
        L_eo, L_ec = len(eo_trim), len(ec_trim)
        # choose expected length as min of the two but prefer 75000 if either equals it
        target = None
        if L_eo >= 75000 and L_ec >= 75000:
            target = 75000
        elif L_eo >= 73750 and L_ec >= 73750:
            # choose 73750 if both smaller than 75000
            if L_eo < 75000 or L_ec < 75000:
                target = 73750
            else:
                target = 75000
        else:
            target = min(L_eo, L_ec)
        eo_trim = eo_trim[:target]
        ec_trim = ec_trim[:target]

        # concat EO + EC
        merged = np.concatenate([eo_trim, ec_trim], axis=0)  # shape (total_samples,)
        total_samples = len(merged)
        n_windows = total_samples // WIN_SAMPLES
        print(f"Processing ID{subject_id}: target len per file {target}, merged {total_samples} samples, windows {n_windows}")

        # For each window compute features
        rows = []
        for w in range(n_windows):
            start = w * WIN_SAMPLES
            stop = start + WIN_SAMPLES
            win = merged[start:stop]
            feats = compute_features_window(win, sf=SFREQ)
            # store subject ID and window index optionally
            feats["subj_id"] = int(subject_id)
            feats["window_idx"] = int(w)
            feats["label"] = label
            feats["pain_score"] = float(pain_score)
            rows.append(feats)

        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            # reorder columns to put subj_id, window_idx, label, pain_score first
            cols = [c for c in ("subj_id","window_idx","label","pain_score") if c in df.columns] + [c for c in df.columns if c not in ("subj_id","window_idx","label","pain_score")]
            df = df[cols]
            out_fname = OUT_DIR / f"ID{subject_id}_feature.csv"
            df.to_csv(out_fname, index=False)
            print(f"Saved features for ID{subject_id} ({n_windows} rows) -> {out_fname}")
        else:
            print(f"No windows for ID{subject_id} (merged length {total_samples})")

    except Exception as e:
        print(f"Error processing {subj_key}: {e}")

print("All done.")
