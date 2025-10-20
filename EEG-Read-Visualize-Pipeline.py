# Pain-Dataset EEG processing & visualization pipeline
# File: EEG-Read-Visualize-Pipeline.py
# Purpose: Read .gdf files, extract metadata, perform standard visualizations and preprocessing steps.
# Notes:
# - This script is written like a Jupyter notebook (use with VSCode/Spyder or convert to .ipynb)
# - Requires: mne, numpy, pandas, matplotlib, seaborn, scipy
# - Recommended: run inside a conda env: conda create -n eeg python=3.10 mne numpy pandas matplotlib seaborn scipy jupyter

# %%
"""SETUP & INSTALL (run once in your environment)
If you don't have required packages, uncomment and run the install cell.
"""
# !pip install mne pandas matplotlib seaborn numpy scipy

# %%
# Imports
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Set plotting defaults
sns.set_theme(style='whitegrid')

# %%
# USER CONFIG - change if needed
DATA_DIR = Path('Pain-Dataset')  # folder containing ID*.gdf files
OUTPUT_DIR = Path('pain_dataset_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# expected missing IDs (from your description)
MISSING_IDS = {12, 17, 28, 29, 32, 34, 36, 42}

# channel names hint: dataset uses 24 electrodes (2 mastoids). We'll rely on the file's channel names.

# frequency bands (Hz) for bandpower
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


# %%
# Utility: find all GDF files
gdf_files = sorted(DATA_DIR.glob('ID*.gdf'))
print(f'Found {len(gdf_files)} .gdf files in {DATA_DIR.absolute()}')

# Quick check: list missing IDs from 0..43
all_ids = set()
for f in gdf_files:
    name = f.stem  # e.g., 'ID0'
    try:
        num = int(''.join(filter(str.isdigit, name)))
        all_ids.add(num)
    except Exception:
        pass
print('IDs present:', sorted(all_ids))

# %%
# Function: read single GDF and extract basic metadata


def read_gdf_metadata(filepath, preload=False):
    """Read .gdf using mne and return raw object and metadata dict.
    preload=False will avoid loading full data into memory.
    """
    try:
        raw = mne.io.read_raw_gdf(
            str(filepath), preload=preload, verbose='ERROR')
    except Exception as e:
        print(f'Error reading {filepath}: {e}')
        return None, {'error': str(e)}

    info = raw.info
    sfreq = info['sfreq']
    ch_names = info['ch_names']
    n_channels = len(ch_names)
    n_samples = raw.n_times
    duration_sec = n_samples / sfreq

    metadata = {
        'file': str(filepath),
        'sfreq': sfreq,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'duration_sec': duration_sec,
        'ch_names': ch_names,
        'annotations': raw.annotations.description if raw.annotations is not None else None,
    }
    return raw, metadata


# %%
# Read metadata for all files (no preload) to build overview table
records = []
raw_cache = {}  # store small Raw objects if needed later with preload=True
for f in gdf_files:
    raw, meta = read_gdf_metadata(f, preload=False)
    if meta is None:
        continue
    records.append(meta)

meta_df = pd.DataFrame(records)
meta_df.to_csv(OUTPUT_DIR / 'files_metadata.csv', index=False)
meta_df

# %%
# Quick summary plots: distribution of durations and sampling rates
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
meta_df['duration_sec'].astype(float).hist(bins=10)
plt.title('Recording duration (sec)')
plt.subplot(1, 2, 2)
meta_df['sfreq'].astype(float).unique()
plt.text(0.1, 0.2, f"Sampling rates found: {meta_df['sfreq'].unique()}")
plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_and_sfreq.png')
plt.show()

# %%
# Helper: pick a subject to inspect in-depth (first available)
if len(gdf_files) == 0:
    raise FileNotFoundError(
        'No GDF files found. Make sure DATA_DIR is correct.')

example_file = gdf_files[0]
print('Example file chosen for demonstrations:', example_file)

# %%
# Load example file with preload (we will crop first 5 seconds as recommended)
raw_example, meta_example = read_gdf_metadata(example_file, preload=True)
print(meta_example)

# Crop beginning extra 5 seconds
raw_example.crop(tmin=5.0, tmax=None)
print(
    f'After cropping: duration = {raw_example.n_times/raw_example.info["sfreq"]:.2f} sec')


# %%
# Basic signal plot (first 10 seconds)
start = 0
stop = min(10, raw_example.times[-1])
raw_example.plot(start=start, duration=stop-start, n_channels=24, show=True)
# The interactive plot will open in a supported environment. For static PNG, use the next cell.

# %%
# Save static plot of first channel set (matplotlib)
fig = raw_example.plot(show=False)
fig.savefig(OUTPUT_DIR / f'{example_file.stem}_raw_plot.png')
plt.close(fig)


# %%

# Power spectral density (Welch) for all channels
psd = raw_example.compute_psd(
    method='welch', fmin=1, fmax=50, n_fft=2048, verbose='ERROR')
psds, freqs = psd.get_data(return_freqs=True)
psds_db = 10 * np.log10(psds)

plt.figure(figsize=(8, 5))
plt.semilogy(freqs, psds.mean(axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
plt.title(f'Average PSD - {example_file.stem}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{example_file.stem}_psd.png')
plt.show()


# %%
# Spectrogram for a single representative channel (e.g., channel 0)
ch_idx = 0
data, times = raw_example[ch_idx, :]
f, t, Sxx = signal.spectrogram(
    data.squeeze(), fs=raw_example.info['sfreq'], nperseg=512)
plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'Spectrogram - {raw_example.ch_names[ch_idx]}')
plt.ylim(1, 50)
plt.colorbar(label='PSD (dB)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{example_file.stem}_spectrogram_ch{ch_idx}.png')
plt.show()

# %%

# Band power calculation helper


def bandpower(data, sf, band, method='welch'):
    """Compute average band power (dB) for data array (n_samples,)"""
    fmin, fmax = band
    f, Pxx = signal.welch(data, fs=sf, nperseg=1024)
    # integrate PSD in band
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    bp = np.trapz(Pxx[idx_band], f[idx_band])
    return 10*np.log10(bp)


# compute band powers for all channels in example
sf = raw_example.info['sfreq']
bandpowers = {b: [] for b in BANDS}
for ch in range(raw_example.info['nchan']):
    chdata = raw_example.get_data(picks=[ch]).squeeze()
    for b, rng in BANDS.items():
        bandpowers[b].append(bandpower(chdata, sf, rng))

bp_df = pd.DataFrame(bandpowers, index=raw_example.ch_names)
bp_df.to_csv(OUTPUT_DIR / f'{example_file.stem}_bandpowers.csv')
print(bp_df.head())


# %%
# Topomap of band power (requires montage / channel positions). We'll try to set a standard 10-20 montage.
try:
    # try to set standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_tmp = raw_example.copy()
    raw_tmp.set_montage(montage, match_case=False, on_missing='warn')

    # compute evoked-like object for band (average across time)
    for band_name, rng in BANDS.items():
        powers = np.array([bandpowers[band_name]])[0]
        # create an info object and Evoked-like array for plotting topomap
        info = raw_tmp.info
        # create a fake evoked object
        evoked = mne.EvokedArray(powers.reshape(-1, 1), info, tmin=0.)
        fig = evoked.plot_topomap(times=0., show=False)
        fig.savefig(
            OUTPUT_DIR / f'{example_file.stem}_topomap_{band_name}.png')
        plt.close(fig)
    print('Topomaps saved (if channel positions matched).')
except Exception as e:
    print('Topomap generation error (likely missing montage info):', e)

# %%


# Define output directory
OUTPUT_DIR = Path("pain_dataset_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Load your example raw file ----
# Example: raw_example = mne.io.read_raw_fif("your_file.fif", preload=True)
# (Assuming 'raw_example' is already loaded)

# Print all channel names to verify
print("Channel names before rename:", raw_example.ch_names)

# --- Step 1: Fix montage naming mismatches (FP1, FP2 → Fp1, Fp2) ---
rename_dict = {}
for ch in raw_example.ch_names:
    if ch.upper() == 'FP1':
        rename_dict[ch] = 'Fp1'
    elif ch.upper() == 'FP2':
        rename_dict[ch] = 'Fp2'

if rename_dict:
    print("Renaming channels:", rename_dict)
    raw_example.rename_channels(rename_dict)

# --- Step 2: Set standard 10-20 montage safely ---
try:
    raw_example.set_montage('standard_1020', on_missing='ignore')
    print(" Montage set successfully with 'standard_1020'")
except Exception as e:
    print(" Montage assignment issue:", e)

# --- Step 3: Run ICA for artifact detection ---
ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
ica.fit(raw_example)

# --- Step 4: Plot ICA components (interactive + saved image) ---
ica.plot_components()  # Opens interactive window

# Save components grid image
fig = ica.plot_components(show=False)
example_file = Path("raw_example_data")  # Replace with your filename stem
fig.savefig(OUTPUT_DIR / f'{example_file.stem}_ica_components.png')
plt.close(fig)

print("✅ ICA components plotted and saved successfully.")


# %%
# OPTIONAL: Automatically find EOG-like components by correlation to frontal channels (if present)
frontal_candidates = []
for ch in raw_example.ch_names:
    if 'Fp' in ch or 'AF' in ch or 'Fz' in ch:
        frontal_candidates.append(ch)

if frontal_candidates:
    print('Frontal channels found:', frontal_candidates)
else:
    print('No clear frontal channels found; auto EOG detection may fail.')

# %%
# Batch processing function: compute summary metrics for each file and save CSV


def summarize_all_files(gdf_files, out_csv=OUTPUT_DIR / 'dataset_summary.csv'):
    rows = []
    for f in gdf_files:
        raw, meta = read_gdf_metadata(f, preload=True)
        if meta.get('error'):
            rows.append({**{'file': str(f), 'error': meta['error']}})
            continue
        # crop first 5 seconds
        raw.crop(tmin=5.0, tmax=None)
        sf = raw.info['sfreq']
        duration = raw.n_times / sf
        # compute average RMS per channel
        data = raw.get_data()
        rms = np.sqrt(np.mean(data**2, axis=1))
        avg_rms = np.mean(rms)
        # bandpower per channel (alpha mean)
        alpha_vals = []
        for ch in range(raw.info['nchan']):
            chdata = raw.get_data(picks=[ch]).squeeze()
            alpha_vals.append(bandpower(chdata, sf, BANDS['alpha']))
        rows.append({
            'file': str(f),
            'sfreq': sf,
            'n_channels': raw.info['nchan'],
            'duration_sec': duration,
            'avg_rms': avg_rms,
            'alpha_mean_db': np.mean(alpha_vals)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


summary_df = summarize_all_files(gdf_files)
summary_df.head()

# %%
# Save workspace objects (example): pickled bandpowers & meta
bp_df.to_pickle(OUTPUT_DIR / f'{example_file.stem}_bandpowers.pkl')
meta_df.to_pickle(OUTPUT_DIR / 'files_metadata.pkl')

# %%
"""
Advanced next steps (not implemented here but suggested):
- Automated artifact removal using EOG/ECG channel detection or template matching
- Epoching the 5-min segments into fixed-length windows (e.g., 2s) and computing features per epoch
- Feature extraction: Hjorth parameters, spectral entropy, fractal dimension, Hjorth, zero-crossing rate, wavelet features
- Connectivity measures: coherence, phase-locking value, imaginary coherence
- Machine learning pipeline: per-subject feature aggregation, normalization, cross-validation
- Visualization dashboard: Plotly/Dash or Voila for interactive exploration

If you want I can extend the notebook to include epoching + feature extraction (Hjorth, spectral entropy, band ratios), or implement an automated artifact detection & removal pipeline.
"""
