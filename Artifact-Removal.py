import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================
INPUT_DIR = Path("Segments")  # main folder
EO_DIR = INPUT_DIR / "EO"
EC_DIR = INPUT_DIR / "EC"

OUTPUT_DIR = Path("DATA-after-Artifact")
EO_OUT = OUTPUT_DIR / "EO"
EC_OUT = OUTPUT_DIR / "EC"
EO_OUT.mkdir(parents=True, exist_ok=True)
EC_OUT.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 15
RANDOM_STATE = 97
LOW_FREQ = 1.0
HIGH_FREQ = None  # low-pass not used (keep up to Nyquist)

# ==========================================================
# HELPER: PROCESS ONE FILE
# ==========================================================


def process_and_clean(file_path: Path, save_dir: Path, cond: str):
    print(f"\n⚙️ Processing {file_path.name} ({cond})")

    # Load raw EEG data
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")

    # Rename channels if needed
    rename_dict = {ch: ch.capitalize()
                   for ch in raw.ch_names if ch.upper() in ["FP1", "FP2"]}
    if rename_dict:
        raw.rename_channels(rename_dict)

    # Set montage
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception as e:
        print(f"⚠️ Montage issue: {e}")

    # Filter and average reference
    raw.pick_types(eeg=True)
    raw.set_eeg_reference("average", projection=True)
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)

    # ICA
    ica = mne.preprocessing.ICA(
        n_components=N_COMPONENTS, random_state=RANDOM_STATE, max_iter="auto")
    ica.fit(raw)

    # Detect artifacts
    eog_inds, eog_scores = ica.find_bads_eog(raw)
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw)
    bad_inds, bad_scores = ica.find_bads_ch(raw)

    exclude_inds = list(set(eog_inds + ecg_inds + bad_inds))
    print(f"🧹 Artifact components detected: {exclude_inds}")

    # Mark excluded components
    ica.exclude = exclude_inds

    # Apply ICA
    raw_clean = ica.apply(raw.copy())

    # Save cleaned data
    fif_path = save_dir / f"{file_path.stem}_clean.fif"
    npy_path = save_dir / f"{file_path.stem}_clean.npy"

    raw_clean.save(fif_path, overwrite=True)
    np.save(npy_path, raw_clean.get_data())

    print(
        f"✅ Cleaned data saved:\n   FIF: {fif_path.name}\n   NPY: {npy_path.name}")

    # (Optional) Save excluded components plot
    fig = ica.plot_components(picks=exclude_inds, show=False)
    fig_path = save_dir / f"{file_path.stem}_excluded_components.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ==========================================================
# PROCESS EO & EC SEPARATELY
# ==========================================================
print("\n================ EO FILES =================")
for f in sorted(EO_DIR.glob("*.fif")):
    process_and_clean(f, EO_OUT, "EO")

print("\n================ EC FILES =================")
for f in sorted(EC_DIR.glob("*.fif")):
    process_and_clean(f, EC_OUT, "EC")

print("\n🎯 All EO and EC files cleaned and saved successfully.")
