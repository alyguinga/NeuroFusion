"""
data/loader.py
─────────────────────────────────────────────────────────────────────────────
Load OpenNeuro ds004505, keep only AD and Normal (CN/HC) subjects.
Returns a list of MNE Epochs objects, one per subject, with metadata.

Usage
─────
    from data.loader import load_dataset
    subjects = load_dataset(data_dir, use_synthetic=True)
    # subjects[i] = {"epochs": mne.Epochs, "label": 0|1,
    #                "subject_id": str, "mmse": float|None}

If the BIDS data is not yet downloaded the helper `download_dataset` can
fetch it via openneuro-py (requires an account token for large datasets).
For development / CI a synthetic mode generates realistic MNE Epochs.
─────────────────────────────────────────────────────────────────────────────
"""

import os, glob, warnings
import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib
matplotlib.use('Agg')  # Headless backend for Kaggle/servers
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

_PLOTTED_FIRST_SUBJECT = False

# ── try importing optional dependencies ──────────────────────────────────
try:
    import openneuro
    HAS_OPENNEURO = True
except ImportError:
    HAS_OPENNEURO = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as CFG


# ─────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────
def load_dataset(data_dir: str = CFG.DATA_DIR,
                 use_synthetic: bool = False,
                 n_synthetic: int = 30) -> list[dict]:
    """
    Parameters
    ----------
    data_dir       : path to the BIDS root (ds004505/)
    use_synthetic  : generate synthetic data instead of reading from disk
    n_synthetic    : number of synthetic subjects (split 50/50 AD/Normal)

    Returns
    -------
    list of dicts with keys:
        epochs     – mne.Epochs (already preprocessed)
        label      – int  0=Normal, 1=AD
        subject_id – str  e.g. "sub-001"
        mmse       – float or None
    """
    if use_synthetic:
        print(f"[loader] Generating {n_synthetic} synthetic subjects …")
        return _make_synthetic(n_synthetic)

    bids_root = os.path.join(data_dir, CFG.OPENNEURO_DATASET)
    if not os.path.isdir(bids_root):
        # try versioned folder name (e.g. ds004504-1.0.8)
        matches = glob.glob(os.path.join(data_dir, CFG.OPENNEURO_DATASET + "-*"))
        if matches:
            bids_root = matches[0]
        # or data_dir itself is the dataset root
        elif os.path.isfile(os.path.join(data_dir, "participants.tsv")):
            bids_root = data_dir
        else:
            raise FileNotFoundError(
                f"BIDS root not found at {bids_root}.\n"
                "Run `download_dataset()` first, or set use_synthetic=True."
            )

    participants = _read_participants_tsv(bids_root)
    records = []
    for row in participants:
        label = CFG.LABEL_MAP.get(row.get("group", ""), None)
        if label is None:
            continue                            # skip MCI / unknown
        raw = _load_raw_bids(bids_root, row["participant_id"])
        if raw is None:
            continue
        epochs = _preprocess(raw)
        if epochs is None or len(epochs) == 0:
            continue
        records.append({
            "epochs":     epochs,
            "label":      label,
            "subject_id": row["participant_id"],
            "mmse":       _parse_float(row.get("mmse", None)),
        })
    print(f"[loader] Loaded {len(records)} subjects  "
          f"(AD={sum(r['label']==1 for r in records)}, "
          f"Normal={sum(r['label']==0 for r in records)})")
    return records


# ─────────────────────────────────────────────────────────────────────────
# BIDS helpers
# ─────────────────────────────────────────────────────────────────────────
def _read_participants_tsv(bids_root: str) -> list[dict]:
    tsv = os.path.join(bids_root, "participants.tsv")
    rows = []
    with open(tsv) as f:
        headers = [h.lower() for h in f.readline().strip().split("\t")]
        for line in f:
            vals = line.strip().split("\t")
            rows.append(dict(zip(headers, vals)))
    return rows


def _load_raw_bids(bids_root: str, subject_id: str) -> mne.io.Raw | None:
    """Try common EEG file patterns inside a BIDS subject folder."""
    patterns = [
        f"{bids_root}/{subject_id}/eeg/*.edf",
        f"{bids_root}/{subject_id}/eeg/*.set",
        f"{bids_root}/{subject_id}/eeg/*.bdf",
        f"{bids_root}/{subject_id}/eeg/*.vhdr",
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            fpath = files[0]
            ext   = os.path.splitext(fpath)[1].lower()
            try:
                if ext == ".set":
                    raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose=False)
                elif ext == ".edf":
                    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                elif ext == ".bdf":
                    raw = mne.io.read_raw_bdf(fpath, preload=True, verbose=False)
                elif ext == ".vhdr":
                    raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose=False)
                else:
                    continue
                return raw
            except Exception as e:
                warnings.warn(f"[loader] Could not read {fpath}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (MNE)
# ─────────────────────────────────────────────────────────────────────────
def _preprocess(raw: mne.io.Raw) -> mne.Epochs | None:
    """
    Full preprocessing pipeline:
      1. Resample to CFG.SFREQ
      2. Bandpass 1-40 Hz + notch 50 Hz
      3. Pick EEG channels only
      4. Set 10-20 standard montage
      5. Average re-reference
      6. ICA artifact removal (eye / muscle)
      7. Pick 12 AD-relevant channels
      8. Epoch into 4 s windows, 50% overlap
      9. Reject epochs with |amplitude| > 100 µV
    """
    global _PLOTTED_FIRST_SUBJECT
    plot_this = not _PLOTTED_FIRST_SUBJECT
    
    try:
        if plot_this:
            _PLOTTED_FIRST_SUBJECT = True
            print("[loader] Saving preprocessing visualizations for representative subject...")
            # Save Raw Signal
            fig = raw.plot(duration=5, n_channels=min(16, len(raw.ch_names)), title="Step 1: Raw Signal", show=False)
            fig.savefig(os.path.join(CFG.PLOT_DIR, "preprocess_raw.png"))
            plt.close(fig)
            
        # 1. Resample
        if raw.info["sfreq"] != CFG.SFREQ:
            raw.resample(CFG.SFREQ, npad="auto")

        # 2. Filter
        raw.filter(CFG.L_FREQ, CFG.H_FREQ, method="fir",
                   fir_window="hamming", verbose=False)
        raw.notch_filter(CFG.NOTCH_FREQ, verbose=False)
        
        if plot_this:
            fig = raw.plot(duration=5, n_channels=min(16, len(raw.ch_names)), title="Step 2: Filtered & Notch", show=False)
            fig.savefig(os.path.join(CFG.PLOT_DIR, "preprocess_filtered.png"))
            plt.close(fig)

        # 3. Pick EEG
        raw.pick_types(eeg=True, stim=False, eog=False)

        # 4. Montage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage("standard_1020", on_missing="ignore")

        # 5. Reference
        raw.set_eeg_reference(CFG.REFERENCE, verbose=False)

        # 6. ICA artifact removal
        n_comp = min(CFG.ICA_N_COMPONENTS, len(raw.ch_names) - 1)
        ica = ICA(n_components=n_comp, method="fastica",
                  random_state=CFG.SEED, max_iter=400)
        ica.fit(raw, verbose=False)
        
        if plot_this:
            # Safely plot ICA components
            try:
                fig = ica.plot_components(show=False)
                # Sometimes plot_components returns a list of figures
                if isinstance(fig, list):
                    fig[0].savefig(os.path.join(CFG.PLOT_DIR, "preprocess_ica.png"))
                    for f in fig: plt.close(f)
                else:
                    fig.savefig(os.path.join(CFG.PLOT_DIR, "preprocess_ica.png"))
                    plt.close(fig)
            except Exception as e:
                warnings.warn(f"Could not plot ICA components: {e}")

        # auto-detect eye / muscle components
        eog_idx, _  = ica.find_bads_eog(raw, verbose=False) if "EOG" in str(raw.ch_names) else ([], {})
        ica.exclude  = eog_idx[:3]   # exclude at most 3
        ica.apply(raw, verbose=False)

        # 7. Pick 16 key channels (keep whichever are present)
        present = [ch for ch in CFG.AD_CHANNELS if ch in raw.ch_names]
        if len(present) < 6:
            warnings.warn(f"[preprocess] Only {len(present)} AD channels found; skipping subject.")
            return None
        raw.pick_channels(present)
        
        if plot_this:
            fig = raw.plot(duration=5, n_channels=len(present), title="Step 4: Cleaned AD Channels", show=False)
            fig.savefig(os.path.join(CFG.PLOT_DIR, "preprocess_clean.png"))
            plt.close(fig)

        # 8. Epoch (fixed-length, no events needed for resting-state)
        step   = CFG.EPOCH_DURATION * (1.0 - CFG.EPOCH_OVERLAP)
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=CFG.EPOCH_DURATION,
            overlap=CFG.EPOCH_DURATION * CFG.EPOCH_OVERLAP,
            verbose=False,
        )

        # 9. Amplitude rejection
        reject = {"eeg": CFG.AMP_THRESHOLD}
        epochs.drop_bad(reject=reject, verbose=False)

        return epochs

    except Exception as e:
        warnings.warn(f"[preprocess] Failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data (for development without downloading the dataset)
# ─────────────────────────────────────────────────────────────────────────
def _make_synthetic(n_subjects: int = 30) -> list[dict]:
    """
    Generate synthetic MNE Epochs that mimic resting-state EEG statistics
    for AD (label=1) and Normal (label=0) subjects.

    AD signal: boosted delta/theta, attenuated alpha, reduced complexity.
    Normal   : balanced spectral profile.
    """
    rng    = np.random.default_rng(CFG.SEED)
    records = []
    n_ch   = len(CFG.AD_CHANNELS)
    n_times = int(CFG.EPOCH_DURATION * CFG.SFREQ)   # 1024 samples
    n_epochs_per_sub = 40

    info = mne.create_info(
        ch_names=CFG.AD_CHANNELS,
        sfreq=CFG.SFREQ,
        ch_types=["eeg"] * n_ch,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage("standard_1020", on_missing="ignore")

    t = np.linspace(0, CFG.EPOCH_DURATION, n_times)

    for i in range(n_subjects):
        label = 1 if i < n_subjects // 2 else 0
        mmse  = float(rng.integers(10, 22)) if label == 1 else float(rng.integers(24, 30))

        data = np.zeros((n_epochs_per_sub, n_ch, n_times))
        for ep in range(n_epochs_per_sub):
            for ci in range(n_ch):
                sig = np.zeros(n_times)
                for band, (flo, fhi) in CFG.BANDS.items():
                    fc   = (flo + fhi) / 2
                    amp  = rng.uniform(0.5, 1.5)
                    # AD: amplify slow bands, suppress fast
                    if label == 1:
                        if band in ("delta", "theta"):
                            amp *= 2.2
                        elif band in ("alpha", "beta"):
                            amp *= 0.5
                    phase = rng.uniform(0, 2 * np.pi)
                    sig  += amp * np.sin(2 * np.pi * fc * t + phase)
                # pink-ish noise
                sig += rng.normal(0, 0.3, n_times)
                data[ep, ci] = sig * 1e-6   # convert to Volts

        epochs = mne.EpochsArray(data, info, verbose=False)
        records.append({
            "epochs":     epochs,
            "label":      label,
            "subject_id": f"sub-syn{i+1:03d}",
            "mmse":       mmse,
        })

    print(f"[loader] Synthetic: {n_subjects} subjects  "
          f"(AD={n_subjects//2}, Normal={n_subjects-n_subjects//2})")
    return records


# ─────────────────────────────────────────────────────────────────────────
def _parse_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────
def download_dataset(token: str | None = None):
    """Download ds004505 from OpenNeuro (requires openneuro-py)."""
    if not HAS_OPENNEURO:
        raise ImportError("pip install openneuro-py")
    dest = os.path.join(CFG.DATA_DIR, CFG.OPENNEURO_DATASET)
    openneuro.download(
        dataset=CFG.OPENNEURO_DATASET,
        target_dir=dest,
        token=token,
    )
    print(f"[loader] Dataset saved to {dest}")
