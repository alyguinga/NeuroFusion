"""
run_tf_comparison.py
═══════════════════════════════════════════════════════════════════════════
STFT vs CWT Time-Frequency Representation Comparison

Standalone script that evaluates the impact of different time-frequency
analysis methods on FusionNet classification accuracy for AD detection.

Pipeline:
  1. Load dataset (same as run_pipeline_v3.py steps 1–3)
  2. Compute TFRs using both STFT and CWT
  3. Visualize TFR outputs side-by-side
  4. Train FusionNet independently with each TFR (same split, same HP)
  5. Generate comparison metrics and plots

Literature basis:
  - PMC/NIH 2024–2025: STFT achieved 98.8% in multi-channel fusion
    frameworks; CWT excels at multi-resolution non-stationary analysis.
  - Both consistently outperform HHT and WVD in deep-learning AD studies.

Usage:
  python run_tf_comparison.py --synthetic --epochs 40
  python run_tf_comparison.py --data_dir ./data --epochs 60
═══════════════════════════════════════════════════════════════════════════
"""

import argparse
import time
import sys
import os
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

import config as CFG
from data.loader import load_dataset
from data.tf_representations import batch_compute_tf, SUPPORTED_METHODS
from features.extractor import extract_all_biomarkers
from visualize import (
    plot_tf_comparison_samples,
    plot_tf_method_comparison,
    plot_tf_auc_comparison,
    plot_tf_radar_chart,
    plot_tf_difference_maps,
    plot_tf_band_energy,
    plot_training_curves
)


# ═════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE (reuses trainer_v3 internals)
# ═════════════════════════════════════════════════════════════════════════

def _train_with_tf_method(X_1d, X_scalo, X_adj, y, groups, X_bio,
                          epochs, batch_size, device, method_name):
    """
    Train FusionNet with a specific TFR method's output.
    Returns metrics dict and trained model.
    """
    from trainer import _train_fusion
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = next(iter(sgkf.split(X_1d, y, groups)))

    print(f"\n  [{method_name.upper()}] Training FusionNet...")
    print(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"    Train AD: {(y[train_idx]==1).sum()} | Normal: {(y[train_idx]==0).sum()}")

    metrics, model, _, _ = _train_fusion(
        X_1d, X_scalo, X_adj, X_bio,
        train_idx, val_idx, y, device,
        epochs=epochs, batch_size=batch_size,
        patience=25, fold_num=1
    )

    print(f"  [{method_name.upper()}] Results: "
          f"Acc={metrics['acc']:.4f} Sens={metrics['sens']:.4f} "
          f"Spec={metrics['spec']:.4f} AUC={metrics['auc']:.4f}")

    return metrics, model


def parse_args():
    p = argparse.ArgumentParser(
        description="STFT vs CWT Time-Frequency Comparison for AD Detection"
    )
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data")
    p.add_argument("--n_subjects", type=int, default=30)
    p.add_argument("--data_dir", default=CFG.DATA_DIR,
                   help="Path to BIDS root")
    p.add_argument("--epochs", type=int, default=40,
                   help="Training epochs per method")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--methods", nargs='+', default=['cwt', 'stft', 'cqt', 'wvd'],
                   choices=SUPPORTED_METHODS,
                   help="TFR methods to compare")
    p.add_argument("--skip_bio", action="store_true",
                   help="Skip biomarker extraction")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print("=" * 70)
    print("  Time-Frequency Method Comparison")
    print(f"  Methods: {[m.upper() for m in args.methods]}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load Dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/5] Loading dataset ...")
    records = load_dataset(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        n_synthetic=args.n_subjects
    )
    if len(records) == 0:
        print("ERROR: no valid subjects loaded.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Extract Epochs (same as pipeline step 3)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[2/5] Extracting epochs ({len(records)} subjects) ...")
    X_parts, y_parts, groups_parts = [], [], []

    for si, rec in enumerate(records):
        data = rec["epochs"].get_data().astype(np.float32)
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True) + 1e-8
        data = (data - mean) / std

        n_ep = len(data)
        X_parts.append(data)
        y_parts.extend([rec["label"]] * n_ep)
        groups_parts.extend([si] * n_ep)
        del rec["epochs"]

    X_1d = np.concatenate(X_parts, axis=0).astype(np.float32)
    del X_parts
    y_all = np.array(y_parts, dtype=np.int64)
    groups_all = np.array(groups_parts, dtype=np.int64)

    print(f"  Raw 1D Tensor : {X_1d.shape}")
    print(f"  AD epochs     : {(y_all==1).sum()}")
    print(f"  Normal epochs : {(y_all==0).sum()}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Compute TFRs + Biomarkers
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/5] Computing Time-Frequency Representations ...")
    freqs = np.arange(2, 45, 2)

    tf_results = {}
    for method in args.methods:
        print(f"\n  --- {method.upper()} ---")
        X_tf = batch_compute_tf(X_1d, sfreq=CFG.SFREQ, method=method,
                                freqs=freqs, decim=4)
        tf_results[method] = X_tf

        # NaN check
        if np.isnan(X_tf).any():
            print(f"  WARNING: NaNs in {method.upper()} output — fixing")
            np.nan_to_num(X_tf, copy=False)

    # Biomarkers (shared across all methods)
    X_bio = None
    X_pli = None
    feature_names = None

    if not args.skip_bio:
        print("\n  Extracting biomarkers (shared across methods) ...")
        X_bio, X_pli, feature_names = extract_all_biomarkers(
            X_1d, sfreq=CFG.SFREQ
        )
        X_adj = X_pli
    else:
        print("\n  Skipping biomarkers — using Pearson connectivity")
        from data.connectivity import batch_compute_connectivity
        X_adj = batch_compute_connectivity(X_1d)

    # NaN cleanup
    for name, tensor in zip(["Raw", "Adj"], [X_1d, X_adj]):
        if np.isnan(tensor).any():
            print(f"  WARNING: NaNs in {name} — fixing")
            np.nan_to_num(tensor, copy=False)
    if X_bio is not None and np.isnan(X_bio).any():
        np.nan_to_num(X_bio, copy=False)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3b: Visualize TFR Comparison (side-by-side)
    # ══════════════════════════════════════════════════════════════════
    print("\n  Generating TFR comparison visualization ...")
    tf_compare_dir = os.path.join(CFG.PLOT_DIR, 'tf_comparison')
    os.makedirs(tf_compare_dir, exist_ok=True)

    plot_tf_comparison_samples(
        tf_results, y_all, freqs, CFG.SFREQ, tf_compare_dir
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Train FusionNet with Each Method
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[4/5] Training FusionNet with each TFR method ...")
    all_metrics = {}

    for method in args.methods:
        print(f"\n{'─' * 60}")
        print(f"  Training with: {method.upper()}")
        print(f"{'─' * 60}")

        # Reset random seeds for fair comparison
        np.random.seed(CFG.SEED)
        torch.manual_seed(CFG.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(CFG.SEED)

        metrics, model = _train_with_tf_method(
            X_1d, tf_results[method], X_adj, y_all, groups_all,
            X_bio, args.epochs, args.batch_size, device, method
        )
        all_metrics[method] = metrics

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Comparison Results & Visualizations
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Generating comparison results & visualizations ...")

    # 5a. Grouped bar chart (Acc/Sens/Spec/AUC)
    plot_tf_method_comparison(all_metrics, tf_compare_dir)

    # 5b. AUC lollipop chart with delta annotations
    plot_tf_auc_comparison(all_metrics, tf_compare_dir)

    # 5c. Radar chart (full performance profile)
    plot_tf_radar_chart(all_metrics, tf_compare_dir)

    # 5d. Difference maps (AD − Normal) for each method
    plot_tf_difference_maps(
        tf_results, y_all, freqs, CFG.SFREQ, tf_compare_dir
    )

    # 5e. Band energy distribution (AD vs Normal per band)
    plot_tf_band_energy(tf_results, y_all, freqs, tf_compare_dir)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"  TF Comparison complete in {elapsed / 60:.1f} min")
    print(f"  Results saved to: {tf_compare_dir}")
    print(f"{'═' * 70}")

    # Determine winner
    best_method = max(all_metrics, key=lambda m: all_metrics[m]['auc'])
    print(f"\n  Best method by AUC: {best_method.upper()} "
          f"(AUC={all_metrics[best_method]['auc']:.4f})")

    return all_metrics


if __name__ == "__main__":
    main()
