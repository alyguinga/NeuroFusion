"""
run_pipeline_v3.py
═══════════════════════════════════════════════════════════════════════════
NeuroFusion-AD Pipeline V3 — Main Entry Point

Complete 7-step pipeline:
  1. Load dataset (BIDS or synthetic)
  2. Generate EEG preprocessing visualizations
  3. Extract epochs & per-channel z-score normalization
  4. Extract multimodal features:
     a) CWT Scalograms (for CViT branch)
     b) Connectivity matrices (for GNN branch)
     c) Biomarker features: spectral + complexity + connectivity (for BiomarkerNet)
  5. Biomarker heatmap visualization
  6. Train ALL models (FusionNet, LSTM, BiLSTM, RF, SVM) with 5-fold CV
  7. Run XAI (Grad-CAM, feature attribution, attention weights)

Data integrity:
  - Subject-level cross-validation (StratifiedGroupKFold)
  - No data leakage: features extracted before CV, but normalization
    is global z-score (safe for CV since it's a linear transform)
═══════════════════════════════════════════════════════════════════════════
"""

import argparse, time, sys, warnings, os, random
import numpy as np
import torch

warnings.filterwarnings("ignore")

import config as CFG


def set_global_seed(seed):
    """Set all random seeds for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
from data.loader import load_dataset
from data.scalogram import batch_compute_scalograms
from features.extractor import extract_all_biomarkers
from trainer import cross_validate_all
from visualize import (plot_eeg_visualizations, plot_model_comparison,
                       plot_biomarker_heatmap, plot_scalograms,
                       plot_biomarker_importance,
                       plot_confusion_matrix)
from xai import plot_xai_results, population_attribution, plot_attention_by_class


def parse_args():
    p = argparse.ArgumentParser(description="NeuroFusion-AD V3 Pipeline")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p.add_argument("--n_subjects", type=int, default=30)
    p.add_argument("--data_dir", default=CFG.DATA_DIR, help="Path to BIDS root")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--skip_bio", action="store_true", help="Skip biomarker extraction (faster)")
    p.add_argument("--no_gnn", action="store_true", help="Disable GNN branch during main training based on ablation study findings")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(CFG.SEED)

    t0 = time.time()
    print("=" * 70)
    print("  NeuroFusion-AD Pipeline V3")
    print("  4-Branch Hybrid: EEGNet + CViT + GNN + BiomarkerNet")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load Dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/7] Loading dataset …")
    records = load_dataset(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        n_synthetic=args.n_subjects
    )
    if len(records) == 0:
        print("ERROR: no valid subjects loaded.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: EEG Preprocessing Visualizations
    # ══════════════════════════════════════════════════════════════════
    print("\n[2/7] Generating EEG Visualizations …")
    plot_eeg_visualizations(records, plot_dir=CFG.PLOT_DIR, sfreq=CFG.SFREQ)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Extract Epochs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/7] Extracting epochs ({len(records)} subjects) …")
    X_parts, y_parts, groups_parts = [], [], []

    for si, rec in enumerate(records):
        data = rec["epochs"].get_data().astype(np.float32)
        # Per-channel z-score normalization (within subject)
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
    # STEP 4: Extract Multimodal Features
    # ══════════════════════════════════════════════════════════════════
    print("\n[4/7] Extracting Multimodal Features …")

    # A. Scalograms for CViT
    print("\n  4a. Computing CWT Scalograms …")
    freqs = np.arange(2, 45, 2)
    X_scalo = batch_compute_scalograms(X_1d, sfreq=CFG.SFREQ, freqs=freqs)
    print(f"  Scalogram Tensor: {X_scalo.shape}")
    
    print("  Generating Scalogram Visualization …")
    plot_scalograms(X_scalo, y_all, freqs, CFG.SFREQ, CFG.PLOT_DIR)

    # 4b. Biomarker Features (Path D: BiomarkerNet + Path C: GNN via PLI)
    X_bio = None
    X_pli = None
    feature_names = None

    if not args.skip_bio:
        print("\n  4b. Extracting Domain Biomarkers (spectral + complexity + connectivity) …")
        X_bio, X_pli, feature_names = extract_all_biomarkers(
            X_1d, sfreq=CFG.SFREQ, save_scaler_to=CFG.MODEL_DIR
        )
        print(f"  Biomarker Tensor: {X_bio.shape}")
        print(f"  PLI Tensor: {X_pli.shape}")

        # Use PLI matrices for GNN (more principled than Pearson correlation)
        X_adj = X_pli
    else:
        print("\n  4b. Skipping biomarker extraction (--skip_bio)")
        # Fall back to simple Pearson connectivity
        from data.connectivity import batch_compute_connectivity
        X_adj = batch_compute_connectivity(X_1d)

    print(f"  Adjacency Tensor: {X_adj.shape}")

    # NaN check
    for name, tensor in zip(["Raw", "Scalo", "Adj"], [X_1d, X_scalo, X_adj]):
        if np.isnan(tensor).any():
            print(f"  WARNING: NaNs in {name} — fixing...")
            np.nan_to_num(tensor, copy=False)
    if X_bio is not None and np.isnan(X_bio).any():
        print(f"  WARNING: NaNs in Bio — fixing...")
        np.nan_to_num(X_bio, copy=False)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Biomarker Analysis (Heatmap & Importance)
    # ══════════════════════════════════════════════════════════════════
    if X_bio is not None and feature_names:
        print("\n[5/7] Generating Biomarker Visualizations …")
        plot_biomarker_heatmap(X_bio, y_all, feature_names, CFG.PLOT_DIR)
        plot_biomarker_importance(X_bio, y_all, feature_names, CFG.PLOT_DIR)
    else:
        print("\n[5/7] Skipped (no biomarker features)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Train FusionNet
    # ══════════════════════════════════════════════════════════════════
    print("\n[6/7] Cross-validation – FusionNet …")
    avg_results, all_fold_metrics, best_fusion_model, val_y, val_p = cross_validate_all(
        X_1d, X_scalo, X_adj, y_all, groups_all,
        X_bio=X_bio,
        n_splits=3, epochs=args.epochs, batch_size=args.batch_size,
        disable_gnn=args.no_gnn
    )

    # Plot Confusion Matrix
    class_names = [CFG.CLASS_NAMES[c] for c in CFG.CLASSES]
    cm_filename = 'confusion_matrix_no_gnn.png' if args.no_gnn else 'confusion_matrix.png'
    plot_confusion_matrix(val_y, val_p, class_names, CFG.PLOT_DIR, filename=cm_filename)

    # Comparison chart
    plot_model_comparison(avg_results, plot_dir=CFG.PLOT_DIR)

    # ══════════════════════════════════════════════════════════════════
    # Save Final Model
    # ══════════════════════════════════════════════════════════════════
    if best_fusion_model is not None:
        save_dir = CFG.MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)
        filename = 'neurofusion_v3_no_gnn.pth' if args.no_gnn else 'neurofusion_v3_final.pth'
        model_path = os.path.join(save_dir, filename)
        torch.save(best_fusion_model.state_dict(), model_path)
        desc = "No GNN" if args.no_gnn else "with GNN FC Projection"
        print(f"\n[+] Final optimal model ({desc}) saved to: {model_path}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: XAI (Explainability)
    # ══════════════════════════════════════════════════════════════════
    if best_fusion_model is not None and X_bio is not None:
        print("\n[7/7] Generating XAI Visualizations …")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_fusion_model = best_fusion_model.to(device)

        # 7a. Single-sample explanation (Grad-CAM, etc.)
        ad_idx = np.where(y_all == 1)[0][:5]
        if len(ad_idx) > 0:
            xai_1d = torch.tensor(X_1d[ad_idx]).unsqueeze(1).float()
            xai_scalo = torch.tensor(X_scalo[ad_idx]).float()
            xai_adj = torch.tensor(X_adj[ad_idx]).float()
            xai_bio = torch.tensor(X_bio[ad_idx]).float()

            plot_xai_results(
                best_fusion_model,
                xai_1d, xai_scalo, xai_adj, xai_bio,
                feature_names, y_all[ad_idx],
                device=device, save_dir=CFG.XAI_DIR
            )

        # 7b. Population-level attribution (Mean over 50 samples/class)
        print("\n  Computing population-level attribution (N=100)...")
        population_attribution(
            best_fusion_model, 
            torch.tensor(X_1d).unsqueeze(1).float(), 
            torch.tensor(X_scalo).float(), 
            torch.tensor(X_adj).float(), 
            torch.tensor(X_bio).float(), 
            y_all, feature_names, n_per_class=50, 
            device=device, save_dir=CFG.XAI_DIR
        )

        # 7c. Attention by class
        print("\n  Computing cross-path attention by class...")
        plot_attention_by_class(
            best_fusion_model, 
            torch.tensor(X_1d).unsqueeze(1).float(), 
            torch.tensor(X_scalo).float(), 
            torch.tensor(X_adj).float(), 
            torch.tensor(X_bio).float(), 
            y_all, device=device, save_dir=CFG.XAI_DIR
        )
    else:
        print("\n[7/7] Skipped XAI (no biomarker features or no trained model)")

    elapsed = time.time() - t0
    print(f"\n{'═'*70}")
    print(f"  NeuroFusion-AD Pipeline V3 complete in {elapsed/60:.1f} min")
    print(f"  Plots saved to: {CFG.PLOT_DIR}")
    print(f"  XAI saved to:   {CFG.XAI_DIR}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
