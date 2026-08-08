"""
v3/visualize.py
═══════════════════════════════════════════════════════════════════════════
Comprehensive visualization module for the NeuroFusion-AD pipeline.

Generates publication-quality figures at each pipeline stage:
  1. EEG Preprocessing — Raw signal, PSD, band-power topomaps
  2. Biomarker Features — Heatmap of AD vs Normal feature patterns
  3. Training Progress — Loss/accuracy curves, train/val gap
  4. Model Comparison — Bar chart of all models' Acc/Sens/Spec/AUC
  5. Per-Fold Metrics — Fold-level performance variance
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import os
import warnings as _w
from contextlib import contextmanager

import sys
import config as CFG


@contextmanager
def _suppress_warnings():
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        yield


# ═════════════════════════════════════════════════════════════════════════
# 1. EEG PREPROCESSING VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════
def plot_eeg_visualizations(records, plot_dir, sfreq=256):
    """Generate raw signal, PSD, and topomap plots for the first subject."""
    if not records:
        print("[viz] No records to visualize.")
        return

    rec = records[0]
    epochs = rec["epochs"]
    raw_data = epochs.get_data()
    ch_names = epochs.info['ch_names']
    label_str = "AD" if rec["label"] == 1 else "Normal"

    concat = raw_data[:5].reshape(len(ch_names), -1)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    with _suppress_warnings():
        info.set_montage('standard_1020', on_missing='ignore')
    raw = mne.io.RawArray(concat, info, verbose=False)

    os.makedirs(plot_dir, exist_ok=True)

    _plot_raw_signal(raw, ch_names, sfreq, label_str, plot_dir)
    _plot_psd(raw, label_str, plot_dir)
    _plot_topomaps(raw, label_str, plot_dir)
    print(f"[viz] Saved 3 EEG visualization plots to {plot_dir}/")


def _plot_raw_signal(raw, ch_names, sfreq, label_str, plot_dir):
    n_ch = len(ch_names)
    fig, axes = plt.subplots(n_ch, 1, figsize=(15, max(12, n_ch * 1.2)), sharex=True)
    if n_ch == 1: axes = [axes]
    times = np.arange(raw.n_times) / sfreq
    t_end = min(int(20 * sfreq), raw.n_times)
    data = raw.get_data()
    for i, ch in enumerate(ch_names):
        axes[i].plot(times[:t_end], data[i, :t_end] * 1e6, lw=0.6, color='steelblue')
        axes[i].set_ylabel(ch, fontsize=7, rotation=0, labelpad=28)
        axes[i].set_yticks([])
        for spine in ['top', 'right', 'left']:
            axes[i].spines[spine].set_visible(False)
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(f'EEG Raw Signal – {label_str} Subject (first 20s)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'viz_raw_signal.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_psd(raw, label_str, plot_dir):
    psd = raw.compute_psd(method='welch', fmin=1, fmax=45, n_fft=2048, verbose=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    psd.plot(axes=ax, show=False)
    ax.axvspan(1, 4, alpha=0.15, color='purple', label='Delta')
    ax.axvspan(4, 8, alpha=0.15, color='blue', label='Theta')
    ax.axvspan(8, 13, alpha=0.15, color='green', label='Alpha')
    ax.axvspan(13, 30, alpha=0.15, color='orange', label='Beta')
    ax.set_title(f'Power Spectral Density – {label_str} Subject', fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'viz_psd.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_topomaps(raw, label_str, plot_dir):
    psd = raw.compute_psd(method='welch', fmin=1, fmax=45, n_fft=2048, verbose=False)
    bands = {
        'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13), 'Beta (13-30 Hz)': (13, 30),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (bname, (fmin, fmax)) in zip(axes, bands.items()):
        psd.plot_topomap(bands={bname: (fmin, fmax)}, axes=ax, show=False)
        ax.set_title(bname, fontsize=10)
    fig.suptitle(f'Band Power Topomaps – {label_str} Subject', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'viz_topomaps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# 1b. SCALOGRAM VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════
def plot_scalograms(X_scalo, y, freqs, sfreq, plot_dir, ch_names=None):
    """
    Visualize CWT scalograms: AD vs Normal comparison.

    Plots a 2×4 grid showing 4 key channels for one AD epoch and one Normal epoch.
    Scalograms show how spectral power evolves over time — AD subjects
    typically show increased low-frequency (delta/theta) power and reduced
    alpha/beta power compared to healthy controls.

    Parameters
    ----------
    X_scalo : ndarray (N, n_ch, n_freqs, n_times)
    y : ndarray (N,)
    freqs : ndarray — CWT center frequencies
    sfreq : float — sampling rate
    plot_dir : str
    ch_names : list of str or None
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Pick one AD and one Normal epoch
    ad_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    if len(ad_idx) == 0 or len(norm_idx) == 0:
        print("[viz] Need both AD and Normal epochs for scalogram comparison")
        return

    ad_epoch = X_scalo[ad_idx[0]]      # (n_ch, F, T)
    norm_epoch = X_scalo[norm_idx[0]]   # (n_ch, F, T)

    n_ch = ad_epoch.shape[0]
    n_times = ad_epoch.shape[2]
    times = np.arange(n_times) / sfreq

    # Select 4 representative channels (frontal, central, parietal, occipital)
    if n_ch >= 16:
        show_ch = [0, 4, 8, 12]  # Fp1-ish, C3-ish, P3-ish, O1-ish
    elif n_ch >= 4:
        show_ch = [0, n_ch//4, n_ch//2, 3*n_ch//4]
    else:
        show_ch = list(range(n_ch))

    n_show = len(show_ch)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col, ch_idx in enumerate(show_ch):
        ch_label = ch_names[ch_idx] if ch_names and ch_idx < len(ch_names) else f"Ch {ch_idx}"

        for row, (data, label, cmap) in enumerate([
            (norm_epoch[ch_idx], 'Normal', 'viridis'),
            (ad_epoch[ch_idx], 'AD', 'inferno'),
        ]):
            ax = axes[row, col]
            im = ax.pcolormesh(times, freqs, data, shading='auto', cmap=cmap)
            ax.set_ylabel('Frequency (Hz)' if col == 0 else '', fontsize=9)
            ax.set_xlabel('Time (s)' if row == 1 else '', fontsize=9)
            ax.set_title(f'{ch_label} — {label}', fontsize=10)

            # Band boundary lines
            for bnd, bname in [(4, 'δ/θ'), (8, 'θ/α'), (13, 'α/β'), (30, 'β/γ')]:
                if freqs[0] <= bnd <= freqs[-1]:
                    ax.axhline(bnd, color='white', linewidth=0.5, linestyle='--', alpha=0.6)

            plt.colorbar(im, ax=ax, pad=0.02, aspect=20)

    fig.suptitle('CWT Scalograms — Normal vs AD\n(Time-Frequency Power Distribution)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'viz_scalograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Scalogram comparison saved to {plot_dir}/viz_scalograms.png")


# ═════════════════════════════════════════════════════════════════════════
# 2. BIOMARKER FEATURE HEATMAP
# ═════════════════════════════════════════════════════════════════════════
def plot_biomarker_heatmap(X_bio, y, feature_names, plot_dir):
    """
    Heatmap showing mean feature values for AD vs Normal.

    Parameters
    ----------
    X_bio : ndarray (N, n_features)
    y : ndarray (N,)
    feature_names : list of str
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Select top 30 most discriminative features (by |mean_AD - mean_Normal|)
    ad_mask = y == 1
    norm_mask = y == 0

    mean_ad = X_bio[ad_mask].mean(axis=0)
    mean_norm = X_bio[norm_mask].mean(axis=0)
    diff = np.abs(mean_ad - mean_norm)

    n_show = min(30, len(feature_names))
    top_idx = np.argsort(diff)[-n_show:][::-1]

    data = np.stack([mean_norm[top_idx], mean_ad[top_idx]], axis=1)
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, max(6, n_show * 0.3)))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'AD'], fontsize=11)
    ax.set_title('Top 30 Discriminative Biomarker Features\n(Z-scored Mean)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Z-score')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'biomarker_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Biomarker heatmap saved to {plot_dir}/biomarker_heatmap.png")


def plot_biomarker_importance(X_bio, y, feature_names, plot_dir):
    """
    Train a Random Forest classifier to rank global biomarker feature importance.
    Plots and saves the top 20 most discriminative features across the entire dataset.
    """
    print("\n  [viz] Training Random Forest for global feature importance...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Train RF on the entire dataset to get global importances
    rf = RandomForestClassifier(n_estimators=100, random_state=CFG.SEED, n_jobs=-1)
    rf.fit(X_bio, y)
    
    importances = rf.feature_importances_
    
    # Sort and get top 20
    n_show = min(20, len(feature_names))
    indices = np.argsort(importances)[::-1][:n_show]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Reverse to plot highest at the top
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances[::-1], color='#8e44ad', alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
    ax.set_title("Global Biomarker Feature Importance (Random Forest)", fontsize=14, pad=15)
    
    # Add value labels
    for i, v in enumerate(top_importances[::-1]):
        ax.text(v + (max(top_importances)*0.01), i, f"{v:.4f}", va='center', fontsize=8)
        
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(plot_dir, "biomarker_importance.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [viz] Global feature importance plot saved to {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# 3. TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, fold_num, plot_dir, filename=None, title_prefix=""):
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict with keys 'tr_loss', 'val_loss', 'tr_acc', 'val_acc' (lists)
    fold_num : int or str
    """
    os.makedirs(plot_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['tr_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['tr_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title_prefix}Loss Curves'.strip())
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['tr_acc'], 'b-', label='Train Acc', linewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title_prefix}Accuracy Curves'.strip())
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Shade overfitting region
    gap = [t - v for t, v in zip(history['tr_acc'], history['val_acc'])]
    ax2.fill_between(epochs, history['val_acc'], history['tr_acc'],
                     alpha=0.15, color='red', label='Gap')

    plt.tight_layout()
    out_filename = filename if filename else f'training_curves_fold{fold_num}.png'
    fig.savefig(os.path.join(plot_dir, out_filename),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, plot_dir, filename='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plot and save a beautiful confusion matrix using seaborn heatmap.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    os.makedirs(plot_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Confusion matrix saved to {save_path}")


# ═════════════════════════════════════════════════════════════════════════
# 4. MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════
def plot_model_comparison(all_results, plot_dir):
    """Bar chart + table comparing all models."""
    os.makedirs(plot_dir, exist_ok=True)

    model_names = list(all_results.keys())
    metrics = ['acc', 'sens', 'spec', 'auc']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']

    n_models = len(model_names)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_models)
    width = 0.18
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [all_results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison – NeuroFusion-AD Pipeline V3', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print table
    print("\n" + "=" * 75)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'AUC':>8}")
    print("-" * 75)
    for name in model_names:
        r = all_results[name]
        print(f"  {name:<20} {r['acc']:>10.4f} {r['sens']:>12.4f} {r['spec']:>12.4f} {r['auc']:>8.4f}")
    print("=" * 75)
    print(f"\n[viz] Comparison chart saved to {plot_dir}/model_comparison.png")


# ═════════════════════════════════════════════════════════════════════════
# 5. PER-FOLD METRICS
# ═════════════════════════════════════════════════════════════════════════
def plot_fold_metrics(fold_metrics, model_name, plot_dir):
    """
    Bar chart showing per-fold Acc/AUC for a single model.

    Parameters
    ----------
    fold_metrics : list of dicts [{acc, sens, spec, auc}, ...]
    model_name : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    n_folds = len(fold_metrics)
    folds = [f'Fold {i+1}' for i in range(n_folds)]

    accs = [m['acc'] for m in fold_metrics]
    aucs = [m['auc'] for m in fold_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_folds)
    width = 0.35

    ax.bar(x - width/2, accs, width, label='Accuracy', color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, aucs, width, label='AUC', color='#9C27B0', alpha=0.85)

    # Annotate
    for i in range(n_folds):
        ax.text(x[i] - width/2, accs[i] + 0.01, f'{accs[i]:.3f}',
                ha='center', fontsize=9)
        ax.text(x[i] + width/2, aucs[i] + 0.01, f'{aucs[i]:.3f}',
                ha='center', fontsize=9)

    ax.axhline(np.mean(accs), color='#2196F3', linestyle='--', alpha=0.5, label=f'Mean Acc: {np.mean(accs):.3f}')
    ax.axhline(np.mean(aucs), color='#9C27B0', linestyle='--', alpha=0.5, label=f'Mean AUC: {np.mean(aucs):.3f}')

    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Per-Fold Performance', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'fold_metrics_{model_name.lower().replace(" ", "_")}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# 5. ABLATION STUDY RESULTS
# ═════════════════════════════════════════════════════════════════════════
def plot_ablation_results(results_dict, plot_dir):
    """
    Plot grouped bar chart comparing ablation variants.

    Parameters
    ----------
    results_dict : dict
        {variant_name: {'acc': val, 'sens': val, 'spec': val, 'auc': val}}
    plot_dir : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    variants = list(results_dict.keys())
    metrics = ['acc', 'sens', 'spec', 'auc']
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']

    x = np.arange(len(variants))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results_dict[v][metric] for v in variants]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)

        # Value labels
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Score')
    ax.set_title('Branch Ablation Study — Performance Contribution', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'ablation_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Ablation comparison chart saved to {save_path}")


# ═════════════════════════════════════════════════════════════════════════
# 6. TIME-FREQUENCY METHOD COMPARISON
# ═════════════════════════════════════════════════════════════════════════
def plot_tf_comparison_samples(tf_dict, y, freqs, sfreq, plot_dir, ch_names=None):
    """
    Side-by-side visualization comparing different TFR methods on the
    same EEG epoch (one AD, one Normal).

    Produces a grid: rows = [Normal, AD], columns = [method1, method2, ...]
    for a single representative channel.

    Parameters
    ----------
    tf_dict : dict
        {method_name: ndarray (N, C, F, T)} — TFR output per method.
    y : ndarray (N,)
    freqs : ndarray — target frequency bins
    sfreq : float
    plot_dir : str
    ch_names : list of str or None
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(tf_dict.keys())
    n_methods = len(methods)

    # Pick one AD and one Normal epoch
    ad_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    if len(ad_idx) == 0 or len(norm_idx) == 0:
        print("[viz] Need both AD and Normal epochs for TFR comparison")
        return

    ad_i = ad_idx[0]
    norm_i = norm_idx[0]

    # Use channel 0 (Fp1) for comparison
    ch_idx = 0
    ch_label = ch_names[ch_idx] if ch_names and ch_idx < len(ch_names) else "Ch 0"

    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 7))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    for col, method in enumerate(methods):
        X_tf = tf_dict[method]
        n_times = X_tf.shape[3]
        times = np.arange(n_times) / (sfreq / 4)  # account for decimation

        for row, (epoch_idx, label, cmap) in enumerate([
            (norm_i, 'Normal', 'viridis'),
            (ad_i, 'AD', 'inferno'),
        ]):
            ax = axes[row, col]
            data = X_tf[epoch_idx, ch_idx]  # (F, T)

            im = ax.pcolormesh(
                times, freqs, data,
                shading='auto', cmap=cmap
            )
            ax.set_ylabel('Frequency (Hz)' if col == 0 else '', fontsize=9)
            ax.set_xlabel('Time (s)' if row == 1 else '', fontsize=9)
            ax.set_title(f'{method.upper()} — {label} ({ch_label})', fontsize=10,
                         fontweight='bold')

            # Band boundary lines
            for bnd in [4, 8, 13, 30]:
                if freqs[0] <= bnd <= freqs[-1]:
                    ax.axhline(bnd, color='white', linewidth=0.5,
                               linestyle='--', alpha=0.6)

            plt.colorbar(im, ax=ax, pad=0.02, aspect=20)

    fig.suptitle(
        'Time-Frequency Representations — STFT vs CWT\n'
        '(Same epoch, same channel)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_comparison_samples.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] TFR comparison saved to {save_path}")


def plot_tf_method_comparison(results_dict, plot_dir):
    """
    Bar chart comparing classification metrics across TFR methods.

    Parameters
    ----------
    results_dict : dict
        {method_name: {'acc': val, 'sens': val, 'spec': val, 'auc': val}}
    plot_dir : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(results_dict.keys())
    metrics = ['acc', 'sens', 'spec', 'auc']
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, 4 * n_methods), 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results_dict[m][metric] for m in methods]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    ax.set_xlabel('TFR Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Time-Frequency Method Comparison — STFT vs CWT\n'
        'FusionNet Performance with Different TF Representations',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in methods], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_method_comparison.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Print results table
    print(f"\n{'═' * 70}")
    print("  TIME-FREQUENCY METHOD COMPARISON")
    print(f"{'═' * 70}")
    print(f"  {'Method':<10} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'AUC':>8}")
    print("-" * 70)
    for m in methods:
        r = results_dict[m]
        print(f"  {m.upper():<10} {r['acc']:>8.4f} {r['sens']:>8.4f} "
              f"{r['spec']:>8.4f} {r['auc']:>8.4f}")
    print(f"{'═' * 70}")
    print(f"\n[viz] TFR method comparison saved to {save_path}")


def plot_tf_auc_comparison(results_dict, plot_dir):
    """
    Dedicated AUC score comparison plot — horizontal lollipop chart
    with delta annotations showing improvement/degradation vs baseline.

    Parameters
    ----------
    results_dict : dict
        {method_name: {'acc': val, 'sens': val, 'spec': val, 'auc': val}}
    plot_dir : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(results_dict.keys())
    aucs = [results_dict[m]['auc'] for m in methods]
    n = len(methods)

    # Color gradient based on AUC value
    best_auc = max(aucs)
    worst_auc = min(aucs)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 1.5)))

    # Sort by AUC descending
    sorted_pairs = sorted(zip(methods, aucs), key=lambda x: x[1])
    s_methods, s_aucs = zip(*sorted_pairs)

    y_pos = np.arange(n)

    # Color map: red (worst) → green (best)
    norm = plt.Normalize(vmin=min(s_aucs) - 0.05, vmax=max(s_aucs) + 0.05)
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in s_aucs]

    # Lollipop stems
    for i, (method, auc, color) in enumerate(zip(s_methods, s_aucs, colors)):
        ax.hlines(y=i, xmin=0.5, xmax=auc, color=color, linewidth=3, alpha=0.7)
        ax.scatter(auc, i, color=color, s=200, zorder=5, edgecolors='black',
                   linewidth=1.5)

        # Value label
        ax.text(auc + 0.008, i, f'{auc:.4f}', va='center', fontsize=11,
                fontweight='bold')

        # Delta annotation (vs first method as baseline)
        if n > 1:
            baseline = s_aucs[0]
            delta = auc - baseline
            if i > 0 and abs(delta) > 0.001:
                sign = '+' if delta > 0 else ''
                delta_color = '#27ae60' if delta > 0 else '#e74c3c'
                ax.text(auc + 0.008, i - 0.25, f'({sign}{delta:.4f})',
                        va='center', fontsize=8, color=delta_color,
                        fontstyle='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.upper() for m in s_methods], fontsize=12,
                       fontweight='bold')
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('AUC Score Comparison — Time-Frequency Methods',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(min(s_aucs) - 0.1, max(s_aucs) + 0.08)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_auc_comparison.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] AUC comparison saved to {save_path}")


def plot_tf_radar_chart(results_dict, plot_dir):
    """
    Radar (spider) chart overlaying all metrics for each TFR method.
    Shows the full performance profile at a glance.

    Parameters
    ----------
    results_dict : dict
        {method_name: {'acc': val, 'sens': val, 'spec': val, 'auc': val}}
    plot_dir : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(results_dict.keys())
    metrics = ['acc', 'sens', 'spec', 'auc']
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    n_metrics = len(metrics)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Method colors
    method_colors = ['#2196F3', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(methods):
        values = [results_dict[method][m] for m in metrics]
        values += values[:1]  # close the polygon

        color = method_colors[i % len(method_colors)]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method.upper(),
                color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Performance Radar — TFR Method Comparison',
                 fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_radar_chart.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Radar chart saved to {save_path}")


def plot_tf_difference_maps(tf_dict, y, freqs, sfreq, plot_dir, ch_names=None):
    """
    Mean TFR difference maps: (Mean AD) − (Mean Normal) for each method.

    Highlights the time-frequency regions where AD and Normal differ most.
    Warm colors = higher power in AD; cool colors = lower power in AD.
    This is a key visualization for clinical interpretability.

    Parameters
    ----------
    tf_dict : dict
        {method_name: ndarray (N, C, F, T)}
    y : ndarray (N,)
    freqs : ndarray
    sfreq : float
    plot_dir : str
    ch_names : list or None
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(tf_dict.keys())
    n_methods = len(methods)

    ad_mask = y == 1
    norm_mask = y == 0

    if ad_mask.sum() == 0 or norm_mask.sum() == 0:
        print("[viz] Need both classes for difference maps")
        return

    # Use channel 0 (Fp1) and channel at ~50% (central region)
    ch_indices = [0]
    first_tf = tf_dict[methods[0]]
    n_ch = first_tf.shape[1]
    if n_ch > 4:
        ch_indices.append(n_ch // 2)  # central channel

    n_ch_show = len(ch_indices)
    fig, axes = plt.subplots(n_ch_show, n_methods, figsize=(6 * n_methods, 5 * n_ch_show))

    if n_methods == 1 and n_ch_show == 1:
        axes = np.array([[axes]])
    elif n_methods == 1:
        axes = axes.reshape(-1, 1)
    elif n_ch_show == 1:
        axes = axes.reshape(1, -1)

    for col, method in enumerate(methods):
        X_tf = tf_dict[method]
        n_times = X_tf.shape[3]
        times = np.arange(n_times) / (sfreq / 4)

        mean_ad = X_tf[ad_mask].mean(axis=0)     # (C, F, T)
        mean_norm = X_tf[norm_mask].mean(axis=0)  # (C, F, T)
        diff = mean_ad - mean_norm                # (C, F, T)

        for row, ch_idx in enumerate(ch_indices):
            ax = axes[row, col]
            ch_label = ch_names[ch_idx] if ch_names and ch_idx < len(ch_names) else f"Ch {ch_idx}"

            vmax = np.abs(diff[ch_idx]).max()
            vmin = -vmax

            im = ax.pcolormesh(
                times, freqs, diff[ch_idx],
                shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
            )
            ax.set_ylabel('Frequency (Hz)' if col == 0 else '', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_title(f'{method.upper()} — ΔPower ({ch_label})\n'
                         f'(AD − Normal)', fontsize=10, fontweight='bold')

            # Band boundaries
            for bnd in [4, 8, 13, 30]:
                if freqs[0] <= bnd <= freqs[-1]:
                    ax.axhline(bnd, color='black', linewidth=0.7,
                               linestyle='--', alpha=0.4)

            cb = plt.colorbar(im, ax=ax, pad=0.02, aspect=20)
            cb.set_label('ΔAmplitude', fontsize=8)

    fig.suptitle(
        'TFR Difference Maps — Where AD Differs from Normal\n'
        'Red = Higher in AD | Blue = Lower in AD',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_difference_maps.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Difference maps saved to {save_path}")


def plot_tf_band_energy(tf_dict, y, freqs, plot_dir):
    """
    Band-specific energy distribution for each TFR method.

    Shows how each method captures power across EEG frequency bands
    (delta, theta, alpha, beta) for AD vs Normal. This helps explain
    *why* one method might outperform another — e.g., if STFT captures
    alpha suppression more clearly than CWT.

    Parameters
    ----------
    tf_dict : dict
        {method_name: ndarray (N, C, F, T)}
    y : ndarray (N,)
    freqs : ndarray
    plot_dir : str
    """
    os.makedirs(plot_dir, exist_ok=True)

    methods = list(tf_dict.keys())
    bands = {
        'Delta\n(1-4 Hz)': (1, 4),
        'Theta\n(4-8 Hz)': (4, 8),
        'Alpha\n(8-13 Hz)': (8, 13),
        'Beta\n(13-30 Hz)': (13, 30),
        'Gamma\n(30-44 Hz)': (30, 44),
    }

    ad_mask = y == 1
    norm_mask = y == 0

    if ad_mask.sum() == 0 or norm_mask.sum() == 0:
        print("[viz] Need both classes for band energy plot")
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        X_tf = tf_dict[method]

        band_names = list(bands.keys())
        n_bands = len(band_names)
        x = np.arange(n_bands)
        width = 0.35

        ad_energies = []
        norm_energies = []

        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if freq_mask.sum() == 0:
                ad_energies.append(0)
                norm_energies.append(0)
                continue

            # Mean energy across channels, time, and frequency bins in band
            ad_energy = X_tf[ad_mask][:, :, freq_mask, :].mean()
            norm_energy = X_tf[norm_mask][:, :, freq_mask, :].mean()
            ad_energies.append(float(ad_energy))
            norm_energies.append(float(norm_energy))

        bars_norm = ax.bar(x - width / 2, norm_energies, width,
                           label='Normal', color='#3498db', alpha=0.85,
                           edgecolor='black', linewidth=0.5)
        bars_ad = ax.bar(x + width / 2, ad_energies, width,
                         label='AD', color='#e74c3c', alpha=0.85,
                         edgecolor='black', linewidth=0.5)

        # Value labels
        for bars in [bars_norm, bars_ad]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.001,
                            f'{h:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=9)
        ax.set_title(f'{method.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frequency Band', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Mean Amplitude', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Band-Specific Energy Distribution — AD vs Normal\n'
        'How each TFR method captures EEG spectral patterns',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'tf_band_energy.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Band energy distribution saved to {save_path}")

