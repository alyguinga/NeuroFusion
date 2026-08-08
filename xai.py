"""
v3/xai_v3.py
═══════════════════════════════════════════════════════════════════════════
Explainable AI (XAI) for the NeuroFusion-AD pipeline.

Implements two complementary explainability methods:

1. Grad-CAM for CViT Scalogram Branch:
   ─────────────────────────────────────
   Gradient-weighted Class Activation Mapping highlights which
   time-frequency regions in the scalogram drive the AD classification.

   Formula:
     α_k = (1/Z) × Σ_i Σ_j (∂y_c / ∂A^k_{ij})     — importance weights
     L_GradCAM = ReLU(Σ_k α_k × A^k)                 — weighted activation
   
   where A^k = feature map k, y_c = score for class c.
   ReLU ensures we only show features with positive influence on the class.
   (Selvaraju et al., 2017; Frontiers 2025 for EEG application)

2. Feature Attribution for BiomarkerNet:
   ──────────────────────────────────────
   Gradient × Input attribution identifies which domain biomarker
   features (PAF, θ/α ratio, SampEn, etc.) most influence the prediction.

   Formula:
     Attribution_i = x_i × (∂y_c / ∂x_i)

   Positive = pushes toward AD, Negative = pushes toward Normal.
   (Shrikumar et al., 2017; similar to SHAP for linear models)

3. Cross-Path Attention Weights:
   ──────────────────────────────
   The fusion layer's attention weights reveal which branches (temporal,
   time-frequency, spatial, domain) are most important per sample.
   No additional computation needed — extracted from forward pass.

References:
  - Selvaraju et al., 2017: Grad-CAM
  - Shrikumar et al., 2017: DeepLIFT / Gradient × Input
  - Frontiers 2025: XAI for clinical EEG acceptance
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

import sys
import config as CFG


def grad_cam_cvit(model, x_1d, x_scalo, x_adj, x_bio, target_class=1, device='cpu'):
    """
    Compute Grad-CAM heatmap for the CViT (scalogram) branch.

    Highlights which time-frequency regions drive the classification.

    Parameters
    ----------
    model : FusionNet — trained model
    x_1d, x_scalo, x_adj, x_bio : single-sample tensors (add batch dim)
    target_class : int — class to explain (1=AD)
    device : str

    Returns
    -------
    heatmap : ndarray (F, T) — Grad-CAM activation map
    """
    model.eval()

    # Ensure batch dimension
    if x_1d.dim() == 3:
        x_1d = x_1d.unsqueeze(0)
    if x_scalo.dim() == 3:
        x_scalo = x_scalo.unsqueeze(0)
    if x_adj.dim() == 2:
        x_adj = x_adj.unsqueeze(0)
    if x_bio is not None and x_bio.dim() == 1:
        x_bio = x_bio.unsqueeze(0)

    x_1d = x_1d.to(device).requires_grad_(False)
    x_scalo = x_scalo.to(device).requires_grad_(True)
    x_adj = x_adj.to(device).requires_grad_(False)
    if x_bio is not None:
        x_bio = x_bio.to(device).requires_grad_(False)

    # Forward pass — get CViT intermediate activations
    # Hook into CViT patch embedding output
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['cvit_embed'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['cvit_embed'] = grad_output[0]

    # Register hooks on CViT patch embedding (the CNN backbone)
    handle_fwd = model.cvit.conv.register_forward_hook(forward_hook)
    handle_bwd = model.cvit.conv.register_full_backward_hook(backward_hook)

    # Forward
    logits, attn_weights = model(x_1d, x_scalo, x_adj, x_bio)
    score = logits[0, target_class]

    # Backward
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if 'cvit_embed' not in activations or 'cvit_embed' not in gradients:
        print("[xai] Warning: Could not capture CViT activations")
        return np.zeros((x_scalo.shape[2], x_scalo.shape[3]))

    # Compute Grad-CAM
    # α_k = global average of gradients (importance weights)
    grads = gradients['cvit_embed'].detach().cpu().numpy()[0]  # (embed_dim, H_out, W_out)
    acts = activations['cvit_embed'].detach().cpu().numpy()[0]

    # Weight each feature map by gradient importance
    weights = np.mean(grads, axis=(1, 2))  # (embed_dim,)
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)  # (H_out, W_out)
    
    heatmap = cam

    # ReLU — only positive contributions
    heatmap = np.maximum(heatmap, 0)

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def feature_attribution(model, x_1d, x_scalo, x_adj, x_bio,
                        feature_names, target_class=1, device='cpu'):
    """
    Gradient × Input attribution for biomarker features.

    Attribution_i = x_i × (∂y_c / ∂x_i)

    Parameters
    ----------
    model : FusionNet
    x_bio : Tensor (n_features,) — single sample biomarker features
    feature_names : list of str
    target_class : int

    Returns
    -------
    attributions : dict {feature_name: attribution_value}
    """
    model.eval()

    if x_1d.dim() == 3: x_1d = x_1d.unsqueeze(0)
    if x_scalo.dim() == 3: x_scalo = x_scalo.unsqueeze(0)
    if x_adj.dim() == 2: x_adj = x_adj.unsqueeze(0)
    if x_bio.dim() == 1: x_bio = x_bio.unsqueeze(0)

    x_1d = x_1d.to(device)
    x_scalo = x_scalo.to(device)
    x_adj = x_adj.to(device)
    x_bio = x_bio.to(device).requires_grad_(True)

    logits, _ = model(x_1d, x_scalo, x_adj, x_bio)
    score = logits[0, target_class]

    model.zero_grad()
    score.backward()

    grads = x_bio.grad.detach().cpu().numpy()[0]
    inputs = x_bio.detach().cpu().numpy()[0]

    # Attribution = input × gradient
    attr = inputs * grads

    # Map to feature names (take top features)
    n_feats = min(len(feature_names), len(attr))
    attributions = {feature_names[i]: float(attr[i]) for i in range(n_feats)}

    return attributions


def plot_xai_results(model, x_1d, x_scalo, x_adj, x_bio,
                     feature_names, y_true, device='cpu', save_dir=None):
    """
    Generate and save all XAI visualizations.

    Produces:
    1. Grad-CAM heatmap for scalogram branch
    2. Top-20 biomarker feature attributions
    3. Cross-path attention weight distribution
    """
    if save_dir is None:
        save_dir = CFG.XAI_DIR
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # Use first sample
    s_1d = x_1d[0]
    s_scalo = x_scalo[0]
    s_adj = x_adj[0]
    s_bio = x_bio[0] if x_bio is not None else None
    label = y_true[0] if hasattr(y_true, '__len__') else y_true

    target_class = 1  # explain AD class

    # ── 1. Grad-CAM ──────────────────────────────────────────────────
    try:
        heatmap = grad_cam_cvit(model, s_1d, s_scalo, s_adj, s_bio,
                                target_class=target_class, device=device)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(heatmap, aspect='auto', cmap='hot', interpolation='bilinear')
        ax.set_title('Grad-CAM: Scalogram Regions Driving AD Classification', fontsize=12)
        ax.set_xlabel('Patch Position')
        ax.set_ylabel('Patch Position')
        plt.colorbar(im, ax=ax, label='Importance')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'gradcam_scalogram.png'), dpi=150)
        plt.close(fig)
        print(f"  [xai] Grad-CAM saved to {save_dir}/gradcam_scalogram.png")
    except Exception as e:
        print(f"  [xai] Grad-CAM failed: {e}")

    # ── 2. Feature Attribution ───────────────────────────────────────
    if s_bio is not None and feature_names:
        try:
            attr = feature_attribution(model, s_1d, s_scalo, s_adj, s_bio,
                                       feature_names, target_class=target_class,
                                       device=device)

            # Top 20 by absolute value
            sorted_attr = sorted(attr.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
            names = [a[0] for a in sorted_attr]
            values = [a[1] for a in sorted_attr]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
            ax.barh(range(len(names)), values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Attribution (Gradient × Input)', fontsize=10)
            ax.set_title('Top 20 Biomarker Feature Attributions for AD', fontsize=12)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.invert_yaxis()

            # Legend
            from matplotlib.patches import Patch
            legend = [Patch(color='#e74c3c', label='Pushes toward AD'),
                      Patch(color='#3498db', label='Pushes toward Normal')]
            ax.legend(handles=legend, loc='lower right', fontsize=9)

            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'feature_attribution.png'), dpi=150)
            plt.close(fig)
            print(f"  [xai] Feature attribution saved to {save_dir}/feature_attribution.png")
        except Exception as e:
            print(f"  [xai] Feature attribution failed: {e}")

    # ── 3. Cross-Path Attention Weights ──────────────────────────────
    try:
        with torch.no_grad():
            if s_1d.dim() == 3: s_1d_b = s_1d.unsqueeze(0).to(device)
            else: s_1d_b = s_1d.unsqueeze(0).to(device)
            s_scalo_b = s_scalo.unsqueeze(0).to(device) if s_scalo.dim() == 3 else s_scalo.unsqueeze(0).to(device)
            s_adj_b = s_adj.unsqueeze(0).to(device) if s_adj.dim() == 2 else s_adj.unsqueeze(0).to(device)
            s_bio_b = s_bio.unsqueeze(0).to(device) if s_bio is not None and s_bio.dim() == 1 else (s_bio.to(device) if s_bio is not None else None)

            _, attn = model(s_1d_b, s_scalo_b, s_adj_b, s_bio_b)

        attn_np = attn.cpu().numpy()[0]  # (n_branches, n_branches)
        has_gnn = hasattr(model, 'use_gnn') and model.use_gnn and ('gnn' not in model.disabled_branches)
        branch_names = ['EEGNet\n(Temporal)', 'CViT\n(Time-Freq)']
        if has_gnn:
            branch_names.append('GNN\n(Spatial)')
        if s_bio is not None:
            branch_names.append('BiomarkerNet\n(Domain)')

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(attn_np, cmap='YlOrRd', vmin=0)
        ax.set_xticks(range(len(branch_names)))
        ax.set_yticks(range(len(branch_names)))
        ax.set_xticklabels(branch_names, fontsize=9)
        ax.set_yticklabels(branch_names, fontsize=9)

        # Annotate with values
        for i in range(attn_np.shape[0]):
            for j in range(attn_np.shape[1]):
                ax.text(j, i, f'{attn_np[i,j]:.3f}', ha='center', va='center', fontsize=10)

        ax.set_title('Cross-Path Attention Weights\n(Row attends to Column)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'cross_path_attention.png'), dpi=150)
        plt.close(fig)
        print(f"  [xai] Attention weights saved to {save_dir}/cross_path_attention.png")
    except Exception as e:
        print(f"  [xai] Attention weight plot failed: {e}")


# ═════════════════════════════════════════════════════════════════════════
# POPULATION-LEVEL ATTRIBUTION
# ═════════════════════════════════════════════════════════════════════════
def population_attribution(model, x_1d, x_scalo, x_adj, x_bio, y,
                           feature_names, n_per_class=50, target_class=1,
                           device='cpu', save_dir=None):
    """
    Average Gradient × Input attributions across multiple samples per class.

    Single-sample attribution is noisy. Averaging over 50+ samples per class
    gives stable, population-level feature importance rankings.

    Parameters
    ----------
    model : FusionNet
    x_1d, x_scalo, x_adj, x_bio : Tensors (N, ...)
    y : ndarray (N,)
    feature_names : list of str
    n_per_class : int — samples per class to average
    target_class : int — class to explain (1=AD)
    device : str
    save_dir : str or None

    Returns
    -------
    mean_attr : dict {feature_name: mean_attribution}
    """
    if save_dir is None:
        save_dir = CFG.XAI_DIR
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # Collect indices per class
    ad_idx = np.where(y == 1)[0][:n_per_class]
    norm_idx = np.where(y == 0)[0][:n_per_class]

    class_attrs = {}  # {'AD': {feat: [vals]}, 'Normal': {feat: [vals]}}

    for label, indices in [('AD', ad_idx), ('Normal', norm_idx)]:
        all_attr = {fn: [] for fn in feature_names}

        for idx in indices:
            try:
                attr = feature_attribution(
                    model,
                    x_1d[idx], x_scalo[idx], x_adj[idx], x_bio[idx],
                    feature_names, target_class=target_class, device=device
                )
                for fn, val in attr.items():
                    all_attr[fn].append(val)
            except Exception:
                continue

        # Compute mean per feature
        class_attrs[label] = {
            fn: float(np.mean(vals)) if vals else 0.0
            for fn, vals in all_attr.items()
        }

    # ── Plot: Top 20 features by |mean attribution| across both classes ──
    # Rank by absolute AD attribution
    ad_attr = class_attrs.get('AD', {})
    sorted_feats = sorted(ad_attr.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    feat_names = [f[0] for f in sorted_feats]
    ad_vals = [ad_attr.get(fn, 0) for fn in feat_names]
    norm_vals = [class_attrs.get('Normal', {}).get(fn, 0) for fn in feat_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(feat_names))
    bar_h = 0.35

    ax.barh(y_pos - bar_h/2, ad_vals, bar_h, color='#e74c3c', alpha=0.8, label=f'AD (n={len(ad_idx)})')
    ax.barh(y_pos + bar_h/2, norm_vals, bar_h, color='#3498db', alpha=0.8, label=f'Normal (n={len(norm_idx)})')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel('Mean Attribution (Gradient × Input)', fontsize=10)
    ax.set_title(f'Population-Averaged Biomarker Attributions (n={len(ad_idx)+len(norm_idx)})', fontsize=12)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, 'population_attribution.png'), dpi=150)
    plt.close(fig)
    print(f"  [xai] Population attribution saved to {save_dir}/population_attribution.png")

    # ── Domain contribution pie chart ────────────────────────────────
    _plot_domain_contribution(ad_attr, feature_names, save_dir)

    return class_attrs


def _classify_domain(feature_name):
    """Classify a feature name into its domain category."""
    spectral_prefixes = ('abs_', 'rbp_', 'paf_', 'theta_alpha_ratio', 'ratio_')
    complexity_prefixes = ('sampen', 'pe_', 'hfd_', 'lzc_', 'hjorth_', 'composite_cx')
    # Anything else is connectivity
    if feature_name.startswith(spectral_prefixes):
        return 'Spectral'
    elif feature_name.startswith(complexity_prefixes):
        return 'Complexity'
    else:
        return 'Connectivity'


def _plot_domain_contribution(attributions, feature_names, save_dir):
    """
    Pie chart showing each domain's share of total |attribution|.

    Groups features by domain (spectral, complexity, connectivity)
    to visualize which domain drives the model's decisions.
    """
    domain_totals = {'Spectral': 0.0, 'Complexity': 0.0, 'Connectivity': 0.0}

    for fn in feature_names:
        val = abs(attributions.get(fn, 0.0))
        domain = _classify_domain(fn)
        domain_totals[domain] += val

    total = sum(domain_totals.values())
    if total < 1e-10:
        return

    labels = list(domain_totals.keys())
    sizes = [domain_totals[d] for d in labels]
    pcts = [100 * s / total for s in sizes]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax.set_title('Domain Contribution to AD Classification\n(by |Gradient × Input|)', fontsize=13)
    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, 'domain_contribution.png'), dpi=150)
    plt.close(fig)
    print(f"  [xai] Domain contribution pie chart saved to {save_dir}/domain_contribution.png")


# ═════════════════════════════════════════════════════════════════════════
# CROSS-PATH ATTENTION BY CLASS
# ═════════════════════════════════════════════════════════════════════════
def plot_attention_by_class(model, x_1d, x_scalo, x_adj, x_bio, y,
                            device='cpu', save_dir=None):
    """
    Compare cross-path attention weights between AD and Normal samples.

    Runs forward pass on all AD and Normal samples, collects 4×4 attention
    matrices, and plots side-by-side heatmaps showing which branches the
    fusion layer values more for each class.

    Parameters
    ----------
    model : FusionNet
    x_1d, x_scalo, x_adj, x_bio : Tensors (N, ...)
    y : ndarray (N,)
    device : str
    save_dir : str or None
    """
    if save_dir is None:
        save_dir = CFG.XAI_DIR
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    has_bio = x_bio is not None
    has_gnn = hasattr(model, 'use_gnn') and model.use_gnn and ('gnn' not in model.disabled_branches)

    branch_names = ['EEGNet\n(Temporal)', 'CViT\n(Time-Freq)']
    if has_gnn:
        branch_names.append('GNN\n(Spatial)')
    if has_bio:
        branch_names.append('BiomarkerNet\n(Domain)')

    class_attns = {}  # {'AD': list of (n_branches, n_branches), ...}

    for label, cls_val in [('AD', 1), ('Normal', 0)]:
        indices = np.where(y == cls_val)[0]
        attns = []

        with torch.no_grad():
            for idx in indices:
                s_1d = x_1d[idx].unsqueeze(0).to(device)
                s_scalo = x_scalo[idx].unsqueeze(0).to(device)
                s_adj = x_adj[idx].unsqueeze(0).to(device)
                s_bio = x_bio[idx].unsqueeze(0).to(device) if has_bio else None

                _, attn = model(s_1d, s_scalo, s_adj, s_bio)
                attns.append(attn.cpu().numpy()[0])

        if attns:
            class_attns[label] = np.mean(attns, axis=0)

    if len(class_attns) < 2:
        print("  [xai] Need both AD and Normal samples for attention comparison")
        return

    # ── Plot side-by-side heatmaps ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (label, attn_np) in zip(axes, class_attns.items()):
        im = ax.imshow(attn_np, cmap='YlOrRd', vmin=0,
                       vmax=max(class_attns['AD'].max(), class_attns['Normal'].max()))
        ax.set_xticks(range(len(branch_names)))
        ax.set_yticks(range(len(branch_names)))
        ax.set_xticklabels(branch_names, fontsize=9)
        ax.set_yticklabels(branch_names, fontsize=9)
        ax.set_title(f'{label} Samples (n={np.sum(y == (1 if label == "AD" else 0))})', fontsize=12)

        for i in range(attn_np.shape[0]):
            for j in range(attn_np.shape[1]):
                ax.text(j, i, f'{attn_np[i,j]:.3f}', ha='center', va='center', fontsize=10)

    fig.suptitle('Cross-Path Attention Weights — AD vs Normal\n(Row attends to Column)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes, label='Attention Weight', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'attention_by_class.png'), dpi=150)
    plt.close(fig)

    # ── Print BiomarkerNet attention summary ─────────────────────────
    if has_bio:
        bio_idx = len(branch_names) - 1
        for label, attn_np in class_attns.items():
            # Column sum = how much other branches attend TO biomarker
            attended_to = attn_np[:, bio_idx].mean()
            # Row sum = how much biomarker attends to others
            attends_to = attn_np[bio_idx, :].mean()
            print(f"  [xai] BiomarkerNet attention ({label}): "
                  f"attended_to={attended_to:.3f}, attends_to={attends_to:.3f}")

    print(f"  [xai] Attention comparison saved to {save_dir}/attention_by_class.png")

