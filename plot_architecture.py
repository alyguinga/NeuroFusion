"""
plot_architecture.py
═══════════════════════════════════════════════════════════════════════════
Generate NeuroFusion-AD model architecture diagrams and heatmaps.

Since our model is PyTorch (not TensorFlow/Keras), we use:
  1. torchinfo — detailed layer-by-layer summary table
  2. Custom matplotlib — publication-quality architecture diagram
  3. Attention heatmap — cross-path attention weights visualization
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG
from models.fusion_net import FusionNet


def plot_model_summary(model, X_1d, X_scalo, X_adj, X_bio, save_dir):
    """Print and save a detailed model summary using torchinfo."""
    from torchinfo import summary

    os.makedirs(save_dir, exist_ok=True)

    # Get the summary as string
    model_stats = summary(
        model,
        input_data=[X_1d, X_scalo, X_adj, X_bio],
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        col_width=18,
        row_settings=["var_names"],
        verbose=0
    )

    summary_str = str(model_stats)
    try:
        print(summary_str)
    except UnicodeEncodeError:
        print(summary_str.encode('ascii', 'replace').decode('ascii'))

    # Save to file
    txt_path = os.path.join(save_dir, 'model_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_str)
    print(f"\n[+] Model summary saved to {txt_path}")

    return model_stats


def plot_architecture_diagram(save_dir):
    """
    Publication-quality architecture diagram of the 4-branch FusionNet.
    Custom matplotlib visualization showing all branches, fusion, and classifier.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Color palette
    colors = {
        'input': '#3498db',
        'eegnet': '#e74c3c',
        'cvit': '#2ecc71',
        'gnn': '#f39c12',
        'bio': '#9b59b6',
        'fusion': '#1abc9c',
        'classifier': '#34495e',
        'output': '#e67e22',
        'arrow': '#7f8c8d',
    }

    def draw_box(ax, x, y, w, h, label, sublabel, color, fontsize=10):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='black',
            linewidth=1.5, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                    fontsize=7, color='white', fontstyle='italic', alpha=0.9)

    def draw_arrow(ax, x1, y1, x2, y2, color='#7f8c8d'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=2, connectionstyle='arc3,rad=0'))

    # ── Title ────────────────────────────────────────────────────────
    ax.text(10, 13.5, 'NeuroFusion-AD — 4-Branch Hybrid Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='#2c3e50')
    ax.text(10, 13.0, 'Cross-Path Multi-Head Self-Attention Fusion',
            ha='center', va='center', fontsize=12, color='#7f8c8d',
            fontstyle='italic')

    # ── Input Layer ──────────────────────────────────────────────────
    input_y = 11.5
    draw_box(ax, 8, input_y, 4, 0.8, 'EEG Input', '(B, 16, 1024) @ 256 Hz',
             colors['input'], fontsize=12)

    # ── Branch boxes ─────────────────────────────────────────────────
    branch_y = 9.2
    branch_h = 1.5
    branch_w = 3.8

    # Path A: EEGNet
    draw_box(ax, 0.3, branch_y, branch_w, branch_h,
             'Path A: EEGNet', 'Raw 1D → Temporal patterns\nConv → DepthwiseConv → SE → SepConv',
             colors['eegnet'], fontsize=11)

    # Path B: CViT
    draw_box(ax, 5.0, branch_y, branch_w, branch_h,
             'Path B: CViT', 'CWT/STFT Scalograms → TF patterns\nPatch Embed → Transformer × 3',
             colors['cvit'], fontsize=11)

    # Path C: GNN
    draw_box(ax, 9.7, branch_y, branch_w, branch_h,
             'Path C: GNN', 'PLI Connectivity → Spatial topology\nGATConv × 2 → Global Pool',
             colors['gnn'], fontsize=11)

    # Path D: BiomarkerNet
    draw_box(ax, 14.4, branch_y, branch_w, branch_h,
             'Path D: BiomarkerNet', 'Domain features → Expert biomarkers\nFC → BN → GELU → Dropout',
             colors['bio'], fontsize=11)

    # Arrows from input to branches
    for bx in [2.2, 6.9, 11.6, 16.3]:
        draw_arrow(ax, 10, input_y, bx, branch_y + branch_h, colors['arrow'])

    # ── Embedding projection ─────────────────────────────────────────
    embed_y = 7.2
    embed_h = 0.6
    for i, (bx, color, label) in enumerate([
        (0.3, colors['eegnet'], 'embed_eeg (48d)'),
        (5.0, colors['cvit'], 'embed_cvit (48d)'),
        (9.7, colors['gnn'], 'embed_gnn (48d)'),
        (14.4, colors['bio'], 'embed_bio (48d)'),
    ]):
        draw_box(ax, bx, embed_y, branch_w, embed_h, label, None, color, fontsize=9)
        draw_arrow(ax, bx + branch_w/2, branch_y, bx + branch_w/2, embed_y + embed_h, color)

    # ── Stack → Cross-Path Attention ─────────────────────────────────
    fusion_y = 5.0
    draw_box(ax, 4, fusion_y, 12, 1.5,
             'Cross-Path Multi-Head Self-Attention',
             'Stack 4 branch embeddings → (B, 4, 48)\n'
             'Q, K, V projections → Attention(Q,K,V) = softmax(QKᵀ/√d) × V\n'
             'Residual + LayerNorm → Mean Pool → (B, 48)',
             colors['fusion'], fontsize=12)

    # Arrows from embeddings to fusion
    for bx in [2.2, 6.9, 11.6, 16.3]:
        draw_arrow(ax, bx, embed_y, 10, fusion_y + 1.5, colors['arrow'])

    # ── Classifier Head ──────────────────────────────────────────────
    cls_y = 3.0
    draw_box(ax, 6, cls_y, 8, 1.0,
             'Classifier Head',
             'Linear(48→96) → BN → GELU → Dropout(0.4) → Linear(96→2)',
             colors['classifier'], fontsize=11)
    draw_arrow(ax, 10, fusion_y, 10, cls_y + 1.0, colors['arrow'])

    # ── Output ───────────────────────────────────────────────────────
    out_y = 1.5
    draw_box(ax, 7.5, out_y, 5, 0.8,
             'Output', 'logits (B, 2) + attn_weights (B, 4, 4)',
             colors['output'], fontsize=11)
    draw_arrow(ax, 10, cls_y, 10, out_y + 0.8, colors['arrow'])

    # ── Input shape annotations ──────────────────────────────────────
    input_shapes = [
        ('x_1d\n(B,1,C,T)', 0.3),
        ('x_scalo\n(B,C,F,T)', 5.0),
        ('x_adj\n(B,C,C)', 9.7),
        ('x_bio\n(B,n_bio)', 14.4),
    ]
    for label, bx in input_shapes:
        ax.text(bx + branch_w/2, branch_y + branch_h + 0.6, label,
                ha='center', va='center', fontsize=8, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1',
                          edgecolor='#bdc3c7', alpha=0.9))

    # ── Legend ────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=colors['eegnet'], label='Path A: EEGNet (Temporal)'),
        mpatches.Patch(facecolor=colors['cvit'], label='Path B: CViT (Time-Frequency)'),
        mpatches.Patch(facecolor=colors['gnn'], label='Path C: GNN (Spatial/Graph)'),
        mpatches.Patch(facecolor=colors['bio'], label='Path D: BiomarkerNet (Domain)'),
        mpatches.Patch(facecolor=colors['fusion'], label='Cross-Path Attention Fusion'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_architecture.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[+] Architecture diagram saved to {save_path}")


def plot_attention_heatmap(model, X_1d, X_scalo, X_adj, X_bio, y,
                           save_dir, device='cpu', n_samples=50):
    """
    Visualize cross-path attention weights as a heatmap.

    Shows how each branch attends to every other branch,
    averaged across samples and split by class (AD vs Normal).
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model = model.to(device)

    branch_names = ['EEGNet', 'CViT', 'GNN', 'BiomarkerNet']

    # Collect attention weights per class
    attn_by_class = {0: [], 1: []}

    with torch.no_grad():
        for cls in [0, 1]:
            idx = np.where(y == cls)[0][:n_samples]
            for i in idx:
                x1 = torch.tensor(X_1d[i:i+1]).to(device)
                xs = torch.tensor(X_scalo[i:i+1]).to(device)
                xa = torch.tensor(X_adj[i:i+1]).to(device)
                xb = torch.tensor(X_bio[i:i+1]).to(device) if X_bio is not None else None

                _, attn = model(x1, xs, xa, xb)
                attn_by_class[cls].append(attn.cpu().numpy()[0])

    # Average attention weights
    mean_attn = {}
    for cls in [0, 1]:
        if attn_by_class[cls]:
            mean_attn[cls] = np.mean(attn_by_class[cls], axis=0)

    if not mean_attn:
        print("[!] No attention weights collected")
        return

    # Determine number of branches from attention shape
    n_branches = mean_attn[0].shape[0] if 0 in mean_attn else mean_attn[1].shape[0]
    branch_labels = branch_names[:n_branches]

    # ── Plot: 1×3 grid (Normal, AD, Difference) ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_labels = {0: 'Normal', 1: 'AD'}
    cmaps = ['Blues', 'Reds']

    for i, cls in enumerate([0, 1]):
        if cls not in mean_attn:
            continue
        ax = axes[i]
        im = ax.imshow(mean_attn[cls], cmap=cmaps[i], vmin=0,
                        vmax=max(mean_attn[cls].max(), 0.01),
                        aspect='equal')
        ax.set_xticks(range(n_branches))
        ax.set_xticklabels(branch_labels, fontsize=9, rotation=45, ha='right')
        ax.set_yticks(range(n_branches))
        ax.set_yticklabels(branch_labels, fontsize=9)
        ax.set_xlabel('Key (attended to)', fontsize=10)
        ax.set_ylabel('Query (attending from)', fontsize=10)
        ax.set_title(f'{class_labels[cls]} — Mean Attention',
                     fontsize=12, fontweight='bold')

        # Annotate cells
        for r in range(n_branches):
            for c in range(n_branches):
                val = mean_attn[cls][r, c]
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if val > mean_attn[cls].max() * 0.6 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    # Difference map (AD - Normal)
    if 0 in mean_attn and 1 in mean_attn:
        ax = axes[2]
        diff = mean_attn[1] - mean_attn[0]
        vmax = np.abs(diff).max()
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_xticks(range(n_branches))
        ax.set_xticklabels(branch_labels, fontsize=9, rotation=45, ha='right')
        ax.set_yticks(range(n_branches))
        ax.set_yticklabels(branch_labels, fontsize=9)
        ax.set_xlabel('Key', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        ax.set_title('ΔAttention (AD − Normal)', fontsize=12, fontweight='bold')

        for r in range(n_branches):
            for c in range(n_branches):
                val = diff[r, c]
                ax.text(c, r, f'{val:+.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(val) > vmax * 0.6 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Cross-Path Attention Weights — Branch Interaction Patterns\n'
                 'How each branch attends to others during classification',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'attention_heatmap.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Attention heatmap saved to {save_path}")


def plot_layer_parameter_heatmap(model, save_dir):
    """
    Heatmap showing parameter count distribution across model layers.
    Helps identify where model capacity is concentrated.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Collect layer info
    layer_names = []
    layer_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_names.append(name)
            layer_params.append(param.numel())

    # Group by component
    components = {
        'EEGNet': [], 'CViT': [], 'GNN': [],
        'BiomarkerNet': [], 'CrossAttention': [], 'Classifier': [], 'Other': []
    }
    comp_params = {k: 0 for k in components}

    for name, params in zip(layer_names, layer_params):
        if 'eegnet' in name:
            components['EEGNet'].append((name, params))
            comp_params['EEGNet'] += params
        elif 'cvit' in name:
            components['CViT'].append((name, params))
            comp_params['CViT'] += params
        elif 'gnn' in name:
            components['GNN'].append((name, params))
            comp_params['GNN'] += params
        elif 'biomarker' in name:
            components['BiomarkerNet'].append((name, params))
            comp_params['BiomarkerNet'] += params
        elif 'cross_attention' in name or 'attention' in name:
            components['CrossAttention'].append((name, params))
            comp_params['CrossAttention'] += params
        elif 'classifier' in name:
            components['Classifier'].append((name, params))
            comp_params['Classifier'] += params
        else:
            components['Other'].append((name, params))
            comp_params['Other'] += params

    # Remove empty components
    comp_params = {k: v for k, v in comp_params.items() if v > 0}

    total_params = sum(comp_params.values())

    # ── Plot 1: Parameter distribution pie chart ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    comp_names = list(comp_params.keys())
    comp_values = list(comp_params.values())
    comp_pcts = [v / total_params * 100 for v in comp_values]

    pie_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                  '#1abc9c', '#34495e', '#95a5a6']

    wedges, texts, autotexts = ax1.pie(
        comp_values, labels=comp_names, autopct='%1.1f%%',
        colors=pie_colors[:len(comp_names)],
        pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    ax1.set_title(f'Parameter Distribution by Component\n'
                  f'Total: {total_params:,} parameters',
                  fontsize=13, fontweight='bold')

    # ── Plot 2: Horizontal bar chart ─────────────────────────────────
    sorted_comps = sorted(comp_params.items(), key=lambda x: x[1])
    names, values = zip(*sorted_comps)

    y_pos = np.arange(len(names))
    bar_colors = [pie_colors[comp_names.index(n) % len(pie_colors)] for n in names]

    bars = ax2.barh(y_pos, values, color=bar_colors, alpha=0.85,
                    edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        pct = val / total_params * 100
        ax2.text(bar.get_width() + total_params * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:,} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Parameters', fontsize=11)
    ax2.set_title('Parameters per Component', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_parameters_heatmap.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Parameter distribution saved to {save_path}")

    return comp_params


def main():
    print("=" * 70)
    print("  NeuroFusion-AD — Model Architecture Visualization")
    print("=" * 70)

    save_dir = CFG.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ── Create model with representative shapes ──────────────────────
    n_channels = 16
    n_times_1d = 1024
    n_freqs = 22
    n_times_scalo = 256
    n_bio_features = 160

    model = FusionNet(
        in_channels=n_channels,
        n_classes=2,
        n_bio_features=n_bio_features,
        n_freqs=n_freqs,
        n_times_scalo=n_times_scalo,
        n_times_1d=n_times_1d,
        embed_dim=48,
        dropout=0.4,
        use_gnn=True,
    )
    model.eval()

    # Dummy inputs
    B = 4
    X_1d = torch.randn(B, 1, n_channels, n_times_1d)
    X_scalo = torch.randn(B, n_channels, n_freqs, n_times_scalo)
    X_adj = torch.randn(B, n_channels, n_channels)
    X_bio = torch.randn(B, n_bio_features)

    # ── 1. Architecture Diagram ──────────────────────────────────────
    print("\n[1/4] Generating architecture diagram ...")
    plot_architecture_diagram(save_dir)

    # ── 2. Model Summary (torchinfo) ─────────────────────────────────
    print("\n[2/4] Generating model summary ...")
    plot_model_summary(model, X_1d, X_scalo, X_adj, X_bio, save_dir)

    # ── 3. Parameter Distribution Heatmap ────────────────────────────
    print("\n[3/4] Generating parameter distribution ...")
    plot_layer_parameter_heatmap(model, save_dir)

    # ── 4. Attention Heatmap (with dummy data) ───────────────────────
    print("\n[4/4] Generating attention heatmap ...")
    # Create dummy labels for visualization
    y_dummy = np.array([0, 0, 1, 1])
    X_1d_np = X_1d.numpy()
    X_scalo_np = X_scalo.numpy()
    X_adj_np = X_adj.numpy()
    X_bio_np = X_bio.numpy()

    plot_attention_heatmap(
        model, X_1d_np, X_scalo_np, X_adj_np, X_bio_np, y_dummy,
        save_dir, device='cpu', n_samples=2
    )

    print(f"\n{'=' * 70}")
    print(f"  All visualizations saved to: {save_dir}")
    print(f"  Files generated:")
    print(f"    - model_architecture.png   (architecture diagram)")
    print(f"    - model_summary.txt        (layer-by-layer summary)")
    print(f"    - model_parameters_heatmap.png (parameter distribution)")
    print(f"    - attention_heatmap.png    (cross-path attention)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
