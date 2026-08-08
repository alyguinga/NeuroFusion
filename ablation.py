"""
v3/ablation.py
═══════════════════════════════════════════════════════════════════════════
Branch ablation study for the NeuroFusion-AD pipeline.

Trains and evaluates 4 FusionNet variants to measure each branch's
contribution to the overall performance:

  1. Full FusionNet     — All 4 branches active (baseline)
  2. No BiomarkerNet    — EEGNet + CViT + GNN (does domain knowledge help?)
  3. BiomarkerNet Only  — Only domain features (can they stand alone?)
  4. No GNN             — EEGNet + CViT + BiomarkerNet (does connectivity help?)

Implementation uses zero-masking via FusionNet's `disabled_branches` parameter,
keeping the architecture and parameter count identical across variants for
a fair comparison.
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import recall_score, roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as CFG
from models.fusion_net import FusionNet
from data.augment import augment_batch


# ═════════════════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ═════════════════════════════════════════════════════════════════════════
ABLATION_VARIANTS = {
    'Full FusionNet':     [],                              # all branches active
    'No BiomarkerNet':    ['biomarker'],                   # remove domain knowledge
    'BiomarkerNet Only':  ['eegnet', 'cvit', 'gnn'],      # domain features alone
    'No GNN':             ['gnn'],                         # remove connectivity
}


def _safe_auc(y_true, y_prob):
    if np.isnan(y_prob).any() or len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def _compute_metrics(y_true, y_pred, y_prob):
    return {
        'acc': float(np.mean(y_pred == y_true)),
        'sens': recall_score(y_true, y_pred, zero_division=0),
        'spec': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'auc': _safe_auc(y_true, y_prob)
    }


def _eval_model(model, loader, criterion, device, has_bio):
    """Evaluate a model on a DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if has_bio:
                x_1d, x_scalo, x_adj, x_bio, y = batch
                x_bio = x_bio.to(device)
            else:
                x_1d, x_scalo, x_adj, y = batch
                x_bio = None

            x_1d = x_1d.to(device)
            x_scalo = x_scalo.to(device)
            x_adj = x_adj.to(device)
            y = y.to(device)

            out, _ = model(x_1d, x_scalo, x_adj, x_bio)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            probs = torch.softmax(out, dim=1)[:, 1]
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def _train_variant(variant_name, disabled_branches,
                   X_1d, X_scalo, X_adj, X_bio,
                   train_idx, val_idx, y, device,
                   epochs=40, batch_size=64):
    """
    Train a single ablation variant.

    Uses a shorter training schedule (40 epochs, no SWA) than the main
    training loop since we're comparing relative performance, not
    maximizing absolute accuracy.
    """
    has_bio = X_bio is not None

    X1_tr = torch.tensor(X_1d[train_idx]).unsqueeze(1)
    X1_val = torch.tensor(X_1d[val_idx]).unsqueeze(1)
    Xs_tr = torch.tensor(X_scalo[train_idx])
    Xs_val = torch.tensor(X_scalo[val_idx])
    Xa_tr = torch.tensor(X_adj[train_idx])
    Xa_val = torch.tensor(X_adj[val_idx])
    y_tr = torch.tensor(y[train_idx])
    y_val = torch.tensor(y[val_idx])

    if has_bio:
        Xb_tr = torch.tensor(X_bio[train_idx])
        Xb_val = torch.tensor(X_bio[val_idx])
        train_ds = TensorDataset(X1_tr, Xs_tr, Xa_tr, Xb_tr, y_tr)
        val_ds = TensorDataset(X1_val, Xs_val, Xa_val, Xb_val, y_val)
    else:
        train_ds = TensorDataset(X1_tr, Xs_tr, Xa_tr, y_tr)
        val_ds = TensorDataset(X1_val, Xs_val, Xa_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_bio_features = X_bio.shape[1] if has_bio else 0

    import random
    # Reset seed for reproducibility
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    torch.cuda.manual_seed(CFG.SEED)
    torch.cuda.manual_seed_all(CFG.SEED)

    model = FusionNet(
        in_channels=X_1d.shape[1],
        n_classes=2,
        n_bio_features=n_bio_features,
        n_freqs=X_scalo.shape[2],
        n_times_scalo=X_scalo.shape[3],
        n_times_1d=X_1d.shape[2],
        embed_dim=32,
        dropout=0.5,      # Raised from 0.4 to prevent overfitting
        use_gnn=True,
        disabled_branches=disabled_branches,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.15)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = 0
    best_metrics = {'acc': 0, 'sens': 0, 'spec': 0, 'auc': 0}
    best_model_state = None
    history = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}

    for ep in range(epochs):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if has_bio:
                x_1d, x_scalo, x_adj, x_bio, yb = batch
                x_bio = x_bio.to(device)
            else:
                x_1d, x_scalo, x_adj, yb = batch
                x_bio = None

            x_1d = x_1d.to(device)
            x_scalo = x_scalo.to(device)
            x_adj = x_adj.to(device)
            yb = yb.to(device)

            x_1d, x_scalo, x_adj, x_bio = augment_batch(
                x_1d, x_scalo, x_adj, x_bio
            )

            optimizer.zero_grad()
            out, _ = model(x_1d, x_scalo, x_adj, x_bio)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        tr_loss = total_loss / max(1, total)
        tr_acc = correct / max(1, total)
        scheduler.step()

        # Evaluate
        val_loss, val_acc, val_y, val_p, val_prob = _eval_model(
            model, val_loader, criterion, device, has_bio
        )
        current_auc = _safe_auc(val_y, val_prob)

        history['tr_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['tr_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        if current_auc > best_auc:
            best_auc = current_auc
            best_metrics = _compute_metrics(val_y, val_p, val_prob)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save model and graphs for No GNN variant
    if variant_name == 'No GNN':
        from visualize import plot_training_curves, plot_confusion_matrix
        plot_training_curves(
            history,
            fold_num="ablation_no_gnn",
            plot_dir=CFG.PLOT_DIR,
            filename="training_curves_no_gnn.png",
            title_prefix="Ablation (No GNN) "
        )
        print(f"    [+] Saved No GNN training curves to: {os.path.join(CFG.PLOT_DIR, 'training_curves_no_gnn.png')}")

        if best_model_state is not None:
            save_path = os.path.join(CFG.MODEL_DIR, "neurofusion_v3_no_gnn.pth")
            torch.save(best_model_state, save_path)
            print(f"    [+] Saved No GNN optimal model to: {save_path}")

            # Evaluate best model state to generate confusion matrix
            model.load_state_dict(best_model_state)
            _, _, b_val_y, b_val_p, _ = _eval_model(model, val_loader, criterion, device, has_bio)
            class_names = [CFG.CLASS_NAMES[c] for c in CFG.CLASSES]
            plot_confusion_matrix(
                b_val_y, b_val_p, class_names, CFG.PLOT_DIR,
                filename="confusion_matrix_no_gnn.png",
                title="Confusion Matrix — Ablation (No GNN)"
            )

    return best_metrics


# ═════════════════════════════════════════════════════════════════════════
# MAIN ABLATION RUNNER
# ═════════════════════════════════════════════════════════════════════════
def run_ablation_study(X_1d, X_scalo, X_adj, y, groups,
                       X_bio=None, epochs=40, batch_size=64):
    """
    Run the full ablation study: train 4 FusionNet variants and compare.

    Parameters
    ----------
    X_1d, X_scalo, X_adj, y, groups, X_bio : same as cross_validate_all
    epochs : int — training epochs per variant (shorter than main training)
    batch_size : int

    Returns
    -------
    results : dict {variant_name: {acc, sens, spec, auc}}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  ABLATION STUDY — Branch Contribution Analysis")
    print(f"{'='*65}")

    # Use same 90/10 split as main training
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = next(iter(sgkf.split(X_1d, y, groups)))

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"  Epochs per variant: {epochs}\n")

    results = {}

    for variant_name, disabled in ABLATION_VARIANTS.items():
        print(f"  Training: {variant_name} (disabled: {disabled or 'none'}) …")
        metrics = _train_variant(
            variant_name, disabled,
            X_1d, X_scalo, X_adj, X_bio,
            train_idx, val_idx, y, device,
            epochs=epochs, batch_size=batch_size
        )
        results[variant_name] = metrics
        print(f"    Acc={metrics['acc']:.4f} Sens={metrics['sens']:.4f} "
              f"Spec={metrics['spec']:.4f} AUC={metrics['auc']:.4f}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ABLATION RESULTS")
    print(f"{'='*65}")
    print(f"  {'Variant':<22} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'AUC':>8}")
    print("-" * 65)

    full_auc = results.get('Full FusionNet', {}).get('auc', 0)
    for name, m in results.items():
        delta = m['auc'] - full_auc
        delta_str = f" ({delta:+.3f})" if name != 'Full FusionNet' else ""
        print(f"  {name:<22} {m['acc']:>8.4f} {m['sens']:>8.4f} "
              f"{m['spec']:>8.4f} {m['auc']:>8.4f}{delta_str}")

    print("=" * 65)

    return results

