"""
v3/trainer_v3.py
═══════════════════════════════════════════════════════════════════════════
Multi-model training engine for the NeuroFusion-AD pipeline V3.

Models trained and compared:
  1. FusionNet — 4-branch hybrid (EEGNet + CViT + GNN + BiomarkerNet)
  2. SVM — Statistical baseline
  3. Random Forest — Statistical baseline
  4. Gradient Boosting — Statistical baseline
  5. XGBoost — Statistical baseline

Training features:
  - StratifiedGroupKFold 3-fold CV (subject-level, no data leakage)
  - Mean ± std reporting across folds
  - CosineAnnealing + SWA scheduler
  - Label smoothing + Mixup augmentation
  - Gradient clipping + early stopping by AUC
  - Classical models saved to disk (joblib)
  - Cross-path attention fusion
═══════════════════════════════════════════════════════════════════════════
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import recall_score, roc_auc_score

import config as CFG
from models.fusion_net import FusionNet
from data.augment import augment_batch
from visualize import plot_training_curves, plot_fold_metrics


# ═════════════════════════════════════════════════════════════════════════
# SEED UTILITY
# ═════════════════════════════════════════════════════════════════════════
def _reset_seed(seed):
    """Reset all random seeds for reproducibility within each fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════
# FUSION MODEL TRAINING (4-branch)
# ═════════════════════════════════════════════════════════════════════════
def _train_epoch_fusion(model, loader, optimizer, criterion, device, has_bio=False):
    """Train one epoch of the 4-branch FusionNet with augmentation."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

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

        # Online augmentation (only modifies x_1d)
        x_1d, x_scalo, x_adj, x_bio = augment_batch(
            x_1d, x_scalo, x_adj, x_bio
        )

        optimizer.zero_grad()
        out, _ = model(x_1d, x_scalo, x_adj, x_bio)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def _eval_epoch_fusion(model, loader, criterion, device, has_bio=False):
    """Evaluate the FusionNet on validation data."""
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


# ═════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════
# MIXUP AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════
def _mixup_batch(x_1d, x_scalo, x_adj, x_bio, y, alpha=0.2):
    """
    Mixup augmentation: blend pairs of samples to smooth decision boundaries.
    x' = λ·x_i + (1-λ)·x_j,  y' = λ·y_i + (1-λ)·y_j
    (Zhang et al., 2018 — proven to reduce overfitting and stabilize val loss)
    """
    if alpha <= 0:
        return x_1d, x_scalo, x_adj, x_bio, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 so original dominates

    B = x_1d.size(0)
    idx = torch.randperm(B, device=x_1d.device)

    x_1d_mix = lam * x_1d + (1 - lam) * x_1d[idx]
    x_scalo_mix = lam * x_scalo + (1 - lam) * x_scalo[idx]
    x_adj_mix = lam * x_adj + (1 - lam) * x_adj[idx]
    x_bio_mix = None
    if x_bio is not None:
        x_bio_mix = lam * x_bio + (1 - lam) * x_bio[idx]

    y_a, y_b = y, y[idx]
    return x_1d_mix, x_scalo_mix, x_adj_mix, x_bio_mix, y_a, y_b, lam


def _train_epoch_fusion_mixup(model, loader, optimizer, criterion, device, has_bio=False):
    """Train one epoch with Mixup + standard augmentation."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

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

        # Standard augmentation first
        x_1d, x_scalo, x_adj, x_bio = augment_batch(
            x_1d, x_scalo, x_adj, x_bio
        )

        # Mixup
        x_1d, x_scalo, x_adj, x_bio, y_a, y_b, lam = _mixup_batch(
            x_1d, x_scalo, x_adj, x_bio, y, alpha=0.4
        )

        optimizer.zero_grad()
        out, _ = model(x_1d, x_scalo, x_adj, x_bio)

        # Mixup loss: weighted combination of losses for both targets
        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        _, preds = torch.max(out, 1)
        correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# ═════════════════════════════════════════════════════════════════════════
# MAIN TRAINING — 3-Fold Stratified Group CV
# ═════════════════════════════════════════════════════════════════════════
def cross_validate_all(X_1d, X_scalo, X_adj, y, groups,
                       X_bio=None, n_splits=3, epochs=60, batch_size=32, disable_gnn=False):
    """
    Train FusionNet with 3-fold subject-level stratified cross-validation.

    Reports mean ± std across folds for robust evaluation.
    Classical baselines are trained on the same fold splits and saved to disk.

    Parameters
    ----------
    X_1d, X_scalo, X_adj, y, groups, X_bio : arrays
    n_splits : int — number of CV folds (default 3)
    epochs : int
    batch_size : int
    disable_gnn : bool — disable GNN branch based on ablation finding

    Returns
    -------
    avg_results, all_fold_metrics, best_fusion_model, val_y_all, val_p_all
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if disable_gnn:
        print("  [Note] GNN branch is DISABLED for FusionNet training.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED)

    model_names = ['FusionNet', 'BiLSTM']
    if X_bio is not None:
        model_names.extend(['SVM', 'Random Forest', 'Gradient Boosting', 'XGBoost'])
    all_fold_metrics = {name: [] for name in model_names}

    PATIENCE = 25
    has_bio = X_bio is not None
    best_fusion_model = None
    best_fusion_auc = 0

    # Collect val predictions from best fold for confusion matrix
    best_fold_val_y = None
    best_fold_val_p = None

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_1d, y, groups)):
        fold_num = fold_idx + 1
        print(f"\n{'='*65}")
        print(f"  FOLD {fold_num}/{n_splits}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
        print(f"  Train AD: {(y[train_idx]==1).sum()}  Normal: {(y[train_idx]==0).sum()}")
        print(f"  Val   AD: {(y[val_idx]==1).sum()}  Normal: {(y[val_idx]==0).sum()}")
        print(f"{'='*65}")

        # Reset seed per fold for reproducibility
        _reset_seed(CFG.SEED + fold_idx)

        # ─── FusionNet ────────────────────────────────────────────
        print(f"\n  [FusionNet] Training Fold {fold_num}...")
        fusion_metrics, fusion_model, val_y, val_p = _train_fusion(
            X_1d, X_scalo, X_adj, X_bio,
            train_idx, val_idx, y, device,
            epochs=epochs, batch_size=batch_size, patience=PATIENCE,
            fold_num=fold_num, disable_gnn=disable_gnn
        )
        all_fold_metrics['FusionNet'].append(fusion_metrics)
        print(f"  [FusionNet] Fold {fold_num}: Acc={fusion_metrics['acc']:.4f} AUC={fusion_metrics['auc']:.4f}")

        # Track best fold model
        if fusion_metrics['auc'] > best_fusion_auc:
            best_fusion_auc = fusion_metrics['auc']
            best_fusion_model = fusion_model
            best_fold_val_y = val_y
            best_fold_val_p = val_p

        # ─── BiLSTM ───────────────────────────────────────────────
        print(f"\n  [BiLSTM] Training Fold {fold_num}...")
        bilstm_metrics, bilstm_model = _train_bilstm(
            X_1d, train_idx, val_idx, y, device,
            epochs=epochs, batch_size=batch_size, patience=PATIENCE,
            fold_num=fold_num
        )
        all_fold_metrics['BiLSTM'].append(bilstm_metrics)
        print(f"  [BiLSTM] Fold {fold_num}: Acc={bilstm_metrics['acc']:.4f} AUC={bilstm_metrics['auc']:.4f}")

        # ─── Classical Baselines ──────────────────────────────────
        if X_bio is not None:
            _train_classical_baselines(
                X_bio, y, train_idx, val_idx,
                all_fold_metrics, fold_num
            )

    # ── Aggregate Results (Mean ± Std) ────────────────────────────────
    avg_results = {}
    print(f"\n\n{'='*75}")
    print(f"  FINAL RESULTS — {n_splits}-Fold Cross-Validation (Mean ± Std)")
    print(f"{'='*75}")
    print(f"  {'Model':<20} {'Acc':>12} {'Sens':>12} {'Spec':>12} {'AUC':>12}")
    print("-" * 75)

    for name in model_names:
        folds = all_fold_metrics[name]
        mean_metrics = {}
        std_metrics = {}
        for metric in ['acc', 'sens', 'spec', 'auc']:
            vals = [f[metric] for f in folds]
            mean_metrics[metric] = np.mean(vals)
            std_metrics[metric] = np.std(vals)

        avg_results[name] = mean_metrics
        avg_results[name]['std'] = std_metrics

        print(f"  {name:<20} "
              f"{mean_metrics['acc']:.4f}±{std_metrics['acc']:.3f} "
              f"{mean_metrics['sens']:.4f}±{std_metrics['sens']:.3f} "
              f"{mean_metrics['spec']:.4f}±{std_metrics['spec']:.3f} "
              f"{mean_metrics['auc']:.4f}±{std_metrics['auc']:.3f}")

    print("=" * 75)

    return avg_results, all_fold_metrics, best_fusion_model, best_fold_val_y, best_fold_val_p


# ═════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES (with model saving)
# ═════════════════════════════════════════════════════════════════════════
def _train_classical_baselines(X_bio, y, train_idx, val_idx,
                               all_fold_metrics, fold_num):
    """Train and save classical ML baselines (SVM, RF, GB, XGBoost)."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC

    X_tr = X_bio[train_idx]
    X_val = X_bio[val_idx]
    y_tr = y[train_idx]
    y_val = y[val_idx]

    save_dir = CFG.MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    if len(np.unique(y_tr)) < 2:
        print("  [WARNING] Only one class in training split. Skipping baselines.")
        dummy = {'acc': 0.5, 'sens': 0.0, 'spec': 1.0, 'auc': 0.5}
        for name in ['SVM', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
            all_fold_metrics[name].append(dummy)
        return

    # 1. SVM
    print(f"  [SVM] Fitting RBF SVM (Fold {fold_num})...")
    svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=CFG.SEED)
    svm.fit(X_tr, y_tr)
    svm_preds = svm.predict(X_val)
    svm_probs = svm.predict_proba(X_val)[:, 1]
    svm_metrics = _compute_metrics(y_val, svm_preds, svm_probs)
    all_fold_metrics['SVM'].append(svm_metrics)
    print(f"  [SVM] Fold {fold_num}: Acc={svm_metrics['acc']:.4f} AUC={svm_metrics['auc']:.4f}")

    # Save best fold model (fold 1 always saved, then overwrite if better)
    if fold_num == 1 or svm_metrics['auc'] > max(m['auc'] for m in all_fold_metrics['SVM'][:-1]):
        joblib.dump(svm, os.path.join(save_dir, 'svm_model.joblib'))

    # 2. Random Forest
    print(f"  [Random Forest] Fitting 100-tree RF (Fold {fold_num})...")
    rf = RandomForestClassifier(n_estimators=100, random_state=CFG.SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_preds = rf.predict(X_val)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_metrics = _compute_metrics(y_val, rf_preds, rf_probs)
    all_fold_metrics['Random Forest'].append(rf_metrics)
    print(f"  [Random Forest] Fold {fold_num}: Acc={rf_metrics['acc']:.4f} AUC={rf_metrics['auc']:.4f}")

    if fold_num == 1 or rf_metrics['auc'] > max(m['auc'] for m in all_fold_metrics['Random Forest'][:-1]):
        joblib.dump(rf, os.path.join(save_dir, 'random_forest_model.joblib'))

    # 3. Gradient Boosting
    print(f"  [Gradient Boosting] Fitting HistGradientBoosting (Fold {fold_num})...")
    gb = HistGradientBoostingClassifier(random_state=CFG.SEED)
    gb.fit(X_tr, y_tr)
    gb_preds = gb.predict(X_val)
    gb_probs = gb.predict_proba(X_val)[:, 1]
    gb_metrics = _compute_metrics(y_val, gb_preds, gb_probs)
    all_fold_metrics['Gradient Boosting'].append(gb_metrics)
    print(f"  [Gradient Boosting] Fold {fold_num}: Acc={gb_metrics['acc']:.4f} AUC={gb_metrics['auc']:.4f}")

    if fold_num == 1 or gb_metrics['auc'] > max(m['auc'] for m in all_fold_metrics['Gradient Boosting'][:-1]):
        joblib.dump(gb, os.path.join(save_dir, 'gradient_boosting_model.joblib'))

    # 4. XGBoost
    print(f"  [XGBoost] Fitting XGBoost (Fold {fold_num})...")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=CFG.SEED,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        xgb_preds = xgb.predict(X_val)
        xgb_probs = xgb.predict_proba(X_val)[:, 1]
        xgb_metrics = _compute_metrics(y_val, xgb_preds, xgb_probs)
        all_fold_metrics['XGBoost'].append(xgb_metrics)
        print(f"  [XGBoost] Fold {fold_num}: Acc={xgb_metrics['acc']:.4f} AUC={xgb_metrics['auc']:.4f}")

        if fold_num == 1 or xgb_metrics['auc'] > max(m['auc'] for m in all_fold_metrics['XGBoost'][:-1]):
            joblib.dump(xgb, os.path.join(save_dir, 'xgboost_model.joblib'))
    except ImportError:
        print("  [XGBoost] xgboost not installed. Using dummy metrics.")
        all_fold_metrics['XGBoost'].append({'acc': 0.5, 'sens': 0.0, 'spec': 1.0, 'auc': 0.5})


# ═════════════════════════════════════════════════════════════════════════
# INTERNAL: Train single fold of FusionNet
# ═════════════════════════════════════════════════════════════════════════
def _train_fusion(X_1d, X_scalo, X_adj, X_bio,
                  train_idx, val_idx, y, device,
                  epochs, batch_size, patience, fold_num=1, disable_gnn=False):
    """
    Train the 4-branch FusionNet with:
    1. Mixup augmentation — smooths decision boundaries
    2. Stochastic Weight Averaging (SWA) — finds flat minima
    3. EMA-smoothed val loss tracking — stable early stopping
    """
    from torch.optim.swa_utils import AveragedModel, SWALR

    has_bio = X_bio is not None

    # Prepare tensors
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

    model = FusionNet(
        in_channels=X_1d.shape[1],
        n_classes=2,
        n_bio_features=n_bio_features,
        n_freqs=X_scalo.shape[2],
        n_times_scalo=X_scalo.shape[3],
        n_times_1d=X_1d.shape[2],
        embed_dim=32,
        dropout=0.5,      # Raised from 0.4 to prevent overfitting
        use_gnn=not disable_gnn,
        disabled_branches=['gnn'] if disable_gnn else []
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.15)

    # Phase 1: CosineAnnealing for first 75% of epochs
    swa_start = int(epochs * 0.75)
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-6)

    # Phase 2: SWA averages weights over last 25%
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)

    best_auc = 0
    best_metrics = {'acc': 0, 'sens': 0, 'spec': 0, 'auc': 0}
    best_model_state = None
    patience_counter = 0

    # EMA-smoothed validation loss for stable early stopping
    ema_val_loss = None
    ema_alpha = 0.3

    # Training history for visualization
    history = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}

    for ep in range(epochs):
        # Train with Mixup
        tr_loss, tr_acc = _train_epoch_fusion_mixup(
            model, train_loader, optimizer, criterion, device, has_bio
        )

        # Evaluate
        val_loss, val_acc, val_y, val_p, val_prob = _eval_epoch_fusion(
            model, val_loader, criterion, device, has_bio
        )

        # Scheduler
        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # EMA-smoothed validation loss
        if ema_val_loss is None:
            ema_val_loss = val_loss
        else:
            ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss

        # Record history
        history['tr_loss'].append(tr_loss)
        history['val_loss'].append(ema_val_loss)
        history['tr_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        current_auc = _safe_auc(val_y, val_prob)

        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            best_metrics = _compute_metrics(val_y, val_p, val_prob)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0:
            phase = "SWA" if ep >= swa_start else "STD"
            print(f"    Ep {ep+1:02d} [{phase}] | Tr {tr_loss:.3f}/{tr_acc:.3f} | Val {val_loss:.3f}/{val_acc:.3f} | EMA {ema_val_loss:.3f} | AUC {current_auc:.3f}")

        # Don't early-stop during SWA phase
        if patience_counter >= patience and ep < swa_start:
            print(f"    Early stop at ep {ep+1}")
            break

    # ── Finalize SWA model ───────────────────────────────────────────
    if ep >= swa_start:
        print(f"    SWA: Averaging weights from epochs {swa_start+1}-{ep+1}")
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        except Exception:
            pass

        swa_model = swa_model.to(device)
        swa_loss, swa_acc, swa_y, swa_p, swa_prob = _eval_epoch_fusion(
            swa_model, val_loader, criterion, device, has_bio
        )
        swa_auc = _safe_auc(swa_y, swa_prob)
        print(f"    SWA Final | Val {swa_loss:.3f}/{swa_acc:.3f} | AUC {swa_auc:.3f}")

        if swa_auc >= best_auc:
            best_metrics = _compute_metrics(swa_y, swa_p, swa_prob)
            best_model_state = {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
            print(f"    SWA model selected (AUC {swa_auc:.3f} >= {best_auc:.3f})")

    # Plot training curves
    plot_training_curves(history, fold_num, CFG.PLOT_DIR)

    # Restore best model
    model_out = FusionNet(
        in_channels=X_1d.shape[1], n_classes=2, n_bio_features=n_bio_features,
        n_freqs=X_scalo.shape[2], n_times_scalo=X_scalo.shape[3],
        n_times_1d=X_1d.shape[2], embed_dim=32, dropout=0.5,
        use_gnn=not disable_gnn, disabled_branches=['gnn'] if disable_gnn else []
    ).to(device)
    if best_model_state:
        clean_state = {k.replace('module.', ''): v for k, v in best_model_state.items()}
        try:
            model_out.load_state_dict(clean_state)
        except Exception:
            model_out.load_state_dict(best_model_state, strict=False)
        model_out = model_out.to(device)

    # Final evaluation
    model_out.eval()
    _, _, val_y, val_p, _ = _eval_epoch_fusion(
        model_out, val_loader, criterion, device, has_bio
    )

    return best_metrics, model_out, val_y, val_p


# ═════════════════════════════════════════════════════════════════════════
# INTERNAL: Train single fold of BiLSTM
# ═════════════════════════════════════════════════════════════════════════
def _train_bilstm(X_1d, train_idx, val_idx, y, device,
                  epochs, batch_size, patience=25, fold_num=1):
    """
    Train BiLSTM model on raw 1D EEG signals, recording training curves.
    """
    from models.bilstm import BiLSTM
    from torch.utils.data import TensorDataset, DataLoader
    from visualize import plot_training_curves

    X1_tr = torch.tensor(X_1d[train_idx]).unsqueeze(1)
    X1_val = torch.tensor(X_1d[val_idx]).unsqueeze(1)
    y_tr = torch.tensor(y[train_idx])
    y_val = torch.tensor(y[val_idx])

    train_ds = TensorDataset(X1_tr, y_tr)
    val_ds = TensorDataset(X1_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BiLSTM(in_channels=X_1d.shape[1], hidden_size=64, num_layers=2, n_classes=2, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = 0
    best_metrics = {'acc': 0, 'sens': 0, 'spec': 0, 'auc': 0}
    best_model_state = None
    patience_counter = 0

    history = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}

    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * y_b.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        tr_loss = total_loss / max(1, total)
        tr_acc = correct / max(1, total)
        scheduler.step()

        # Evaluate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                out = model(x_b)
                loss = criterion(out, y_b)
                val_loss += loss.item() * y_b.size(0)
                probs = torch.softmax(out, dim=1)[:, 1]
                _, preds = torch.max(out, 1)
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)
                all_labels.extend(y_b.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        v_loss = val_loss / max(1, val_total)
        v_acc = val_correct / max(1, val_total)
        v_y = np.array(all_labels)
        v_p = np.array(all_preds)
        v_prob = np.array(all_probs)
        current_auc = _safe_auc(v_y, v_prob)

        history['tr_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['tr_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)

        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            best_metrics = _compute_metrics(v_y, v_p, v_prob)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0:
            print(f"    Ep {ep+1:02d} [BiLSTM] | Tr {tr_loss:.3f}/{tr_acc:.3f} | Val {v_loss:.3f}/{v_acc:.3f} | AUC {current_auc:.3f}")

        if patience_counter >= patience:
            print(f"    Early stop at ep {ep+1}")
            break

    # Save training curves plot for BiLSTM
    plot_training_curves(
        history, 
        fold_num=fold_num, 
        plot_dir=CFG.PLOT_DIR, 
        filename=f"training_curves_bilstm_fold{fold_num}.png",
        title_prefix="BiLSTM "
    )

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_metrics, model

