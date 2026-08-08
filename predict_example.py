import os
import numpy as np
import torch
import torch.nn as nn
import config as CFG
from data.loader import load_dataset
from data.scalogram import batch_compute_scalograms
from features.extractor import extract_all_biomarkers
from models.fusion_net import FusionNet

def predict_on_new_examples():
    print("=" * 60)
    print("  NeuroFusion-AD: Run Inference on New Examples")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Generate/Load a few new subject records (synthetic for test)
    print("\n[1/4] Loading new subject examples (synthetic)...")
    records = load_dataset(data_dir=CFG.DATA_DIR, use_synthetic=True, n_synthetic=3)
    
    # Extract epochs
    X_parts = []
    y_true = []
    for rec in records:
        data = rec["epochs"].get_data().astype(np.float32)
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True) + 1e-8
        data = (data - mean) / std
        X_parts.append(data)
        y_true.extend([rec["label"]] * len(data))
        del rec["epochs"]
        
    X_1d = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_true = np.array(y_true, dtype=np.int64)
    # Take a small subset of 5 epochs for rapid inference demonstration
    X_1d = X_1d[:5]
    y_true = y_true[:5]
    print(f"Loaded {len(X_1d)} epochs from {len(records)} test subjects.")
    
    # 2. Extract multimodal features
    print("\n[2/4] Preprocessing & extracting features...")
    freqs = np.arange(2, 45, 2)
    print("  Computing scalograms...")
    X_scalo = batch_compute_scalograms(X_1d, sfreq=CFG.SFREQ, freqs=freqs)
    
    print("  Extracting domain biomarkers & PLI...")
    mean_path = os.path.join(CFG.MODEL_DIR, "X_bio_mean.npy")
    std_path = os.path.join(CFG.MODEL_DIR, "X_bio_std.npy")
    mean = np.load(mean_path) if os.path.exists(mean_path) else None
    std = np.load(std_path) if os.path.exists(std_path) else None
    if mean is not None and std is not None:
        print("  Found saved biomarker scaler. Applying calibration...")
    else:
        print("  [WARNING] No biomarker scaler found. Normalizing on current batch (may degrade prediction accuracy).")
    
    X_bio, X_pli, _ = extract_all_biomarkers(X_1d, sfreq=CFG.SFREQ, mean=mean, std=std)
    X_adj = X_pli
    
    # Check for NaNs
    np.nan_to_num(X_1d, copy=False)
    np.nan_to_num(X_scalo, copy=False)
    np.nan_to_num(X_adj, copy=False)
    np.nan_to_num(X_bio, copy=False)
    
    # 3. Instantiate model & Load weights
    print("\n[3/4] Initializing FusionNet & loading weights...")
    n_bio_features = X_bio.shape[1]
    
    model = FusionNet(
        in_channels=X_1d.shape[1],
        n_classes=2,
        n_bio_features=n_bio_features,
        n_freqs=X_scalo.shape[2],
        n_times_scalo=X_scalo.shape[3],
        n_times_1d=X_1d.shape[2],
        embed_dim=32,
        dropout=0.4,
        use_gnn=True,
        disabled_branches=[]
    )
    
    model_path = os.path.join(CFG.MODEL_DIR, 'neurofusion_v3_final.pth')
    if os.path.exists(model_path):
        print(f"  Found trained model weight file: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("  Model weights loaded successfully.")
    else:
        print(f"  [WARNING] Checkpoint not found at: {model_path}")
        print("  Using random model weights for demonstration purposes.")
        
    model = model.to(device)
    model.eval()
    
    # Prepare PyTorch Tensors
    t_1d = torch.tensor(X_1d).unsqueeze(1).to(device)
    t_scalo = torch.tensor(X_scalo).to(device)
    t_adj = torch.tensor(X_adj).to(device)
    t_bio = torch.tensor(X_bio).to(device)
    
    # 4. Predict
    print("\n[4/4] Running inference...")
    with torch.no_grad():
        logits, attn_weights = model(t_1d, t_scalo, t_adj, t_bio)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
    print("\nInference Results:")
    print("-" * 65)
    print(f"{'Epoch ID':<10} | {'True Label':<12} | {'Predicted':<12} | {'Confidence':<10}")
    print("-" * 65)
    
    correct = 0
    for idx in range(min(15, len(preds))):
        true_name = CFG.CLASS_NAMES[y_true[idx]]
        pred_name = CFG.CLASS_NAMES[preds[idx].item()]
        conf = probs[idx, preds[idx]].item() * 100
        is_correct = "✓" if preds[idx].item() == y_true[idx] else "✗"
        if preds[idx].item() == y_true[idx]:
            correct += 1
        print(f"Epoch {idx:<4}     | {true_name:<12} | {pred_name:<12} {is_correct} | {conf:.1f}%")
        
    acc = correct / min(15, len(preds)) * 100
    print("-" * 65)
    print(f"Batch prediction accuracy: {acc:.1f}%")
    print("=" * 60)

if __name__ == '__main__':
    predict_on_new_examples()
