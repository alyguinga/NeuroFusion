# NeuroFusion-AD: Hybrid Deep Learning Pipeline for EEG-Based Alzheimer's Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

NeuroFusion-AD is a multi-modal, hybrid deep learning framework engineered for robust 2-class classification of **Alzheimer's Disease (AD)** versus **Healthy Control (Normal)** subjects using resting-state electroencephalography (EEG). The framework fuses multi-domain EEG representations—raw temporal waveforms, 2D time-frequency scalograms, functional connectivity graphs, and tabular neurophysiological biomarkers—into a unified 4-branch neural architecture (**FusionNet**).

---

## 🌟 Overview & Key Features

Alzheimer's Disease leads to progressive cognitive decline accompanied by distinct EEG alterations, including slowing of background rhythms, loss of functional connectivity, and complexity reduction. NeuroFusion-AD captures these multifaceted biomarkers through specialized parallel neural network branches:

1. **EEGNet Branch**: Extracts temporal-spatial features directly from raw multi-channel EEG signals.
2. **Compact Vision Transformer (CViT) Branch**: Processes 2D Continuous Wavelet Transform (CWT) time-frequency scalograms to capture localized time-frequency perturbations.
3. **Graph Neural Network (GNN) Branch**: Models spatial topography and inter-channel functional connectivity matrices (Phase Lag Index / Pearson coherence).
4. **BiomarkerNet Branch**: Integrates expert engineered neurophysiological biomarkers (spectral power ratios, sample entropy, Higuchi fractal dimension).

A cross-attention fusion mechanism dynamically weights information from each modality, providing high classification accuracy alongside comprehensive Explainable AI (XAI) interpretations.

---

## 📊 Dataset Description

The pipeline is designed and benchmarked on the open-access **OpenNeuro ds004504** dataset:
- **Subjects**: Resting-state EEG recordings from Alzheimer's Disease (AD) patients, Frontotemporal Dementia (FTD) patients, and Healthy Controls (CN).
- **Pipeline Target**: 2-Class classification (**AD vs Normal / Control**).
- **Montage**: 19 scalp channels placed according to the international 10-20 system (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz).
- **Condition**: Resting-state, eyes-closed paradigm.
- **Sampling Rate**: Resampled and preprocessed to standard 250 Hz analysis windowing.

---

## 📁 Repository Directory Structure

```
NeuroFusion/
├── README.md                   # Project documentation and usage guide
├── LICENSE                     # MIT License file
├── .gitignore                  # Git exclusion rules for binary files, caches, and environments
├── config.py                   # Global pipeline configurations, hyperparameter settings, and paths
├── requirements.txt            # Python package dependencies
├── run_pipeline.py             # Main entry script for full cross-validation pipeline execution
├── run_tf_comparison.py        # Comparative script evaluating STFT vs CWT vs Morlet representations
├── plot_architecture.py        # Architectural visualization script for the 4-branch FusionNet
├── ablation.py                 # Ablation study module evaluating single-branch vs multi-branch models
├── trainer.py                  # PyTorch training loops, cross-validation routines, and metrics tracker
├── visualize.py                # Plotting utilities for training curves, confusion matrices, and PSD
├── xai.py                      # Explainable AI module (Grad-CAM, SHAP, attention maps)
├── predict_example.py          # Single-subject inference script for deployment demonstration
├── Kaggle_Starter.ipynb        # Ready-to-run Jupyter notebook for Kaggle and Google Colab
│
├── data/                       # Data loading and signal transformation module
│   ├── __init__.py             # Package initializer
│   ├── loader.py               # OpenNeuro ds004504 dataset loader and EEGLAB preprocessor
│   ├── scalogram.py            # CWT scalogram image generator for time-frequency analysis
│   ├── connectivity.py         # Functional connectivity matrix calculator (PLI, Coherence, Pearson)
│   ├── augment.py              # EEG data augmentation (time shifts, noise injection, channel dropout)
│   └── tf_representations.py  # Multi-domain time-frequency feature map transformations
│
├── features/                   # Feature engineering and biomarker extraction module
│   ├── __init__.py             # Package initializer
│   ├── spectral.py             # Spectral band power calculator (delta, theta, alpha, beta, gamma)
│   ├── complexity.py           # Non-linear dynamics metrics (Sample Entropy, Higuchi Fractal Dim)
│   ├── connectivity_features.py # Graph-theoretical network topological features (clustering, path length)
│   └── extractor.py            # Orchestrator aggregating spectral, complexity, and connectivity features
│
├── models/                     # Deep learning architecture definitions
│   ├── __init__.py             # Package initializer
│   ├── cvit.py                 # Compact Vision Transformer branch for 2D CWT scalograms
│   ├── gnn.py                  # Graph Neural Network branch for scalp functional topology
│   ├── fusion_net.py           # Master 4-branch hybrid FusionNet with cross-attention fusion
│   ├── biomarker_net.py        # Multi-Layer Perceptron (BiomarkerNet) for tabular features
│   ├── eegnet.py               # Compact 1D CNN branch for raw temporal EEG waveforms
│   └── bilstm.py               # Bidirectional LSTM baseline network for temporal sequence benchmark
│
├── utils/                      # Helper utilities and visualization styles
│   ├── __init__.py             # Package initializer
│   └── plots.py                # Custom seaborn/matplotlib figure formatting and styling setup
│
├── xai/                        # Pre-generated Explainable AI artifact outputs
│   ├── attention_by_class.png   # Class-wise cross-attention distribution maps
│   ├── domain_contribution.png  # Relative contribution of temporal, spectral, spatial, and tabular domains
│   ├── feature_attribution.png  # Integrated gradients feature attribution across EEG channels
│   ├── gradcam_scalogram.png    # Grad-CAM saliency heatmaps overlaid on time-frequency scalograms
│   ├── cross_path_attention.png # Inter-branch attention weights across parallel sub-networks
│   └── population_attribution.png # Group-level attribution summary across AD and Control cohorts
│
└── examples/                   # Sample publication-ready plots showcasing pipeline performance
    ├── model_architecture.png  # Diagram of the 4-branch hybrid NeuroFusion pipeline
    ├── confusion_matrix.png    # Cross-validation confusion matrix for AD vs Normal classification
    ├── model_comparison.png    # Benchmark comparison of NeuroFusion vs single-branch models
    ├── training_curves_fold1.png # Training loss and validation accuracy curves for Fold 1
    ├── psd_comparison.png      # Power Spectral Density (PSD) comparison between AD and Control
    ├── shap_importance.png     # SHAP summary ranking tabular biomarker importance
    ├── attention_by_class.png  # Class-conditioned attention weights
    ├── domain_contribution.png # Branch contribution pie/bar charts
    ├── feature_attribution.png # Integrated gradients feature importance distribution
    └── gradcam_scalogram.png   # Grad-CAM spatial-frequency activation maps
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model training)

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/NeuroFusion.git
   cd NeuroFusion
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Instructions

### 1. Full Pipeline Execution
To execute the complete pipeline including dataset loading, multi-domain feature extraction, stratified K-fold cross-validation training, and evaluation:
```bash
python run_pipeline.py
```

### 2. Time-Frequency Representation Comparison
To compare model performance across different time-frequency decomposition methods (STFT, CWT, Morlet wavelets):
```bash
python run_tf_comparison.py
```

### 3. Ablation Study
To run ablation experiments evaluating the marginal contribution of each branch (EEGNet, CViT, GNN, BiomarkerNet):
```bash
python ablation.py
```

### 4. Single-Subject Inference Example
To run inference on a single EEG record and output classification probability scores:
```bash
python predict_example.py --input path/to/sample_eeg.set
```

### 5. Explainable AI Analysis
To generate interpretability maps (Grad-CAM, SHAP, and cross-attention distributions):
```bash
python xai.py
```

---

## ☁️ Kaggle & Google Colab Integration

For zero-setup execution on cloud platforms with GPU acceleration:
- Open `Kaggle_Starter.ipynb` in **Google Colab** or upload to **Kaggle Notebooks**.
- The notebook automatically clones required data dependencies, configures PyTorch GPU acceleration, runs 5-fold cross-validation, and renders inline XAI visualizations.

---

## 🏗️ Model Architecture: 4-Branch FusionNet

NeuroFusion uses a modular 4-branch hybrid architecture designed to exploit complementary signal properties:

```
                      ┌───────────────────────────┐
                      │    Raw EEG (1D Temporal)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
┌─────────────────────────┐  ┌─────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│       EEGNet            │  │    CViT     │  │          GNN            │  │      BiomarkerNet       │
│  (1D Conv Temporal-     │  │ (2D Scalogram│  │  (Functional Topography │  │ (Tabular Spectral &     │
│   Spatial Features)     │  │  Transformer│  │   Graph Convolution)    │  │  Complexity Features)   │
└───────────┬─────────────┘  └──────┬──────┘  └────────────┬────────────┘  └────────────┬────────────┘
            │                       │                      │                            │
            └───────────────┬───────┴──────────────┬───────┘                            │
                            │                      │                                    │
                            ▼                      ▼                                    ▼
                      ┌───────────────────────────────────────────────────────────────────────┐
                      │                     Cross-Attention Fusion Layer                      │
                      └───────────────────────────────────┬───────────────────────────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ Classification Head (AD/CN)│
                                            └───────────────────────────┘
```

- **EEGNet**: Preserves fine-grained temporal dynamics using depthwise separable convolutions.
- **CViT**: Captures multi-scale time-frequency patterns from CWT scalograms via self-attention mechanism.
- **GNN**: Constructs brain functional connectivity networks (PLI / Pearson matrices) to capture graph topological shifts.
- **BiomarkerNet**: Models clinical tabular biomarkers (band ratios, Sample Entropy, Higuchi Fractal Dimension).

---

## 🔍 Explainable AI (XAI) & Interpretability

NeuroFusion provides multi-level model transparency to ensure clinical trustworthiness:

- **Grad-CAM Scalograms**: Highlights exact time-frequency windows driving model predictions.
- **SHAP Importance**: Quantifies global and local contributions of tabular spectral and complexity biomarkers.
- **Cross-Path Attention Weights**: Reveals relative reliance on raw waveforms vs. scalograms vs. connectivity vs. tabular biomarkers.
- **Integrated Gradients**: Maps channel-level scalp topographies associated with Alzheimer's EEG slowing.

Sample artifacts can be viewed in the `examples/` and `xai/` directories.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References & Citation

If you use NeuroFusion-AD in your research, please cite:

```bibtex
@article{neurofusion2026,
  title={NeuroFusion-AD: Multi-Modal Hybrid Deep Learning with Cross-Attention Fusion for EEG-Based Alzheimer's Disease Detection},
  author={NeuroFusion Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

---
*For dataset access, refer to OpenNeuro accession ID [ds004504](https://openneuro.org/datasets/ds004504).*
