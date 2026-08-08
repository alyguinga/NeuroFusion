import numpy as np

def compute_correlation_matrix(data):
    """
    Computes the Pearson correlation matrix for a single epoch.
    data: (n_channels, n_times)
    Returns: (n_channels, n_channels)
    """
    return np.corrcoef(data)

def batch_compute_connectivity(X):
    """
    X: (n_epochs, n_channels, n_times)
    Returns: (n_epochs, n_channels, n_channels)
    """
    print("Computing Connectivity Matrices (Absolute Pearson Correlation)...")
    n_epochs, n_channels, _ = X.shape
    adj_matrices = np.zeros((n_epochs, n_channels, n_channels), dtype=np.float32)
    
    for i in range(n_epochs):
        # Add a small epsilon to avoid NaN on constant channels
        corr = np.corrcoef(X[i] + 1e-8 * np.random.randn(*X[i].shape))
        # Use absolute correlation to ensure non-negative adjacency for GNN
        adj_matrices[i] = np.abs(corr).astype(np.float32)
        
    return adj_matrices
