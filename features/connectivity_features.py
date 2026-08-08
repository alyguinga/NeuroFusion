"""
v3/features/connectivity_features.py
═══════════════════════════════════════════════════════════════════════════
Functional connectivity biomarkers for EEG-based AD detection.

AD as a "Disconnection Syndrome":
  - Disrupted communication between brain regions
  - PLI decreases across cortical networks
  - Small-worldness reduced (loss of efficient network topology)

Mathematical Foundations:
─────────────────────────
1. Phase Lag Index (PLI):
   PLI_{ij} = |⟨sign(Δφ_{ij}(t))⟩|
   where Δφ_{ij}(t) = phase_i(t) - phase_j(t)
   computed via Hilbert transform.

   PLI = 0: no phase coupling or perfect synchrony (volume conduction)
   PLI = 1: perfect non-zero phase coupling
   Key advantage over coherence: robust to volume conduction artifacts.
   (Stam et al., 2007; NeuroImage 2025)

2. Graph Metrics from PLI connectivity matrix:

   a) Clustering Coefficient (C):
      C_i = (2 × T_i) / (k_i × (k_i - 1))
      where T_i = # triangles through node i,
            k_i = degree of node i (# connections above threshold).
      Measures local integration / functional segregation.

   b) Characteristic Path Length (L):
      L = (1 / N(N-1)) × Σ_{i≠j} d_{ij}
      where d_{ij} = shortest path between nodes i and j
      in the thresholded binary graph (edge weights inverted).
      Measures global efficiency of information transfer.

   c) Small-Worldness (σ):
      σ = (C / C_rand) / (L / L_rand)
      where C_rand, L_rand are from equivalent random graphs.
      σ > 1 indicates small-world topology (high clustering + short paths).
      σ is REDUCED in AD (disrupted network architecture).

References:
  - Stam et al., 2007: PLI original paper
  - NeuroImage 2025: PLI in AD
  - Frontiers 2024-2025: Graph metrics in neurodegeneration
  - Watts & Strogatz, 1998: Small-world networks
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as CFG


# ═════════════════════════════════════════════════════════════════════════
# 1. PHASE LAG INDEX (PLI)
# ═════════════════════════════════════════════════════════════════════════
def compute_pli(epoch_data):
    """
    Compute Phase Lag Index matrix for a single epoch.

    PLI_{ij} = |mean(sign(Δφ_{ij}(t)))|

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    pli_matrix : ndarray, shape (n_channels, n_channels)
        Symmetric PLI connectivity matrix in [0, 1].
    """
    n_ch, n_times = epoch_data.shape

    # Compute instantaneous phase via Hilbert transform
    # analytic_signal = x(t) + j × H{x(t)}
    # phase = angle(analytic_signal)
    analytic = hilbert(epoch_data, axis=-1)
    phase = np.angle(analytic)

    pli = np.zeros((n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            # Phase difference
            dphi = phase[i] - phase[j]
            # PLI = |mean(sign(Δφ))|
            pli_val = np.abs(np.mean(np.sign(dphi)))
            pli[i, j] = pli_val
            pli[j, i] = pli_val

    return pli


# ═════════════════════════════════════════════════════════════════════════
# 2. GRAPH METRICS
# ═════════════════════════════════════════════════════════════════════════
def _threshold_matrix(matrix, threshold=0.3):
    """
    Threshold a connectivity matrix to create a binary adjacency.

    Parameters
    ----------
    matrix : ndarray (N, N)
    threshold : float — edges below this value are removed

    Returns
    -------
    adj : ndarray (N, N) — binary adjacency matrix
    """
    adj = (matrix > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def clustering_coefficient(adj):
    """
    Compute mean clustering coefficient of a binary graph.

    C_i = (2 × T_i) / (k_i × (k_i - 1))
    where T_i = # triangles through node i, k_i = degree of node i.

    Parameters
    ----------
    adj : ndarray (N, N) — binary adjacency matrix

    Returns
    -------
    mean_cc : float — mean clustering coefficient across nodes
    """
    N = adj.shape[0]
    cc = np.zeros(N)

    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc[i] = 0
            continue

        # Count triangles: how many pairs of neighbors are connected?
        triangles = 0
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                if adj[neighbors[ni], neighbors[nj]] > 0:
                    triangles += 1

        cc[i] = (2 * triangles) / (k * (k - 1))

    return np.mean(cc)


def characteristic_path_length(adj):
    """
    Compute characteristic path length using BFS shortest paths.

    L = (1 / N(N-1)) × Σ_{i≠j} d_{ij}

    Parameters
    ----------
    adj : ndarray (N, N) — binary adjacency matrix

    Returns
    -------
    cpl : float — characteristic path length (inf if graph disconnected)
    """
    N = adj.shape[0]
    total_dist = 0
    n_pairs = 0

    for src in range(N):
        # BFS from src
        visited = np.zeros(N, dtype=bool)
        visited[src] = True
        queue = [src]
        dist = np.full(N, np.inf)
        dist[src] = 0

        while queue:
            node = queue.pop(0)
            for neighbor in range(N):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        for tgt in range(N):
            if tgt != src and dist[tgt] < np.inf:
                total_dist += dist[tgt]
                n_pairs += 1

    if n_pairs == 0:
        return float('inf')

    return total_dist / n_pairs


def small_worldness(adj, n_rand=5):
    """
    Compute small-worldness ratio σ.

    σ = (C / C_rand) / (L / L_rand)

    C_rand, L_rand computed from random graphs with same density.
    σ > 1 indicates small-world network.
    σ is REDUCED in AD (NeuroImage 2025).

    Parameters
    ----------
    adj : ndarray (N, N) — binary adjacency
    n_rand : int — number of random graphs to average

    Returns
    -------
    sigma : float — small-worldness ratio
    """
    C = clustering_coefficient(adj)
    L = characteristic_path_length(adj)

    if L == float('inf') or L == 0:
        return 0.0

    N = adj.shape[0]
    n_edges = int(np.sum(adj) / 2)

    # Generate random graphs with same number of nodes and edges
    C_rands = []
    L_rands = []

    for _ in range(n_rand):
        rand_adj = np.zeros((N, N), dtype=np.float32)
        edges_placed = 0
        max_attempts = n_edges * 20

        attempts = 0
        while edges_placed < n_edges and attempts < max_attempts:
            i, j = np.random.randint(0, N, 2)
            if i != j and rand_adj[i, j] == 0:
                rand_adj[i, j] = 1
                rand_adj[j, i] = 1
                edges_placed += 1
            attempts += 1

        C_rands.append(clustering_coefficient(rand_adj))
        L_rands.append(characteristic_path_length(rand_adj))

    C_rand = np.mean(C_rands) if C_rands else 1e-10
    L_rand = np.mean([l for l in L_rands if l < float('inf')]) if L_rands else 1.0

    if C_rand < 1e-10 or L_rand < 1e-10:
        return 0.0

    sigma = (C / C_rand) / (L / L_rand)
    return sigma


def global_efficiency(adj):
    """
    Compute global efficiency of a binary graph.

    E_glob = (1 / N(N-1)) × Σ_{i≠j} (1 / d_{ij})

    More robust than characteristic path length for disconnected graphs
    because unreachable pairs contribute 0 instead of inf.
    (Latora & Marchiori, 2001; Frontiers 2024)

    Parameters
    ----------
    adj : ndarray (N, N) — binary adjacency matrix

    Returns
    -------
    efficiency : float
    """
    N = adj.shape[0]
    if N < 2:
        return 0.0

    total_eff = 0.0
    for src in range(N):
        # BFS from src
        visited = np.zeros(N, dtype=bool)
        visited[src] = True
        queue = [src]
        dist = np.full(N, np.inf)
        dist[src] = 0

        while queue:
            node = queue.pop(0)
            for neighbor in range(N):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        for tgt in range(N):
            if tgt != src and dist[tgt] < np.inf:
                total_eff += 1.0 / dist[tgt]

    return total_eff / (N * (N - 1))


def _bandpass_filter_conn(data, low, high, sfreq=256, order=4):
    """Zero-phase Butterworth bandpass filter for connectivity."""
    nyq = sfreq / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def compute_alpha_pli(epoch_data, sfreq=256):
    """
    Compute PLI on alpha-band (8-13 Hz) filtered signals.

    AD connectivity disruption is strongest in the alpha band.
    (Stam et al. 2007; NeuroImage 2025)

    Parameters
    ----------
    epoch_data : ndarray (n_ch, n_times)
    sfreq : float

    Returns
    -------
    mean_alpha_pli : float
    std_alpha_pli : float
    """
    filtered = _bandpass_filter_conn(epoch_data, 8.0, 13.0, sfreq)
    pli_alpha = compute_pli(filtered)
    n_ch = epoch_data.shape[0]
    triu_vals = pli_alpha[np.triu_indices(n_ch, k=1)]
    return float(np.mean(triu_vals)), float(np.std(triu_vals))


# ═════════════════════════════════════════════════════════════════════════
# COMBINED CONNECTIVITY FEATURE EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════
def extract_connectivity_features(epoch_data, pli_threshold=0.3, sfreq=256):
    """
    Extract connectivity features from a single epoch.

    Returns graph-level statistics only (no raw PLI pairwise values).
    The GNN branch handles raw PLI topology via its adjacency matrix input,
    so feeding pairwise values to BiomarkerNet is redundant.
    (Research consensus: GNNs outperform flat MLP on pairwise connectivity.)

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
    pli_threshold : float — threshold for binary graph construction
    sfreq : float

    Returns
    -------
    features : ndarray (8,)
        [mean_pli, std_pli, clustering_coeff, path_length,
         small_world, global_eff, mean_alpha_pli, std_alpha_pli]
    pli_matrix : ndarray (n_ch, n_ch) — for GNN input
    """
    n_ch = epoch_data.shape[0]

    # Compute broadband PLI matrix
    pli = compute_pli(epoch_data)

    # Graph metrics from thresholded PLI
    adj = _threshold_matrix(pli, threshold=pli_threshold)

    mean_pli = np.mean(pli[np.triu_indices(n_ch, k=1)])
    std_pli = np.std(pli[np.triu_indices(n_ch, k=1)])
    cc = clustering_coefficient(adj)
    cpl = characteristic_path_length(adj)
    if cpl == float('inf'):
        cpl = n_ch  # cap at max possible for disconnected graphs
    sw = small_worldness(adj, n_rand=3)
    ge = global_efficiency(adj)

    # Alpha-band PLI (AD-sensitive)
    mean_alpha_pli, std_alpha_pli = compute_alpha_pli(epoch_data, sfreq)

    features = np.array([
        mean_pli, std_pli, cc, cpl, sw, ge,
        mean_alpha_pli, std_alpha_pli
    ], dtype=np.float32)

    return features, pli


def batch_extract_connectivity(X_1d):
    """
    Extract connectivity features + PLI matrices for all epochs.

    Parameters
    ----------
    X_1d : ndarray, shape (N, n_channels, n_times)

    Returns
    -------
    X_conn : ndarray, shape (N, n_features)
    X_pli : ndarray, shape (N, n_ch, n_ch) — PLI matrices for GNN
    """
    N, n_ch, _ = X_1d.shape

    # Get feature size
    f0, pli0 = extract_connectivity_features(X_1d[0])
    n_feat = len(f0)

    X_conn = np.zeros((N, n_feat), dtype=np.float32)
    X_pli = np.zeros((N, n_ch, n_ch), dtype=np.float32)

    X_conn[0] = f0
    X_pli[0] = pli0

    for i in range(1, N):
        X_conn[i], X_pli[i] = extract_connectivity_features(X_1d[i])
        if (i + 1) % 200 == 0:
            print(f"    [connectivity] Processed {i+1}/{N} epochs")

    np.nan_to_num(X_conn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_pli, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  [connectivity] Extracted {n_feat} features per epoch ({N} epochs)")
    return X_conn, X_pli


def get_connectivity_feature_names(n_channels):
    """Return human-readable feature names."""
    names = [
        'mean_pli', 'std_pli', 'clustering_coeff', 'char_path_length',
        'small_worldness', 'global_efficiency', 'mean_alpha_pli', 'std_alpha_pli'
    ]
    return names
