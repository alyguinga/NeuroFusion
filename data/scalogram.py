import numpy as np
from mne.time_frequency import tfr_array_morlet

def batch_compute_scalograms(X, sfreq, freqs=np.arange(2, 45, 2), n_jobs=1, chunk_size=50, decim=4):
    """
    Computes Morlet wavelet scalograms for a batch of EEG epochs with memory-efficient chunking.
    X: (n_epochs, n_channels, n_times)
    decim: Decimation factor to reduce time dimension (e.g., 4 reduces 1024 to 256)
    Returns: (n_epochs, n_channels, n_freqs, n_times/decim) representing amplitude.
    """
    n_epochs, n_channels, n_times = X.shape
    n_freqs = len(freqs)
    n_times_out = int(np.ceil(n_times / decim))
    
    print(f"Computing CWT Scalograms (using MNE Morlet) in chunks of {chunk_size}...")
    
    # Calculate number of cycles for each frequency
    n_cycles = freqs / 2.0
    n_cycles[n_cycles < 2] = 2
    
    # Pre-allocate output array to avoid repeated re-allocations
    X_scalo = np.zeros((n_epochs, n_channels, n_freqs, n_times_out), dtype=np.float32)
    
    for start in range(0, n_epochs, chunk_size):
        end = min(start + chunk_size, n_epochs)
        X_chunk = X[start:end]
        
        # tfr_array_morlet returns power.
        power_chunk = tfr_array_morlet(
            X_chunk, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, 
            output='power', n_jobs=n_jobs, use_fft=True, decim=decim
        )
        
        X_scalo[start:end] = np.sqrt(power_chunk).astype(np.float32)
        
        if (start + chunk_size) % (chunk_size * 5) == 0 or end == n_epochs:
            print(f"  Processed {end}/{n_epochs} epochs")
            
    return X_scalo
