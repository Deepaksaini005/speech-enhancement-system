import numpy as np

def compute_snr(clean, enhanced):
    """Compute the Signal-to-Noise Ratio (SNR) in dB between clean and enhanced audio.
    
    Higher is better (indicating enhanced signal is closer to the clean reference).
    """
    clean = np.asarray(clean, dtype=np.float32)
    enhanced = np.asarray(enhanced, dtype=np.float32)
    
    min_len = min(len(clean), len(enhanced))
    if min_len == 0:
        return 0.0
        
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - enhanced) ** 2)
    
    if noise_power < 1e-12:
        return 100.0  # Cap at high value if perfect match
    if signal_power < 1e-12:
        return -100.0
        
    return float(10 * np.log10(signal_power / noise_power))
