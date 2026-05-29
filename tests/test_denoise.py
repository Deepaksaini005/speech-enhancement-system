import numpy as np
from src.processing.denoise import denoise
from src.evaluation.metrics import compute_snr

def test_denoise_reduces_noise_energy():
    # Create a synthetic signal with a silent segment and a signal segment
    np.random.seed(42)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Clean signal: quiet background with a 400Hz tone in the second half
    clean = np.zeros(len(t))
    clean[len(t)//2:] = 0.5 * np.sin(2 * np.pi * 400 * t[len(t)//2:])
    
    # Add mild random noise (std = 0.05)
    noise = np.random.normal(0, 0.05, len(clean))
    noisy = clean + noise
    
    # Run the denoising function
    denoised = denoise(noisy, sr=sr, strength=0.9)
    
    # Verify exact length preservation
    assert len(denoised) == len(noisy)
    
    # Verify finiteness
    assert np.all(np.isfinite(denoised))
    
    # Verify noise energy reduction in the silent first half
    noise_energy_before = np.sqrt(np.mean(noisy[:len(t)//2] ** 2))
    noise_energy_after = np.sqrt(np.mean(denoised[:len(t)//2] ** 2))
    
    print(f"Noise energy before: {noise_energy_before:.6f}, after: {noise_energy_after:.6f}")
    
    # The noise energy in the silent section should be significantly reduced
    assert noise_energy_after < noise_energy_before
    
    # Verify that SNR is finite
    snr_val = compute_snr(clean, denoised)
    assert np.isfinite(snr_val)
