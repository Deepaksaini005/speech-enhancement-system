import numpy as np
import librosa


def spectral_gate(signal, sr=16000, n_fft=1024, hop_length=256, noise_duration=0.5, n_std_thresh=1.5, prop_decrease=1.0):
    """Simple spectral-gating denoiser.

    - Estimates noise from the first `noise_duration` seconds.
    - Applies a binary mask based on a threshold (noise mean * n_std_thresh).
    - Returns time-domain denoised signal. Does not change pitch.
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    # Estimate noise from initial frames
    noise_frames = max(1, int((noise_duration * sr) / hop_length))
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Threshold and mask
    thresh = noise_mag * n_std_thresh
    mask = magnitude >= thresh

    # Soft attenuation for masked components
    attenuated = magnitude * mask + magnitude * (~mask) * (1.0 - 0.9 * prop_decrease)

    S_denoised = attenuated * np.exp(1j * phase)
    denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(signal))
    return denoised


def spectral_subtraction(signal, sr=16000, n_fft=1024, hop_length=256, noise_duration=0.5, strength=0.8):
    """Apply spectral subtraction denoising.

    strength: 0.0 (no effect) .. 1.0 (maximum subtraction)
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    noise_frames = max(1, int((noise_duration * sr) / hop_length))
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Subtract scaled noise estimate
    subtracted = magnitude - strength * noise_mag
    subtracted = np.maximum(subtracted, 0.0)

    S_denoised = subtracted * np.exp(1j * phase)
    denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(signal))
    return denoised


def denoise(signal, sr=16000, noise_duration=0.5, strength=0.8):
    """Default denoiser using spectral subtraction with robust noise floor and smoothing.

    Uses the quietest frames to estimate noise and then applies aggressive, frequency-dependent
    subtraction. Strength 0..1 controls how much background is removed.
    """
    try:
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        n_fft = 1024
        hop_length = 256
        S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = np.abs(S), np.angle(S)

        # Use the quietest frames as the noise profile for robust background estimation.
        frame_energy = magnitude.mean(axis=0)
        n_frames = frame_energy.shape[0]
        k = max(1, int(n_frames * 0.1))
        idx = np.argsort(frame_energy)[:k]
        noise_mag = np.mean(magnitude[:, idx], axis=1, keepdims=True)

        # Over-subtract more strongly when requested, while keeping a small floor to avoid artifacts.
        subtraction_factor = 1.2 + 0.8 * strength
        reduced = magnitude - subtraction_factor * noise_mag
        floor = np.maximum(1e-4, 0.01 * magnitude)
        subtracted = np.maximum(reduced, floor)

        # Smooth the reduced spectrum in time to reduce musical noise and preserve speech clarity.
        kernel_size = 5
        pad = kernel_size // 2
        mag_padded = np.pad(subtracted, ((0, 0), (pad, pad)), mode='reflect')
        smoothed = np.empty_like(subtracted)
        for j in range(subtracted.shape[1]):
            smoothed[:, j] = mag_padded[:, j:j + kernel_size].mean(axis=1)

        gain = np.clip(smoothed / (magnitude + 1e-12), 0.0, 1.0)
        min_gain = 0.02 + 0.18 * (1.0 - strength)
        gain = np.maximum(gain, min_gain)

        S_denoised = gain * magnitude * np.exp(1j * phase)
        denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(signal))
        return denoised
    except Exception:
        try:
            return spectral_subtraction(signal, sr=sr, noise_duration=noise_duration, strength=strength)
        except Exception:
            return spectral_gate(signal, sr=sr, noise_duration=noise_duration, prop_decrease=float(strength))
