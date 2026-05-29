import numpy as np
import librosa


def load_audio(path, sr=16000):
    signal, _ = librosa.load(str(path), sr=sr, mono=True)
    return signal


def preprocess_audio(signal, sr=16000, duration=None):
    if duration is not None:
        target_length = int(sr * duration)
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            signal = signal[:target_length]

    peak = np.max(np.abs(signal)) if signal.size else 1.0
    if peak < 1e-9:
        peak = 1.0
    return signal / peak
