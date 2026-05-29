import librosa


def compute_mel_spectrogram(signal, sr=16000, n_mels=128):
    return librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
