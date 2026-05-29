import librosa


def compute_mfcc(signal, sr=16000, n_mfcc=13):
    return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
