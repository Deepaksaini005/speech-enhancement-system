import torch


def compute_spectrogram(signal, n_fft=512, hop_length=128):
    return torch.stft(torch.tensor(signal), n_fft=n_fft, hop_length=hop_length)
