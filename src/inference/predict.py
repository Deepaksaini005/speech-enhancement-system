import torch
import numpy as np


def predict(model, noisy_signal, device=None):
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()

    signal = np.asarray(noisy_signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[np.newaxis, np.newaxis, :]
    elif signal.ndim == 2:
        signal = signal[np.newaxis, :, :]

    tensor = torch.from_numpy(signal).to(device)
    with torch.no_grad():
        enhanced = model(tensor)

    enhanced = enhanced.squeeze().cpu().numpy()
    return enhanced
