from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import torch
from src.inference.predict import predict
from src.models.cnn_model import CNNModel
import importlib

# Import denoiser lazily inside function to avoid import-time errors in Streamlit


def _safe_load_state_dict(model, state):
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    try:
        model.load_state_dict(state)
        return True
    except Exception:
        if isinstance(state, dict):
            stripped = {k.replace("module.", "") if isinstance(k, str) else k: v for k, v in state.items()}
            try:
                model.load_state_dict(stripped, strict=False)
                return True
            except Exception:
                return False
        return False


def load_model(model_path=None, device=None):
    device = device or torch.device("cpu")
    model = CNNModel().to(device)
    model.is_trained = False
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists() and model_path.stat().st_size > 0:
            try:
                state = torch.load(str(model_path), map_location=device)
                if _safe_load_state_dict(model, state):
                    model.is_trained = True
                else:
                    print(f"Warning: model file {model_path} contained incompatible data; using fresh model")
            except Exception as e:
                print(f"Warning: failed to load model from {model_path}: {e}; using fresh model")
        else:
            print(f"Info: model file {model_path} missing or empty; using fresh uninitialized model")
    return model


def enhance_audio(model, input_path, output_path, sr=16000, strength=1.0, noise_duration=0.5, use_model=False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    signal, file_sr = sf.read(str(input_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if file_sr != sr:
        signal = librosa.resample(signal, orig_sr=file_sr, target_sr=sr)

    noisy_signal = signal.copy()

    # Apply default denoising (preserve pitch) using lazy import
    denoised_signal = None
    try:
        denoise_mod = importlib.import_module("src.processing.denoise")
        denoise_fn = getattr(denoise_mod, "denoise", None)
        if denoise_fn is None:
            raise ImportError("denoise function not found in src.processing.denoise")
        denoised_signal = denoise_fn(signal, sr=sr, noise_duration=noise_duration, strength=strength)
    except Exception as e:
        print(f"Warning: denoising failed or unavailable: {e}; continuing with original signal")
        denoised_signal = signal

    # If requested and successfully trained, run model-based enhancement on the original noisy input.
    if use_model and getattr(model, "is_trained", False):
        try:
            device = next(model.parameters()).device
            peak_val = np.max(np.abs(noisy_signal)) if (noisy_signal is not None and noisy_signal.size > 0) else 0.0
            norm_signal = noisy_signal / peak_val if peak_val > 1e-12 else noisy_signal
            model_output = predict(model, norm_signal, device=device)
            model_output = np.asarray(model_output, dtype=np.float32)

            if model_output.shape != noisy_signal.shape:
                desired = noisy_signal.shape[0]
                if model_output.shape[0] > desired:
                    model_output = model_output[:desired]
                else:
                    pad_width = desired - model_output.shape[0]
                    model_output = np.pad(model_output, (0, pad_width), mode="constant")

            enhanced = np.clip(model_output * peak_val, -1.0, 1.0)
            enhanced = np.clip(0.8 * enhanced + 0.2 * denoised_signal, -1.0, 1.0)
        except Exception as e:
            print(f"Warning: model prediction failed: {e}; falling back to denoised signal")
            enhanced = denoised_signal
    else:
        enhanced = denoised_signal

    # Normalize volume using target RMS energy to guarantee uniform and rich loudness (even with pops/transients)
    rms = np.sqrt(np.mean(enhanced ** 2)) if (enhanced is not None and enhanced.size > 0) else 0.0
    if rms > 1e-6:
        target_rms = 0.15
        scale = target_rms / rms
        # Clip peaks safely to avoid digital distortion
        enhanced = np.clip(enhanced * scale, -0.98, 0.98)

    sf.write(str(output_path), enhanced, sr, subtype="PCM_16")
    return output_path
