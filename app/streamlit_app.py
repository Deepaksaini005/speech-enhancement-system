import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
import streamlit as st
import importlib

MODEL_PATH = ROOT / "models" / "best_model.pth"

st.title("Voice Enhancement AI")
st.write("Upload noisy audio and enhance it with a trained model.")

@st.cache_resource
def get_model(model_path, mtime):
    try:
        mod = importlib.import_module("src.inference.enhance_audio")
        return mod.load_model(model_path)
    except Exception as e:
        st.warning(f"Failed to load model module: {e}. Using an uninitialized model instead.")
        try:
            mod = importlib.import_module("src.inference.enhance_audio")
            return mod.load_model(None)
        except Exception as e2:
            st.error(f"Cannot import inference module: {e2}")
            raise

# Get modification time and size to invalidate cache when model is trained or updated
mtime = MODEL_PATH.stat().st_mtime if (MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0) else 0
model = get_model(MODEL_PATH, mtime)

uploaded_file = st.file_uploader("Upload noisy audio file", type=["wav", "flac", "mp3", "ogg"])
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    st.markdown("---")
    # Let the user select the enhancement mode
    enhancement_mode = st.radio(
        "Enhancement Mode",
        ["Classic Spectral Subtraction (Crystal Clear - Recommended)", "Deep Learning Model (CNN - Experimental)"]
    )
    
    # Denoise strength slider
    strength = st.slider("Denoise strength (0 = off, 1 = max)", min_value=0.0, max_value=1.0, value=0.95, step=0.05)

    if st.button("Enhance audio"):
        with st.spinner("Enhancing audio..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in.flush()
                input_path = Path(tmp_in.name)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = Path(tmp_out.name)

            try:
                mod = importlib.import_module("src.inference.enhance_audio")
                importlib.reload(mod)
                use_model = "Deep Learning Model" in enhancement_mode
                enhanced_path = mod.enhance_audio(model, input_path, output_path, strength=strength, use_model=use_model)
            except Exception as e:
                st.error(f"Enhancement failed: {e}")
                enhanced_path = None

        if enhanced_path is not None:
            st.success("Enhancement complete.")
            with open(enhanced_path, "rb") as f:
                enhanced_bytes = f.read()
                st.audio(enhanced_bytes, format="audio/wav")
                st.download_button("Download enhanced audio", data=enhanced_bytes, file_name="enhanced.wav", mime="audio/wav")
