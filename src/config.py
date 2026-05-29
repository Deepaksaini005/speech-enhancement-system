from pathlib import Path

class Config:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT / "models"
    OUTPUTS_DIR = ROOT / "outputs"
    SAMPLE_RATE = 16000
    DURATION = 4.0
