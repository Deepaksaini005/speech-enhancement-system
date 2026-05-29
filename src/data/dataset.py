import csv
from pathlib import Path
from torch.utils.data import Dataset
from src.data.preprocessing import load_audio, preprocess_audio

class VoiceEnhancementDataset(Dataset):
    def __init__(self, file_pairs_csv, transform=None, sr=16000, duration=None):
        self.file_pairs_csv = Path(file_pairs_csv)
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.examples = self._load_pairs()

    def _load_pairs(self):
        if not self.file_pairs_csv.exists():
            return []

        examples = []
        with self.file_pairs_csv.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                clean_path = Path(row.get("clean_path", "")).expanduser()
                noisy_path = Path(row.get("noisy_path", "")).expanduser()
                if clean_path.exists() and noisy_path.exists():
                    examples.append((clean_path, noisy_path))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.examples[idx]
        clean = load_audio(clean_path, sr=self.sr)
        noisy = load_audio(noisy_path, sr=self.sr)
        clean = preprocess_audio(clean, sr=self.sr, duration=self.duration)
        noisy = preprocess_audio(noisy, sr=self.sr, duration=self.duration)

        if self.transform is not None:
            clean, noisy = self.transform(clean, noisy)

        return {
            "clean": clean,
            "noisy": noisy,
            "clean_path": str(clean_path),
            "noisy_path": str(noisy_path),
        }
