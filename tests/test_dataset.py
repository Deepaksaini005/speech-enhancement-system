import pytest

from src.data.dataset import VoiceEnhancementDataset


def test_dataset_loads():
    dataset = VoiceEnhancementDataset("data/metadata/file_pairs.csv")
    assert hasattr(dataset, "__len__")
