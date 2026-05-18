import torch
from src.utils import get_device, set_seed, CLASS_NAMES, NUM_CLASSES


def test_class_names():
    assert len(CLASS_NAMES) == 6
    assert "cardboard" in CLASS_NAMES
    assert "glass" in CLASS_NAMES


def test_num_classes():
    assert NUM_CLASSES == 6


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_get_device_returns_valid():
    device = get_device()
    assert device in ["cuda", "mps", "cpu"]
