import numpy as np
from PIL import Image
from src.preprocessing import get_transforms


def test_train_transforms_output_shape():
    tfm = get_transforms("train")
    img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
    out = tfm(img)
    assert out.shape == (3, 224, 224)


def test_val_transforms_output_shape():
    tfm = get_transforms("val")
    img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
    out = tfm(img)
    assert out.shape == (3, 224, 224)


def test_val_transforms_deterministic():
    tfm = get_transforms("val")
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out1 = tfm(img)
    out2 = tfm(img)
    import torch
    assert torch.allclose(out1, out2), "Val transforms must be deterministic"


def test_transforms_normalization():
    # Pure black image: channel 0 = (0 - 0.485) / 0.229 ≈ -2.12 → below 0
    tfm = get_transforms("val")
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out = tfm(img)
    assert out.min() < 0, "ImageNet normalization should push pure-black image below 0"
