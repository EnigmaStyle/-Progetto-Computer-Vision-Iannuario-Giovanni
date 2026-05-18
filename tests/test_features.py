import numpy as np
from src.features import extract_hog, extract_hog_batch


def test_hog_output_is_1d():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    feat = extract_hog(img)
    assert feat.ndim == 1
    assert len(feat) > 0


def test_hog_batch_shape():
    imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)]
    feats = extract_hog_batch(imgs)
    assert feats.shape[0] == 5
    assert feats.ndim == 2


def test_hog_consistent_size():
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    f1 = extract_hog(img1)
    f2 = extract_hog(img2)
    assert len(f1) == len(f2), "HOG output must be same size regardless of input resolution"
