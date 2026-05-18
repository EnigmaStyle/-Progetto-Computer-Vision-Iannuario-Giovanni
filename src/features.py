from typing import List
import cv2
import numpy as np
from skimage.feature import hog

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    channel_axis=-1,
)


def extract_hog(image: np.ndarray) -> np.ndarray:
    """Extract HOG descriptor from an HxWx3 uint8 RGB image."""
    img = cv2.resize(image, (224, 224))
    return hog(img, **HOG_PARAMS).astype(np.float32)


def extract_hog_batch(images: List[np.ndarray]) -> np.ndarray:
    """Vectorise HOG extraction over a list of images."""
    return np.stack([extract_hog(img) for img in images])
