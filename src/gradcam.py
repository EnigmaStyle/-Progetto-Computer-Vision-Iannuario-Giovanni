import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._features: torch.Tensor = None
        self._gradients: torch.Tensor = None

        target_layer.register_forward_hook(self._save_features)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, _, __, output):
        self._features = output

    def _save_gradients(self, _, __, grad_output):
        self._gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a (224, 224) float32 heatmap in [0, 1]."""
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, class_idx].sum().backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)
