import numpy as np
from src.evaluate import compute_metrics


def test_compute_metrics_perfect():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_metrics_returns_all_keys():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 0, 0, 2, 2]
    m = compute_metrics(y_true, y_pred)
    for key in ("accuracy", "macro_f1", "weighted_f1", "confusion_matrix", "report"):
        assert key in m


def test_confusion_matrix_shape():
    y_true = [i % 6 for i in range(30)]
    y_pred = [i % 6 for i in range(30)]
    m = compute_metrics(y_true, y_pred)
    assert m["confusion_matrix"].shape == (6, 6)


def test_gradcam_output():
    import torch
    from src.deep_model import WasteClassifierCNN
    from src.gradcam import GradCAM

    model = WasteClassifierCNN(num_classes=6)
    model.eval()
    cam = GradCAM(model, target_layer=model.backbone[-1])
    x = torch.randn(1, 3, 224, 224)
    heatmap = cam(x, class_idx=0)
    assert heatmap.shape == (224, 224)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6
