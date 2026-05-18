import torch
from src.deep_model import WasteClassifierCNN


def test_forward_pass_shape():
    model = WasteClassifierCNN(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)


def test_freeze_backbone():
    model = WasteClassifierCNN(num_classes=6)
    model.freeze_backbone()
    for name, p in model.backbone.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"


def test_unfreeze_backbone():
    model = WasteClassifierCNN(num_classes=6)
    model.freeze_backbone()
    model.unfreeze_backbone()
    for name, p in model.backbone.named_parameters():
        assert p.requires_grad, f"{name} should be trainable"


def test_classifier_always_trainable():
    model = WasteClassifierCNN(num_classes=6)
    model.freeze_backbone()
    for name, p in model.classifier.named_parameters():
        assert p.requires_grad, f"classifier.{name} must remain trainable when backbone is frozen"
