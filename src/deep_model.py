import torch
import torch.nn as nn
import torchvision.models as models
from src.utils import NUM_CLASSES


class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        efficientnet = models.efficientnet_b0(weights=weights)
        self.backbone = efficientnet.features
        self.pool = efficientnet.avgpool
        in_features = efficientnet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
