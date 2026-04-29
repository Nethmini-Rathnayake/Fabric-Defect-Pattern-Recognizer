import torch
import torch.nn as nn
from torchvision import models


DEFECT_CLASSES = [
    "oil_stain",
    "dye_stain",
    "hole_snag",
    "drop_stitch",
    "weave_distortion",
    "slub_nep",
    "shade_variation",
    "shrinkage",
]


class FabricDefectModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(checkpoint_path: str, device: str = "cpu") -> FabricDefectModel:
    model = FabricDefectModel()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model
