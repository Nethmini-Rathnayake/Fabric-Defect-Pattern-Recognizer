import torch
import torch.nn as nn
from torchvision import models


DEFECT_CLASSES = [
    "normal",
    "stain",
    "tear",
    "weave_distortion",
]


class FabricDefectModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(checkpoint_path: str, device: str = "cpu") -> FabricDefectModel:
    model = FabricDefectModel()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model
