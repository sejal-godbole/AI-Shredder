import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(checkpoint_path=None, num_classes=16, device='cpu'):
    model = EfficientNetB0Classifier(num_classes=num_classes)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model
