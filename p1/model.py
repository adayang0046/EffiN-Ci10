import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights  # Import weights class

def get_efficientnet(num_classes=10):
    # Use the updated way to load pretrained weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)  # Use weights instead of 'pretrained=True'

    # Modify the classifier to match CIFAR-10's 10 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
