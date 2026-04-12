"""Model architectures and weight management.

Two architectures are available:

- CustomCNN: A 6.7M-parameter CNN trained from scratch (the primary model).
- PretrainedEfficientNet: EfficientNet-B0 backbone with a custom classification head.
"""

import os

import gdown
import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """Custom 6.7M-parameter CNN for binary melanoma classification.

    Architecture: 4 conv blocks with BatchNorm + SiLU activations, progressive
    dropout, followed by a 3-layer classifier head. Outputs raw logits.
    Expects 224x224 RGB input.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PretrainedEfficientNet(nn.Module):
    """EfficientNet-B0 with a custom classification head.

    Uses torchvision's pretrained EfficientNet-B0 as the backbone, replacing
    the default classifier with BatchNorm + SiLU + Dropout layers.
    Outputs raw logits.
    """

    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)


def download_weights_from_gdrive(model_path, file_id):
    """Download model weights from Google Drive if not already present."""
    if os.path.exists(model_path):
        print(f"{model_path} already exists, skipping download.")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from {url}")
    gdown.download(url, str(model_path), quiet=False)


def load_model(model_class, weights_path, device="cpu"):
    """Instantiate a model and load saved weights.

    Args:
        model_class: One of CustomCNN or PretrainedEfficientNet.
        weights_path: Path to a .pth state_dict file.
        device: 'cpu' or 'cuda'.

    Returns:
        The model in eval mode on the specified device.
    """
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    return model.to(device)
