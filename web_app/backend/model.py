import os

import gdown
import torch.nn as nn


def download_model_from_gdrive(model_path, file_id):
    if os.path.exists(model_path):
        print(f"{model_path} already exists, skipping download.")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from {url}")
    gdown.download(url, model_path, quiet=False)


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomEfficientNet, self).__init__()

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
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 1)  # Binary classification (logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
