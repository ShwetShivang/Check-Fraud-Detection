"""
CNN Baseline Classifier for ink type classification.

A standard CNN that classifies individual patches as ink class A or B.
Used as a baseline comparison against the Siamese network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class InkClassifierCNN(nn.Module):
    """
    Lightweight CNN for binary ink classification.

    Architecture (~100K params):
      Conv2d(3→16) → BN → ReLU → MaxPool
      Conv2d(16→32) → BN → ReLU → MaxPool
      Conv2d(32→64) → BN → ReLU → AdaptiveAvgPool
      FC(64→32) → ReLU → Dropout → FC(32→num_classes)
    """

    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES

        self.features = nn.Sequential(
            # Block 1: 64×64 → 32×32
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×32 → 16×16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16×16 → adaptive pool → 1×1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Input patch (B, 3, 64, 64)

        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_feature_maps(self, x):
        """Extract feature maps from the last conv layer (for Grad-CAM)."""
        return self.features(x)


if __name__ == "__main__":
    model = InkClassifierCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN Classifier total parameters: {total_params:,}")

    x = torch.randn(4, 3, 64, 64)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    probs = model.predict_proba(x)
    print(f"Probabilities: {probs}")
