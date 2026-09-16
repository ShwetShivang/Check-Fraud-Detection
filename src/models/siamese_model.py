"""
Siamese Network for ink similarity learning.

Architecture:
  - Twin CNN branches with shared weights
  - Outputs embedding vectors
  - Contrastive loss for training
  - Euclidean distance for similarity scoring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SiameseEmbeddingNetwork(nn.Module):
    """
    CNN branch for the Siamese network.
    Maps a 64×64 RGB patch to a compact embedding vector.

    Architecture (lightweight — ~200K params):
      Conv2d(3→32) → BN → ReLU → MaxPool
      Conv2d(32→64) → BN → ReLU → MaxPool
      Conv2d(64→128) → BN → ReLU → AdaptiveAvgPool
      FC → Embedding (64-dim)
    """

    def __init__(self, embedding_dim=None):
        super().__init__()
        embedding_dim = embedding_dim or config.EMBEDDING_DIM

        self.features = nn.Sequential(
            # Block 1: 64×64 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global pooling → (128,)
            nn.AdaptiveAvgPool2d(1),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        """Extract embedding from an image patch."""
        x = self.features(x)
        x = self.embedding(x)
        return x


class SiameseCNN(nn.Module):
    """
    Full Siamese Network.
    Takes two image patches and outputs the distance between their embeddings.
    """

    def __init__(self, embedding_dim=None):
        super().__init__()
        self.branch = SiameseEmbeddingNetwork(embedding_dim)

    def forward_one(self, x):
        """Forward pass through one branch."""
        return self.branch(x)

    def forward(self, x1, x2):
        """
        Forward pass for both inputs.

        Args:
            x1: First patch (B, 3, 64, 64)
            x2: Second patch (B, 3, 64, 64)

        Returns:
            out1: Embedding of x1 (B, embedding_dim)
            out2: Embedding of x2 (B, embedding_dim)
        """
        out1 = self.branch(x1)
        out2 = self.branch(x2)
        return out1, out2

    def get_distance(self, x1, x2):
        """Compute Euclidean distance between embeddings."""
        out1, out2 = self.forward(x1, x2)
        distance = F.pairwise_distance(out1, out2)
        return distance

    def get_similarity(self, x1, x2):
        """
        Compute similarity score between 0 and 1.
        Higher = more similar (same ink).
        """
        distance = self.get_distance(x1, x2)
        # Convert distance to similarity using exponential decay
        # Normalize by the contrastive margin so that distances near the margin
        # map to ~0.37 similarity, and smaller distances map to higher similarity
        similarity = torch.exp(-distance / config.CONTRASTIVE_MARGIN)
        return similarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese network training.

    L = (1-Y) * 0.5 * D²  +  Y * 0.5 * max(0, margin - D)²

    Where:
      Y = 0 for same ink (similar), Y = 1 for different ink (dissimilar)
      D = Euclidean distance between embeddings
    """

    def __init__(self, margin=None):
        super().__init__()
        self.margin = margin or config.CONTRASTIVE_MARGIN

    def forward(self, output1, output2, label):
        """
        Args:
            output1: Embedding from branch 1 (B, embedding_dim)
            output2: Embedding from branch 2 (B, embedding_dim)
            label: 0 = different ink, 1 = same ink

        Returns:
            Contrastive loss scalar
        """
        distance = F.pairwise_distance(output1, output2)

        # label=1 means same ink → pull together (minimize distance)
        # label=0 means different ink → push apart (maximize distance up to margin)
        loss = (label) * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(
                   torch.clamp(self.margin - distance, min=0.0), 2
               )
        return loss.mean() * 0.5


if __name__ == "__main__":
    # Quick sanity check
    model = SiameseCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Siamese CNN total parameters: {total_params:,}")

    # Test forward pass
    x1 = torch.randn(4, 3, 64, 64)
    x2 = torch.randn(4, 3, 64, 64)
    out1, out2 = model(x1, x2)
    print(f"Embedding shape: {out1.shape}")

    dist = model.get_distance(x1, x2)
    print(f"Distances: {dist}")

    sim = model.get_similarity(x1, x2)
    print(f"Similarities: {sim}")

    # Test loss
    criterion = ContrastiveLoss()
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    loss = criterion(out1, out2, labels)
    print(f"Contrastive loss: {loss.item():.4f}")
