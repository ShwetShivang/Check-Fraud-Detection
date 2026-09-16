"""
Training script for all three models:
  1. Siamese Network (contrastive loss)
  2. CNN Baseline (cross-entropy)
  3. Classical ML (SVM / Random Forest)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, save_figure, EarlyStopping, AverageMeter, count_parameters
from src.data_loader import load_dataset
from src.patch_generator import (
    generate_pairs, generate_single_patches,
    PatchPairDataset, SinglePatchDataset
)
from src.models.siamese_model import SiameseCNN, ContrastiveLoss
from src.models.cnn_model import InkClassifierCNN
from src.models.classical_model import ClassicalInkClassifier


def split_data(data_list, train_ratio=0.7, val_ratio=0.15):
    """Split a list into train/val/test sets."""
    n = len(data_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data_list[:n_train]
    val = data_list[n_train:n_train + n_val]
    test = data_list[n_train + n_val:]

    return train, val, test


# ═══════════════════════════════════════════════════════════════════════
# 1. SIAMESE NETWORK TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_siamese(pairs=None, dataset=None):
    """
    Train the Siamese Network with contrastive loss.

    Returns:
        model: Trained SiameseCNN
        history: Dict with training metrics
    """
    print("\n" + "=" * 60)
    print("  TRAINING SIAMESE NETWORK")
    print("=" * 60)

    # Generate pairs if not provided
    if pairs is None:
        if dataset is None:
            dataset = load_dataset()
        pairs = generate_pairs(dataset, max_pairs_per_image=80)

    # Split
    train_pairs, val_pairs, test_pairs = split_data(pairs)
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Create data loaders
    train_dataset = PatchPairDataset(train_pairs)
    val_dataset = PatchPairDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = SiameseCNN().to(config.DEVICE)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {config.DEVICE}")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        # === Train ===
        model.train()
        train_loss = AverageMeter()

        for batch_idx, (pa, pb, labels) in enumerate(train_loader):
            pa = pa.to(config.DEVICE)
            pb = pb.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(pa, pb)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), pa.size(0))

        # === Validate ===
        model.eval()
        val_loss = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for pa, pb, labels in val_loader:
                pa = pa.to(config.DEVICE)
                pb = pb.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                out1, out2 = model(pa, pb)
                loss = criterion(out1, out2, labels)
                val_loss.update(loss.item(), pa.size(0))

                # Accuracy: predict same if distance < margin/2
                distances = torch.nn.functional.pairwise_distance(out1, out2)
                predicted = (distances < config.CONTRASTIVE_MARGIN / 2).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)
        scheduler.step(val_loss.avg)

        history['train_loss'].append(train_loss.avg)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch + 1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss.avg:.4f} | "
              f"Val Loss: {val_loss.avg:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            torch.save(model.state_dict(),
                       os.path.join(config.MODEL_DIR, "siamese_best.pth"))

        # Early stopping
        if early_stopping(val_loss.avg):
            print(f"  → Early stopping at epoch {epoch + 1}")
            break

    # Load best weights
    model.load_state_dict(
        torch.load(os.path.join(config.MODEL_DIR, "siamese_best.pth"),
                    map_location=config.DEVICE, weights_only=True)
    )

    # Plot training curves
    _plot_training_curves(history, "Siamese Network", "siamese_training")

    # Save test pairs for evaluation
    torch.save({
        'test_pairs': test_pairs,
        'history': history
    }, os.path.join(config.MODEL_DIR, "siamese_metadata.pth"))

    print(f"  [OK] Siamese training complete. Best val loss: {best_val_loss:.4f}")
    return model, history


# ═══════════════════════════════════════════════════════════════════════
# 2. CNN BASELINE TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_cnn(patches=None, dataset=None):
    """
    Train the CNN baseline classifier.

    Returns:
        model: Trained InkClassifierCNN
        history: Dict with training metrics
    """
    print("\n" + "=" * 60)
    print("  TRAINING CNN BASELINE")
    print("=" * 60)

    # Generate patches if not provided
    if patches is None:
        if dataset is None:
            dataset = load_dataset()
        patches = generate_single_patches(dataset)

    # Split
    train_patches, val_patches, test_patches = split_data(patches)
    print(f"  Train: {len(train_patches)}, Val: {len(val_patches)}, Test: {len(test_patches)}")

    # Create data loaders
    train_dataset = SinglePatchDataset(train_patches)
    val_dataset = SinglePatchDataset(val_patches)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = InkClassifierCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"  Parameters: {count_parameters(model):,}")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        # === Train ===
        model.train()
        train_loss = AverageMeter()

        for patches_batch, labels in train_loader:
            patches_batch = patches_batch.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            logits = model(patches_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), patches_batch.size(0))

        # === Validate ===
        model.eval()
        val_loss = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for patches_batch, labels in val_loader:
                patches_batch = patches_batch.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                logits = model(patches_batch)
                loss = criterion(logits, labels)
                val_loss.update(loss.item(), patches_batch.size(0))

                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)
        scheduler.step(val_loss.avg)

        history['train_loss'].append(train_loss.avg)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch + 1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss.avg:.4f} | "
              f"Val Loss: {val_loss.avg:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            torch.save(model.state_dict(),
                       os.path.join(config.MODEL_DIR, "cnn_best.pth"))

        if early_stopping(val_loss.avg):
            print(f"  → Early stopping at epoch {epoch + 1}")
            break

    # Load best
    model.load_state_dict(
        torch.load(os.path.join(config.MODEL_DIR, "cnn_best.pth"),
                    map_location=config.DEVICE, weights_only=True)
    )

    _plot_training_curves(history, "CNN Baseline", "cnn_training")

    # Save test patches
    torch.save({
        'test_patches': test_patches,
        'history': history
    }, os.path.join(config.MODEL_DIR, "cnn_metadata.pth"))

    print(f"  [OK] CNN training complete. Best val loss: {best_val_loss:.4f}")
    return model, history


# ═══════════════════════════════════════════════════════════════════════
# 3. CLASSICAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_classical(patches=None, dataset=None):
    """
    Train classical ML models (SVM and Random Forest).

    Returns:
        models: Dict of trained classifiers
        results: Dict with training metrics
    """
    print("\n" + "=" * 60)
    print("  TRAINING CLASSICAL MODELS")
    print("=" * 60)

    if patches is None:
        if dataset is None:
            dataset = load_dataset()
        patches = generate_single_patches(dataset, augment=False)

    # Split
    train_patches, val_patches, test_patches = split_data(patches)

    X_train = [p for p, _ in train_patches]
    y_train = [l for _, l in train_patches]
    X_val = [p for p, _ in val_patches]
    y_val = [l for _, l in val_patches]
    X_test = [p for p, _ in test_patches]
    y_test = [l for _, l in test_patches]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    models = {}
    results = {}

    for method in ['svm', 'rf']:
        print(f"\n  --- {method.upper()} ---")
        clf = ClassicalInkClassifier(method=method)
        clf.fit(X_train, y_train)

        val_acc = clf.score(X_val, y_val)
        print(f"  Validation accuracy: {val_acc:.4f}")

        # Save model
        save_path = os.path.join(config.MODEL_DIR, f"classical_{method}.pkl")
        clf.save(save_path)

        models[method] = clf
        results[method] = {
            'val_acc': val_acc,
            'test_data': (X_test, y_test)
        }

    # Save test data for evaluation
    import pickle
    test_data_path = os.path.join(config.MODEL_DIR, "classical_test_data.pkl")
    with open(test_data_path, 'wb') as f:
        pickle.dump({'svm': (X_test, y_test), 'rf': (X_test, y_test)}, f)
    print(f"  [OK] Saved test data: {test_data_path}")

    print(f"\n  [OK] Classical training complete")
    return models, results


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _plot_training_curves(history, title, filename):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} — Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    fig.tight_layout()
    save_figure(fig, filename)


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Train All Models
# ═══════════════════════════════════════════════════════════════════════

def train_all():
    """Train all three model types and save results."""
    set_seed(42)
    start_time = time.time()

    print("\n" + "═" * 60)
    print("  CHEQUE FRAUD DETECTION — FULL TRAINING PIPELINE")
    print("═" * 60)

    # Load dataset
    dataset = load_dataset()

    # Generate data
    print("\n[1/4] Generating patch pairs...")
    pairs = generate_pairs(dataset, max_pairs_per_image=80)

    print("\n[2/4] Generating single patches...")
    singles = generate_single_patches(dataset, augment=True)

    # Train models
    print("\n[3/4] Training models...")
    siamese_model, siamese_history = train_siamese(pairs)
    cnn_model, cnn_history = train_cnn(singles)
    classical_models, classical_results = train_classical(singles)

    elapsed = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"  [OK] ALL TRAINING COMPLETE -- {elapsed:.1f}s total")
    print(f"{'═' * 60}")

    return {
        'siamese': (siamese_model, siamese_history),
        'cnn': (cnn_model, cnn_history),
        'classical': (classical_models, classical_results)
    }


if __name__ == "__main__":
    train_all()
