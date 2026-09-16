"""
Evaluation module for all three models.

Computes:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrices
  - Model comparison table
  - ROC curves & AUC
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import save_figure
from src.models.siamese_model import SiameseCNN, ContrastiveLoss
from src.models.cnn_model import InkClassifierCNN
from src.models.classical_model import ClassicalInkClassifier
from src.patch_generator import PatchPairDataset, SinglePatchDataset
from torch.utils.data import DataLoader


def evaluate_siamese(model=None, test_pairs=None):
    """
    Evaluate the Siamese network on test pairs.

    Returns:
        Dict with metrics
    """
    print("\n--- Evaluating Siamese Network ---")

    # Load model if not provided
    if model is None:
        model = SiameseCNN().to(config.DEVICE)
        model_path = os.path.join(config.MODEL_DIR, "siamese_best.pth")
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=config.DEVICE, weights_only=True)
            )
        else:
            print("  ⚠ No trained Siamese model found. Run train.py first.")
            return None

    # Load test data if not provided
    if test_pairs is None:
        meta_path = os.path.join(config.MODEL_DIR, "siamese_metadata.pth")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu', weights_only=False)
            test_pairs = meta['test_pairs']
        else:
            print("  ⚠ No test data found.")
            return None

    if not test_pairs:
        print("  ⚠ Empty test set.")
        return None

    test_dataset = PatchPairDataset(test_pairs)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_labels = []
    all_preds = []
    all_distances = []

    with torch.no_grad():
        for pa, pb, labels in test_loader:
            pa = pa.to(config.DEVICE)
            pb = pb.to(config.DEVICE)

            out1, out2 = model(pa, pb)
            distances = F.pairwise_distance(out1, out2)

            # Predict: same ink if distance < threshold
            predicted = (distances < config.CONTRASTIVE_MARGIN / 2).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_distances.extend(distances.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    distances = np.array(all_distances)

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred, "Siamese Network")

    # Compute similarity scores for ROC
    # Convert distances to similarity (higher = more similar)
    similarity = np.exp(-distances)
    metrics['scores'] = similarity
    metrics['y_true'] = y_true

    # Plot confusion matrix
    _plot_confusion_matrix(y_true, y_pred, "Siamese Network", "siamese_confusion")

    # Plot distance distribution
    _plot_distance_distribution(distances, y_true, "siamese_distance_dist")

    return metrics


def evaluate_cnn(model=None, test_patches=None):
    """
    Evaluate the CNN baseline on test patches.

    Returns:
        Dict with metrics
    """
    print("\n--- Evaluating CNN Baseline ---")

    if model is None:
        model = InkClassifierCNN().to(config.DEVICE)
        model_path = os.path.join(config.MODEL_DIR, "cnn_best.pth")
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=config.DEVICE, weights_only=True)
            )
        else:
            print("  ⚠ No trained CNN model found. Run train.py first.")
            return None

    if test_patches is None:
        meta_path = os.path.join(config.MODEL_DIR, "cnn_metadata.pth")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu', weights_only=False)
            test_patches = meta['test_patches']
        else:
            print("  ⚠ No test data found.")
            return None

    if not test_patches:
        print("  ⚠ Empty test set.")
        return None

    test_dataset = SinglePatchDataset(test_patches)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(config.DEVICE)
            logits = model(patches)
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)

    metrics = _compute_metrics(y_true, y_pred, "CNN Baseline")
    metrics['scores'] = y_scores
    metrics['y_true'] = y_true

    _plot_confusion_matrix(y_true, y_pred, "CNN Baseline", "cnn_confusion")

    return metrics


def evaluate_classical(models=None, results=None):
    """
    Evaluate classical ML models.

    Returns:
        Dict with metrics for each model type
    """
    print("\n--- Evaluating Classical Models ---")

    all_metrics = {}

    for method in ['svm', 'rf']:
        print(f"\n  {method.upper()}:")

        # Get model
        if models and method in models:
            clf = models[method]
        else:
            clf = ClassicalInkClassifier(method=method)
            model_path = os.path.join(config.MODEL_DIR, f"classical_{method}.pkl")
            if os.path.exists(model_path):
                clf.load(model_path)
            else:
                print(f"  ⚠ No trained {method.upper()} model found.")
                continue

        # Get test data
        if results and method in results:
            X_test, y_test = results[method]['test_data']
        else:
            # Try loading from saved file
            import pickle
            test_data_path = os.path.join(config.MODEL_DIR, "classical_test_data.pkl")
            if os.path.exists(test_data_path):
                with open(test_data_path, 'rb') as f:
                    test_data = pickle.load(f)
                if method in test_data:
                    X_test, y_test = test_data[method]
                else:
                    print(f"  ⚠ No test data for {method.upper()}.")
                    continue
            else:
                print(f"  ⚠ No test data for {method.upper()}.")
                continue

        if not X_test:
            print(f"  ⚠ Empty test set for {method.upper()}.")
            continue

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        y_true = np.array(y_test)
        y_pred = np.array(y_pred)

        name = f"Classical {method.upper()}"
        metrics = _compute_metrics(y_true, y_pred, name)
        metrics['scores'] = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        metrics['y_true'] = y_true

        _plot_confusion_matrix(y_true, y_pred, name, f"classical_{method}_confusion")

        all_metrics[method] = metrics

    return all_metrics


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def compare_models(siamese_metrics, cnn_metrics, classical_metrics):
    """
    Generate comparison table and visualization.
    """
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    # Collect all metrics
    all_models = {}
    if siamese_metrics:
        all_models['Siamese'] = siamese_metrics
    if cnn_metrics:
        all_models['CNN'] = cnn_metrics
    if classical_metrics:
        for method, m in classical_metrics.items():
            all_models[f'{method.upper()}'] = m

    if not all_models:
        print("  No models to compare.")
        return

    # Print comparison table
    print(f"\n  {'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "-" * 55)
    for name, m in all_models.items():
        print(f"  {name:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models_list = list(all_models.keys())
    x = np.arange(len(models_list))
    width = 0.2

    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        values = [all_models[m][metric] for m in models_list]
        ax.bar(x + i * width, values, width, label=metric.capitalize(),
               color=color, alpha=0.85)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, "model_comparison")

    # ROC Curves
    _plot_roc_curves(all_models)

    return all_models


def _compute_metrics(y_true, y_pred, model_name):
    """Compute all classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print(f"  {model_name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
    }


def _plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Different Ink', 'Same Ink'],
                yticklabels=['Different Ink', 'Same Ink'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {title}')

    save_figure(fig, filename)


def _plot_distance_distribution(distances, labels, filename):
    """Plot the distribution of Siamese distances for same vs different ink."""
    fig, ax = plt.subplots(figsize=(8, 5))

    same_mask = labels == 1
    diff_mask = labels == 0

    if same_mask.any():
        ax.hist(distances[same_mask], bins=30, alpha=0.6, color='#4CAF50',
                label='Same Ink', density=True)
    if diff_mask.any():
        ax.hist(distances[diff_mask], bins=30, alpha=0.6, color='#E91E63',
                label='Different Ink', density=True)

    ax.axvline(x=config.CONTRASTIVE_MARGIN / 2, color='red',
               linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Density')
    ax.set_title('Siamese Network — Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, filename)


def _plot_roc_curves(all_models):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for i, (name, metrics) in enumerate(all_models.items()):
        if 'scores' in metrics and 'y_true' in metrics:
            try:
                fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['scores'])
                roc_auc = auc(fpr, tpr)
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception:
                pass

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    save_figure(fig, "roc_curves")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all():
    """Run full evaluation pipeline."""
    print("\n" + "═" * 60)
    print("  CHEQUE FRAUD DETECTION — EVALUATION")
    print("═" * 60)

    siamese_metrics = evaluate_siamese()
    cnn_metrics = evaluate_cnn()
    classical_metrics = evaluate_classical()

    comparison = compare_models(siamese_metrics, cnn_metrics, classical_metrics)

    print("\n  [OK] Evaluation complete. Figures saved to:", config.FIGURES_DIR)
    return comparison


if __name__ == "__main__":
    evaluate_all()
